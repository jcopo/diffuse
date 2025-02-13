from abc import abstractmethod
from typing import Callable, Optional
from jaxtyping import Array, PRNGKeyArray
from diffuse.integrator.base import IntegratorState
import jax.numpy as jnp
from dataclasses import dataclass
import jax
from diffuse.diffusion.sde import SDE
from diffuse.integrator.base import Integrator
from diffuse.base_forward_model import ForwardModel, MeasurementState
from diffuse.utils.mapping import pmapper
import einops
from diffuse.denoisers.utils import ess, normalize_log_weights
from blackjax.smc.resampling import stratified
from typing import Tuple
from diffuse.denoisers.base import DenoiserState, BaseDenoiser


class CondDenoiserState(DenoiserState):
    """Conditional denoiser state"""

    log_weights: Array = jnp.array([])


@dataclass
class CondDenoiser(BaseDenoiser):
    integrator: Integrator
    sde: SDE
    score: Callable[[Array, float], Array]
    forward_model: ForwardModel
    resample: Optional[bool] = False
    ess_low: Optional[float] = 0.2
    ess_high: Optional[float] = 0.5

    def init(
        self, position: Array, rng_key: PRNGKeyArray, dt: float
    ) -> CondDenoiserState:
        n_particles = position.shape[0]
        log_weights = jnp.log(jnp.ones(n_particles) / n_particles)
        keys = jax.random.split(rng_key, n_particles)
        integrator_state = self.integrator.init(
            position, keys, jnp.zeros(n_particles), dt + jnp.zeros(n_particles)
        )

        return CondDenoiserState(integrator_state, log_weights)

    def generate(
        self,
        rng_key: PRNGKeyArray,
        measurement_state: MeasurementState,
        n_steps: int,
        n_particles: int,
    ):
        dt = self.sde.tf / n_steps

        key, subkey = jax.random.split(rng_key)
        cntrst_thetas = jax.random.normal(
            subkey, (n_particles, *measurement_state.y.shape)
        )

        key, subkey = jax.random.split(key)
        state = self.init(cntrst_thetas, subkey, dt)

        def body_fun(state: CondDenoiserState, key: PRNGKeyArray):
            posterior = self.posterior_logpdf(
                key, measurement_state.y, measurement_state.mask_history
            )
            state_next = self.batch_step(key, state, posterior, measurement_state)
            return state_next, None

        keys = jax.random.split(key, n_steps)
        return jax.lax.scan(body_fun, state, keys)

    def step(
        self, state: CondDenoiserState, score: Callable[[Array, float], Array]
    ) -> CondDenoiserState:
        r"""
        sample p(\theta_t-1 | \theta_t, \y_t-1, \xi)
        """
        integrator_state, log_weights = state
        integrator_state_next = self.integrator(integrator_state, score)

        return CondDenoiserState(integrator_state_next, log_weights)

    def batch_step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        score: Callable[[Array, float], Array],
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        r"""
        batch step for conditional diffusion
        """

        state_next = pmapper(self.step, state, score=score)

        if self.resample:
            state_next = self.resampler(state_next, measurement_state, rng_key)

        return state_next

    def resampler(
        self,
        state_next: CondDenoiserState,
        measurement_state: MeasurementState,
        rng_key: PRNGKeyArray,
    ) -> Tuple[Array, Array]:
        forward_time = self.sde.tf - state_next.integrator_state.t
        state_forward = state_next.integrator_state._replace(t=forward_time)

        denoised_state = pmapper(
            self.sde.tweedie, state_forward, score=self.score, batch_size=16
        )
        diff = (
            self.forward_model.measure_from_mask(
                measurement_state.mask_history, denoised_state.position
            )
            - measurement_state.y
        )
        abs_diff = jnp.abs(diff[..., 0] + 1j * diff[..., 1])
        log_weights = jax.scipy.stats.norm.logpdf(
            abs_diff, 0, self.forward_model.sigma_prob
        )
        log_weights = einops.einsum(
            measurement_state.mask_history, log_weights, "..., b ... -> b"
        )
        _norm = jax.scipy.special.logsumexp(log_weights, axis=0)
        log_weights = log_weights.reshape((-1,)) - _norm
        position, log_weights = self._resample(
            state_next.integrator_state.position, log_weights, rng_key
        )

        integrator_state_next = IntegratorState(position, forward_time)
        state_next = CondDenoiserState(integrator_state_next, log_weights)
        return state_next

    def _resample(
        self, position: Array, log_weights: Array, rng_key: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        weights = jax.nn.softmax(log_weights, axis=0)
        ess_val = ess(log_weights)
        n_particles = position.shape[0]
        idx = stratified(rng_key, weights, n_particles)

        return jax.lax.cond(
            (ess_val < self.ess_high * n_particles)
            & (ess_val > self.ess_low * n_particles),
            lambda x: (x[idx], normalize_log_weights(log_weights[idx])),
            lambda x: (x, normalize_log_weights(log_weights)),
            position,
        )

    def batch_step_pooled(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        score: Callable[[Array, float], Array],
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        r"""
        batch step for conditional diffusion
        """
        state_next: CondDenoiserState = pmapper(self.step, state, score=score)
        return state_next

    @abstractmethod
    def posterior_logpdf(
        self, rng_key: PRNGKeyArray, y_meas: Array, design_mask: Array
    ) -> Callable[[Array, float], Array]:
        pass

    @abstractmethod
    def pooled_posterior_logpdf(
        self,
        rng_key: PRNGKeyArray,
        y_cntrst: Array,
        y_past: Array,
        design: Array,
        mask_history: Array,
    ) -> Callable[[Array, float], Array]:
        pass
