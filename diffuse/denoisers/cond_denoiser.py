from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple

import einops
import jax
import jax.experimental
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, PRNGKeyArray
from blackjax.smc.resampling import stratified

from diffuse.integrator.base import Integrator, IntegratorState
from diffuse.diffusion.sde import SDE, SDEState
from diffuse.base_forward_model import ForwardModel, MeasurementState
from diffuse.utils.plotting import sigle_plot

def _vmapper(fn, type):
    def _set_axes(path, value):
        # Vectorize only particles and rng_key fields
        if any(field in str(path) for field in ["position", "rng_key", "weights"]):
            return 0
        return None

    # Create tree with selective vectorization
    in_axes = jax.tree_util.tree_map_with_path(_set_axes, type)
    return jax.vmap(fn, in_axes=(in_axes, None))

def ess(log_weights: Array) -> float:
    return jnp.exp(log_ess(log_weights))


def log_ess(log_weights: Array) -> float:
    """Compute the effective sample size.

    Parameters
    ----------
    log_weights: 1D Array
        log-weights of the sample

    Returns
    -------
    log_ess: float
        The logarithm of the effective sample size

    """
    return 2 * jsp.special.logsumexp(log_weights) - jsp.special.logsumexp(
        2 * log_weights
    )


class CondDenoiserState(NamedTuple):
    integrator_state: IntegratorState
    weights: Array


@dataclass
class CondDenoiser:
    """Conditional denoiser for conditional diffusion"""

    integrator: Integrator
    # logpdf: Callable[[SDEState, Array], Array]  # x -> t -> logpdf(x, t)
    sde: SDE
    score: Callable[[Array, float], Array]  # x -> t -> score(x, t)
    forward_model: ForwardModel
    _resample: bool = False

    def init(
        self, position: Array, rng_key: PRNGKeyArray, dt: float
    ) -> CondDenoiserState:
        n_particles = position.shape[0]
        weights = jnp.ones(n_particles) / n_particles
        keys = jax.random.split(rng_key, n_particles)
        integrator_state = self.integrator.init(
            position, keys, jnp.array(0.0), jnp.array(dt)
        )
        return CondDenoiserState(integrator_state, weights)

    def generate(self, rng_key: PRNGKeyArray, forward_model: ForwardModel, measurement_state: MeasurementState, design: Array, n_steps: int, n_particles: int):
        dt = self.sde.tf / n_steps

        key, subkey = jax.random.split(rng_key)
        cntrst_thetas = jax.random.normal(subkey, (n_particles, *measurement_state.y.shape))

        key, subkey = jax.random.split(key)
        state = self.init(cntrst_thetas, subkey, dt)


        def body_fun(state: CondDenoiserState, key: PRNGKeyArray):
            posterior = self.posterior_logpdf(key, measurement_state.y, forward_model.make(design))
            state_next = self.batch_step(key, state, posterior, measurement_state)
            return _fix_time(state_next), state_next.integrator_state.position

        keys = jax.random.split(key, n_steps)
        return jax.lax.scan(body_fun, state, keys)

    def step(
        self, state: CondDenoiserState, score: Callable[[Array, float], Array]
    ) -> CondDenoiserState:
        r"""
        sample p(\theta_t-1 | \theta_t, \y_t-1, \xi)
        """
        integrator_state, weights = state
        integrator_state_next = self.integrator(integrator_state, score)

        return CondDenoiserState(integrator_state_next, weights)

    def batch_step(
        self, rng_key: PRNGKeyArray, state: CondDenoiserState, score: Callable[[Array, float], Array], measurement_state: MeasurementState
    ) -> CondDenoiserState:
        r"""
        batch step for conditional diffusion
        """
        # vmap over position, rng_key, weights
        state_next = _vmapper(self.step, state)(state, score)
        integrator_state_next = state_next.integrator_state
        integrator_state, weights = state.integrator_state, state.weights

        mask = measurement_state.mask_history
        y = measurement_state.y
        t = integrator_state.t
        tf = self.sde.tf

        # resample if necessary
        if self._resample:
            position = integrator_state_next.position
            weights = state_next.weights
            y_noised = self.y_noiser(mask, rng_key, SDEState(y, 0), tf-t).position
            A_theta = self.forward_model.measure_from_mask(mask, position)

            alpha_t = jnp.exp(self.sde.beta.integrate(0.0, t))
            #jax.experimental.io_callback(sigle_plot, None, y_noised)
            logsprobs = jax.scipy.stats.norm.logpdf(y_noised, A_theta, alpha_t)
            logsprobs = self.forward_model.measure_from_mask(mask, logsprobs)
            logsprobs = einops.reduce(logsprobs, "t ... -> t ", "sum")


            position, weights = self._resampling(position, logsprobs, rng_key)

        return CondDenoiserState(integrator_state_next, weights)

    def posterior_logpdf(
        self, rng_key: PRNGKeyArray, y_meas: Array, design_mask: Array
    ):
        # will be called backward in time
        # with t = tf - t, t from 0 to tf
        def _posterior_logpdf(x, t):
            tf = self.sde.tf
            y_t = self.y_noiser(
                design_mask, rng_key, SDEState(y_meas, 0), t
            ).position
            alpha_t = jnp.exp(self.sde.beta.integrate(0.0, tf - t))
            #guidance = jax.grad(self.forward_model.logprob_y)(x, y_t, design) #/ alpha_t
            guidance = self.forward_model.grad_logprob_y(x, y_t, design_mask) / alpha_t
            return guidance + self.score(x, t)

        return _posterior_logpdf


    def pooled_posterior_logpdf(
        self,
        rng_key: PRNGKeyArray,
        y_cntrst: Array,
        y_past: Array,
        design: Array,
        mask_history: Array,
    ):
        rng_key1, rng_key2 = jax.random.split(rng_key)
        vec_noiser = jax.vmap(
            self.y_noiser, in_axes=(None, None, SDEState(0, None), None)
        )
        mask = self.forward_model.make(design)

        # will be called backward in time
        # with t = tf - t, t from 0 to tf
        def _pooled_posterior_logpdf(x, t):
            y_t = vec_noiser(
                mask, rng_key1, SDEState(y_cntrst, 0), t
            ).position
            tf = self.sde.tf
            alpha_t = jnp.exp(self.sde.beta.integrate(0.0, tf - t))
            guidance = jax.vmap(
                #jax.grad(self.forward_model.logprob_y), in_axes=(None, 0, None)
                self.forward_model.grad_logprob_y, in_axes=(None, 0, None)
            )(x, y_t, mask) / alpha_t
            past_contribution = self.posterior_logpdf(rng_key2, y_past, mask_history)
            # import pdb; pdb.set_trace()
            # jax.debug.print("guidance: {}", guidance)
            return guidance.mean(axis=0) + past_contribution(x, t)

        return _pooled_posterior_logpdf

    def y_noiser(
        self, mask: Array, key: PRNGKeyArray, state: SDEState, ts: float
    ) -> SDEState:
        r"""
        Generate y^{(t)} = \sqrt{\bar{\alpha}_t} y + \sqrt{1-\bar{\alpha}_t} A_\xi \epsilon
        """
        y, t = state

        int_b = self.sde.beta.integrate(ts, t)
        alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)

        rndm = jax.random.normal(key, y.shape)
        res = alpha * y + jnp.sqrt(beta) * self.forward_model.measure_from_mask(
            mask, rndm
        )
        return SDEState(res, ts)

    def _resampling(
        self, position: Array, log_weights: Array, rng_key: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        """Resample particles based on weights if effective sample size is in target range"""
        _norm = jax.scipy.special.logsumexp(log_weights, axis=0)
        log_weights = log_weights - _norm
        weights = jnp.exp(log_weights)

        ess_val = ess(log_weights)
        n_particles = position.shape[0]
        key_resample = jax.random.split(rng_key)[1]
        idx = stratified(key_resample, weights, n_particles)

        # jax.debug.print("ess_val: {}", ess_val/n_particles)
        return jax.lax.cond(
            (ess_val < 0.9 * n_particles) & (ess_val > 0.1 * n_particles),
            lambda x: (x[idx], log_weights[idx]),
            lambda x: (x, log_weights),
            position,
        )


def _fix_time(denoiser_state: CondDenoiserState):
    # Create new integrator states with fixed time
    new_denoiser_integrator = denoiser_state.integrator_state._replace(
        t=denoiser_state.integrator_state.t[0],
        dt=denoiser_state.integrator_state.dt[0]
    )

    # Return new denoiser states with updated integrator states
    return denoiser_state._replace(integrator_state=new_denoiser_integrator)
