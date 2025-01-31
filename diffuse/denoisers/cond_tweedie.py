from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple

import einops
import jax
import jax.experimental
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, PRNGKeyArray
from blackjax.smc.resampling import stratified
import chex

from diffuse.integrator.base import Integrator, IntegratorState
from diffuse.diffusion.sde import SDE, SDEState
from diffuse.base_forward_model import ForwardModel, MeasurementState
from diffuse.utils.plotting import sigle_plot, plot_lines

def _vmapper(fn, type):
    def _set_axes(path, value):
        # Vectorize only particles and rng_key fields
        if any(field in str(path) for field in ["position", "rng_key", "weights"]):
            return 0
        return None

    # Create tree with selective vectorization
    in_axes = jax.tree_util.tree_map_with_path(_set_axes, type)

    return jax.pmap(
        jax.vmap(fn, in_axes=(in_axes, None)),
        axis_name='devices',
        in_axes=(in_axes, None),
        static_broadcasted_argnums=1
    )

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
class CondTweedie:
    """Conditional denoiser using Diffusion Posterior Sampling with Tweedie's formula"""

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
        weights = jnp.log(jnp.ones(n_particles) / n_particles)
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
            posterior = self.posterior_logpdf(
                key, measurement_state.y, measurement_state.mask_history
            )
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
        # Shard the state across devices
        num_devices = jax.device_count()
        state = jax.tree_map(
            lambda x: x.reshape((num_devices, -1, *x.shape[1:])) if len(x.shape) > 0 else x,
            state
        )

        # vmap over position, rng_key, weights
        state_next = _vmapper(self.step, state)(state, score)

        # Reshape back to original dimensions
        state_next = jax.tree_map(
            lambda x: x.reshape((-1, *x.shape[2:])) if len(x.shape) > 0 else x,
            state_next
        )

        integrator_state_next = state_next.integrator_state
        integrator_state, weights = state.integrator_state, state.weights

        weights = weights.reshape((-1,))
        return CondDenoiserState(integrator_state_next, weights)


    def posterior_logpdf(
        self, rng_key: PRNGKeyArray, y_meas: Array, design_mask: Array
    ):
        # Using Tweedie's formula for posterior sampling
        def _posterior_logpdf(x, t):

            # Apply Tweedie's formula to get denoised prediction
            denoised = self.sde.tweedie(SDEState(x, t), self.score).position

            # Compute residual: (y - AE[X_0|X_t])
            v = self.forward_model.grad_logprob_y(denoised, y_meas, design_mask)

            # Compute (I + ∇score)ᵀv using forward-mode autodiff
            def score_fn(x_):
                return self.score(x_, t)

            score_val, tangents = jax.jvp(score_fn, (x,), (v,))

            guidance = (v - tangents) #* alpha_t

            return guidance + score_val

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
        mask = self.forward_model.make(design)

        def _pooled_posterior_logpdf(x, t):

            # Apply Tweedie's formula
            denoised = self.sde.tweedie(SDEState(x, t), self.score).position

            # Compute guidance using denoised prediction for contrast samples
            guidance = jax.vmap(
                self.forward_model.grad_logprob_y,
                in_axes=(None, 0, None)
            )(denoised, y_cntrst, mask)

            # Add past contribution using Tweedie
            past_contribution = self.posterior_logpdf(rng_key2, y_past, mask_history)

            return guidance.mean(axis=0) + past_contribution(x, t)

        return _pooled_posterior_logpdf


    def __pooled_posterior_logpdf(
        self,
        rng_key: PRNGKeyArray,
        y_cntrst: Array,
        y_past: Array,
        design: Array,
        mask_history: Array,
    ):
        rng_key1, rng_key2 = jax.random.split(rng_key)
        mask = self.forward_model.make(design)

        def _pooled_posterior_logpdf(x, t):

            # Apply Tweedie's formula
            denoised = self.sde.tweedie(SDEState(x, t), self.score).position

            # Compute guidance using denoised prediction for contrast samples
            residual = jax.vmap(
                self.forward_model.grad_logprob_y,
                in_axes=(None, 0, None)
            )(denoised, y_cntrst, mask)

            # Compute (I + ∇score)ᵀv using forward-mode autodiff
            def score_fn(x_):
                return self.score(x_, t)
            score_val, tangents = jax.vmap(lambda a,b: jax.jvp(score_fn, (a,), (b,)), in_axes=(None, 0))(x, residual)
            guidance = (residual - tangents) #* alpha_t

            # Apply VJP with the residual
            #guidance = jax.vmap(vjp_score)(residual)

            # Add past contribution using Tweedie
            past_contribution = self.posterior_logpdf(rng_key2, y_past, mask_history)
            return guidance.mean(axis=0) + score_val.mean(axis=0) + past_contribution(x, t)

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
        res = alpha * y + jnp.sqrt(beta) * einops.einsum(mask, rndm, "... , ... c -> ... c")
        #self.forward_model.measure_from_mask(    mask, rndm)
        return SDEState(res, ts)

    def _resampling(
        self, position: Array, log_weights: Array, rng_key: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        """Resample particles based on weights if effective sample size is in target range"""
        _norm = jax.scipy.special.logsumexp(log_weights, axis=0)
        #weights = jnp.exp(log_weights - _norm)
        weights = jax.nn.softmax(log_weights, axis=0)

        ess_val = ess(log_weights)
        n_particles = position.shape[0]
        key_resample = jax.random.split(rng_key)[1]
        idx = stratified(key_resample, weights, n_particles)

        # jax.debug.print("ess_val: {}", ess_val/n_particles)
        return jax.lax.cond(
            (ess_val < 0.5 * n_particles), #& (ess_val > 0.2 * n_particles),
            #(ess_val < 0.4 * n_particles) & (ess_val > 0.2 * n_particles),
            #lambda x: (x[idx], _normalize_log_weights(log_weights[idx])),
            lambda x: (x, _normalize_log_weights(log_weights)),
            lambda x: (x, _normalize_log_weights(log_weights)),
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

def _normalize_log_weights(log_weights: Array) -> Array:
    return jax.nn.log_softmax(log_weights, axis=0)
