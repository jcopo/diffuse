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
from diffuse.utils.mapping import pmapper


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
class CondTweediePP:
    """Conditional denoiser using Diffusion Posterior Sampling with Tweedie's formula"""

    integrator: Integrator
    # logpdf: Callable[[SDEState, Array], Array]  # x -> t -> logpdf(x, t)
    sde: SDE
    score: Callable[[Array, float], Array]  # x -> t -> score(x, t)
    forward_model: ForwardModel
    _resample: bool = False
    pooled_jvp: bool = False

    def init(
        self, position: Array, rng_key: PRNGKeyArray, dt: float
    ) -> CondDenoiserState:
        n_particles = position.shape[0]
        weights = jnp.log(jnp.ones(n_particles) / n_particles)
        keys = jax.random.split(rng_key, n_particles)
        integrator_state = self.integrator.init(
            position, keys, jnp.zeros(n_particles), dt + jnp.zeros(n_particles)
        )
        return CondDenoiserState(integrator_state, weights)

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
        integrator_state, weights = state
        integrator_state_next = self.integrator(integrator_state, score)

        return CondDenoiserState(integrator_state_next, weights)

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
        # Pmap step
        state_next: CondDenoiserState = pmapper(self.step, state, score=score)

        forward_time = self.sde.tf - state_next.integrator_state.t
        state_forward = state_next.integrator_state._replace(t=forward_time)

        # Pmap tweedie
        denoised_state: IntegratorState = pmapper(
            self.sde.tweedie, state_forward, score=self.score, batch_size=16
        )

        denoised = denoised_state.position

        diff = (
            self.forward_model.measure_from_mask(
                measurement_state.mask_history, denoised
            )
            - measurement_state.y
        )
        abs_diff = jnp.abs(
            diff[..., 0] + 1j * diff[..., 1]
        )  # TODO: think about generalizing this

        log_weights = jax.scipy.stats.norm.logpdf(
            abs_diff, 0, self.forward_model.sigma_prob
        )
        log_weights = einops.einsum(
            measurement_state.mask_history, log_weights, "..., b ... -> b"
        )

        integrator_state_next = state_next.integrator_state
        _norm = jax.scipy.special.logsumexp(log_weights, axis=0)

        log_weights = log_weights.reshape((-1,)) - _norm

        if self._resample:
            position, log_weights = self._resampling(
                integrator_state_next.position, log_weights, rng_key
            )
            integrator_state_next = integrator_state_next._replace(position=position)

        return CondDenoiserState(integrator_state_next, log_weights)

    def posterior_logpdf(
        self, rng_key: PRNGKeyArray, y_meas: Array, design_mask: Array
    ):
        def _posterior_logpdf(x, t):
            # Compute alpha, beta
            int_b = jnp.squeeze(self.sde.beta.integrate(t, self.sde.beta.t0))
            alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)
            scale = beta / alpha

            # Compute (A C A^T + \sigma^2 I) v efficiently
            def tweedie_fn(x_):
                return self.sde.tweedie(SDEState(x_, t), self.score).position

            def efficient(v):
                restored_v = self.forward_model.restore_from_mask(
                    design_mask, jnp.zeros_like(x), v
                )
                _, tangents = jax.jvp(tweedie_fn, (x,), (restored_v,))
                measured_tangents = self.forward_model.measure_from_mask(
                    design_mask, tangents
                )
                return scale * measured_tangents + self.forward_model.sigma_prob * v

            # Compute residual: (y - AE[X_0|X_t])
            score_val = self.score(
                x, t
            )  # Don't use tweedie function to avoid computing the score twice
            denoised = (x + beta * score_val) / alpha
            b = y_meas - self.forward_model.measure_from_mask(design_mask, denoised)

            res, _ = jax.scipy.sparse.linalg.cg(efficient, b, maxiter=2)
            restored_res = self.forward_model.restore_from_mask(
                design_mask, jnp.zeros_like(x), res
            )
            _, guidance = jax.jvp(tweedie_fn, (x,), (restored_res,))

            return guidance + score_val

        return _posterior_logpdf

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

        state_next: CondDenoiserState = pmapper(self.step, state, score)

        integrator_state_next = state_next.integrator_state
        log_weights = state_next.weights

        return CondDenoiserState(integrator_state_next, log_weights)

    def pooled_posterior_logpdf(
        self,
        rng_key: PRNGKeyArray,
        y_cntrst: Array,
        y_past: Array,
        design: Array,
        mask_history: Array,
    ):
        _, rng_key2 = jax.random.split(rng_key)
        mask = self.forward_model.make(design)

        def _pooled_posterior_logpdf(x, t):
            # Compute alpha, beta
            int_b = jnp.squeeze(self.sde.beta.integrate(t, self.sde.beta.t0))
            alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)
            scale = beta / alpha

            # Compute (A C A^T + \sigma^2 I) v efficiently
            def tweedie_fn(x_):
                return self.sde.tweedie(SDEState(x_, t), self.score).position

            def efficient(v):
                restored_v = self.forward_model.restore_from_mask(
                    mask, jnp.zeros_like(x), v
                )
                _, tangents = jax.jvp(tweedie_fn, (x,), (restored_v,))
                measured_tangents = self.forward_model.measure_from_mask(mask, tangents)
                return scale * measured_tangents + self.forward_model.sigma_prob * v

            # Compute residual: (y - AE[X_0|X_t])
            score_val = self.score(
                x, t
            )  # Don't use tweedie function to avoid computing the score twice
            denoised = (x + beta * score_val) / alpha

            b = jnp.mean(
                y_cntrst - self.forward_model.measure_from_mask(mask, denoised), axis=0
            )

            res, _ = jax.scipy.sparse.linalg.cg(efficient, b, maxiter=2)
            restored_res = self.forward_model.restore_from_mask(
                mask, jnp.zeros_like(x), res
            )
            _, guidance = jax.jvp(tweedie_fn, (x,), (restored_res,))

            past_contribution = self.posterior_logpdf(rng_key2, y_past, mask_history)
            return guidance + past_contribution(x, t)

        return _pooled_posterior_logpdf

    def _resampling(
        self, position: Array, log_weights: Array, rng_key: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        """Resample particles based on weights if effective sample size is in target range"""
        weights = jax.nn.softmax(log_weights, axis=0)

        ess_val = ess(log_weights)
        n_particles = position.shape[0]
        key_resample = jax.random.split(rng_key)[1]
        idx = stratified(key_resample, weights, n_particles)

        return jax.lax.cond(
            (ess_val < 0.5 * n_particles) & (ess_val > 0.2 * n_particles),
            # (ess_val < 0.4 * n_particles) & (ess_val > 0.2 * n_particles),
            lambda x: (x[idx], _normalize_log_weights(log_weights[idx])),
            lambda x: (x, _normalize_log_weights(log_weights)),
            position,
        )


def _normalize_log_weights(log_weights: Array) -> Array:
    return jax.nn.log_softmax(log_weights, axis=0)
