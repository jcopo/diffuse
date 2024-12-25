from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from blackjax.smc.resampling import stratified

from diffuse.integrator.base import Integrator, IntegratorState
from diffuse.diffusion.sde import SDE, SDEState
from diffuse.base_forward_model import ForwardModel


class CondDenoiserState(NamedTuple):
    integrator_state: IntegratorState
    weights: Array


@dataclass
class CondDenoiser:
    """Conditional denoiser for conditional diffusion"""

    integrator: Integrator
    logpdf: Callable[[SDEState, Array], Array]  # x -> t -> logpdf(x, t)
    sde: SDE
    score: Callable[[Array, float], Array]  # x -> t -> score(x, t)
    _resample: bool

    def init(
        self, position: Array, rng_key: PRNGKeyArray, dt: float
    ) -> CondDenoiserState:
        weights = jnp.ones_like(position) / position.sum()
        integrator_state = self.integrator.init(position, rng_key, 0.0, dt)
        return CondDenoiserState(integrator_state, weights)

    def step(
        self,
        state: CondDenoiserState,
        rng_key: PRNGKeyArray,
        y_meas: Array,
        score_likelihood: Callable[[Array, Array], Array],
    ) -> CondDenoiserState:
        r"""
        sample p(\theta_t-1 | \theta_t, \y_t-1, \xi)
        """
        integrator_state, weights = state
        def score_cond(x, t):
            y_t = self.sde.path()#TODO)
            return self.score(x, t) + score_likelihood(x, y_t)
        integrator_state_next = self.integrator(integrator_state, score_cond)

        position = integrator_state_next.position
        if self._resample:
            weights = self.logpdf(integrator_state_next, integrator_state)
            position, weights = self._resampling(position, weights, rng_key)

        return CondDenoiserState(integrator_state_next, weights)

    def posterior_logpdf(self, rng_key:PRNGKeyArray, t:float, y_meas:Array, design:Array, mask:ForwardModel):
        y_t = self.y_noiser(mask, rng_key, SDEState(y_meas, 0), t)
        def posterior_logpdf(x, t):
            guidance = jax.grad(mask.logprob_y)(x, y_t, design)
            return guidance + self.score(x, t)

        return posterior_logpdf

    def pooled_posterior_logpdf(self, rng_key:PRNGKeyArray, t:float, y_cntrst:Array, y_past:Array, design:Array, mask:ForwardModel):
        vec_noiser = jax.vmap(self.y_noiser, in_axes=(None, None, SDEState(0, None), None))
        y_t = vec_noiser(mask, rng_key, SDEState(y_cntrst, 0), t)
        def pooled_posterior_logpdf(x, t):
            guidance = jax.vmap(jax.grad(mask.logprob_y), in_axes=(None, 0, None))(x, y_t, design)
            past_contribution = self.posterior_logpdf(x, t, y_past, design, mask)
            return guidance.mean(axis=0) + past_contribution(x, t)

        return pooled_posterior_logpdf

    def y_noiser(
        self, mask: Array, key: PRNGKeyArray, state: SDEState, ts: float
    ) -> SDEState:
        r"""
        Generate x_ts | x_t ~ N(.| exp(-0.5 \int_ts^t \beta(s) ds) x_0, 1 - exp(-\int_ts^t \beta(s) ds))
        """
        x, t = state

        int_b = self.sde.beta.integrate(ts, t)
        alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)

        rndm = jax.random.normal(key, x.shape)
        res = alpha * x + jnp.sqrt(beta) * mask.measure_from_mask(mask, rndm)
        return SDEState(res, ts)

    def _resampling(
        self, position: Array, weights: Array, rng_key: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        """Resample particles based on weights if effective sample size is in target range"""
        _norm = jax.scipy.special.logsumexp(weights, axis=0)
        log_weights = weights - _norm
        weights = jnp.exp(log_weights)

        ess_val = ess(log_weights)
        n_particles = position.shape[0]
        key_resample = jax.random.split(rng_key)[1]
        idx = stratified(key_resample, weights, n_particles)

        return jax.lax.cond(
            (ess_val < 0.6 * n_particles) & (ess_val > 0.2 * n_particles),
            lambda x: (x[idx], weights[idx]),
            lambda x: (x, weights),
            position,
        )
