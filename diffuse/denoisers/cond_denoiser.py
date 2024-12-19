
from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from blackjax.smc.resampling import stratified

from diffuse.integrator.base import Integrator, IntegratorState
from diffuse.diffusion.sde import SDE, SDEState


class CondDenoiserState(NamedTuple):
    integrator_state: IntegratorState
    weights: Array


@dataclass
class CondDenoiser:
    """Conditional denoiser for conditional diffusion"""
    integrator: Integrator
    logpdf: Callable[[SDEState, Array], Array] # x -> t -> logpdf(x, t)
    sde: SDE
    score: Callable[[Array, float], Array] # x -> t -> score(x, t)
    _resample: bool

    def init(self, position: Array, rng_key: PRNGKeyArray, dt: float) -> CondDenoiserState:
        weights = jnp.ones_like(position) / position.sum()
        integrator_state = self.integrator.init(position, rng_key, 0., dt)
        return CondDenoiserState(integrator_state, weights)

    def step(
        self,
        state: CondDenoiserState,
        rng_key: PRNGKeyArray,
    ) -> CondDenoiserState:
        r"""
        sample p(\theta_t-1 | \theta_t, \y_t-1, \xi)
        """
        integrator_state, weights = state
        integrator_state_next = self.integrator(integrator_state, self.score)

        position = integrator_state_next.position
        if self._resample:
            weights = self.logpdf(integrator_state_next, integrator_state)
            position, weights = self._resampling(position, weights, rng_key)

        return CondDenoiserState(integrator_state_next, weights)

    def _resampling(self, position: Array, weights: Array, rng_key: PRNGKeyArray) -> Tuple[Array, Array]:
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
