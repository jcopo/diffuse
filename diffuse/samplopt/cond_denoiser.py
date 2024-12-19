
from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from blackjax.smc.resampling import stratified

from diffuse.integrator.base import Integrator, IntegratorState
from diffuse.diffusion.sde import SDE
from diffuse.samplopt.inference import SDEState


class CondDenoiserState(NamedTuple):
    position: Array
    weights: Array
    integrator_state: IntegratorState
    t: float


@dataclass
class CondDenoiser:
    """Conditional denoiser for conditional diffusion"""
    integrator: Integrator
    logpdf: Callable[[SDEState, Array], Array]
    sde: SDE
    _resample: bool

    def init(self, position: Array, rng_key: PRNGKeyArray, dt: float, tf: float) -> CondDenoiserState:
        weights = jnp.ones_like(position) / position.shape[0]
        integrator_state = self.integrator.init(position, rng_key, dt)
        return CondDenoiserState(position, weights, integrator_state, tf)

    def particle_step(
        self,
        state: CondDenoiserState,
        rng_key: PRNGKeyArray,
        drift_y: Array,
    ) -> CondDenoiserState:
        r"""
        sample p(\theta_t-1 | \theta_t, \y_t-1, \xi)
        """
        position, weights, integrator_state, tf = state
        drift_x = self.sde.reverse_drift(SDEState(position, tf))
        diffusion = self.sde.reverse_diffusion(SDEState(position, tf))
        integrator_state_next = self.integrator(integrator_state, drift_x + drift_y, diffusion)
        # weights = jax.vmap(logpdf, in_axes=(SDEState(0, None),))(sde_state)
        position = integrator_state_next.position
        if self._resample:
            weights = self.logpdf(integrator_state_next, integrator_state, drift_x)
            position, weights = self._resampling(position, weights, rng_key)

        return CondDenoiserState(position, weights, integrator_state_next, tf)

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
