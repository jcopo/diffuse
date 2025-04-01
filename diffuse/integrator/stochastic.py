from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from diffuse.integrator.base import IntegratorState, Integrator

__all__ = ["EulerMaruyamaIntegrator"]


@dataclass
class EulerMaruyamaIntegrator(Integrator):
    """Euler-Maruyama stochastic integrator for SDEs"""

    def __call__(
        self, integrator_state: IntegratorState, score: Callable
    ) -> IntegratorState:
        """Perform one Euler-Maruyama integration step: dx = drift*dt + diffusion*dW"""
        position, rng_key, step = integrator_state
        t, t_next = self.timer(step), self.timer(step + 1)
        dt = t - t_next
        drift = self.sde.beta(t) * (0.5 * position + score(position, t))
        diffusion = jnp.sqrt(self.sde.beta(t))
        noise = jax.random.normal(rng_key, position.shape) * jnp.sqrt(dt)

        dx = drift * dt + diffusion * noise
        _, rng_key_next = jax.random.split(rng_key)
        return IntegratorState(position + dx, rng_key_next, step + 1)
