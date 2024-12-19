from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.integrator.base import IntegratorState
from diffuse.diffusion.sde import SDE


class EulerMaruyamaState(IntegratorState):
    """Euler-Maruyama integrator state"""
    position: Array
    rng_key: PRNGKeyArray
    t: float
    dt: float


@dataclass
class EulerMaruyama:
    """Euler-Maruyama stochastic integrator for SDEs"""
    sde: SDE

    def init(
        self, position: Array, rng_key: PRNGKeyArray, t: float, dt: float
    ) -> EulerMaruyamaState:
        """Initialize integrator state with position, random key and timestep"""
        return EulerMaruyamaState(position, rng_key, t, dt)

    def __call__(
        self, integrator_state: EulerMaruyamaState, score: Callable
    ) -> EulerMaruyamaState:
        """Perform one Euler-Maruyama integration step: dx = drift*dt + diffusion*dW"""
        position, rng_key, t, dt = integrator_state
        drift = self.sde.reverse_drift(integrator_state, score)
        diffusion = self.sde.reverse_diffusion(integrator_state)

        dx = drift * dt + diffusion * jax.random.normal(
            rng_key, position.shape
        ) * jnp.sqrt(dt)

        return EulerMaruyamaState(position + dx, rng_key, t + dt, dt)
