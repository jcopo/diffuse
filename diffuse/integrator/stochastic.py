from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from diffuse.integrator.base import IntegratorState, Integrator
from diffuse.diffusion.sde import SDE
from diffuse.predictor import Predictor

__all__ = ["EulerMaruyamaIntegrator"]


@dataclass
class EulerMaruyamaIntegrator(Integrator):
    """Euler-Maruyama stochastic integrator for Stochastic Differential Equations (SDEs).

    Implements the Euler-Maruyama method for numerical integration of SDEs of the form:
    dX(t) = μ(X,t)dt + σ(X,t)dW(t)

    where:
    - μ(X,t) is the drift term: β(t) * (0.5 * X + score(X,t))
    - σ(X,t) is the diffusion term: sqrt(β(t))
    - dW(t) is the Wiener process increment
    - β(t) is the noise schedule

    The method advances the solution using the discrete approximation:
    X(t + dt) = X(t) + μ(X,t)dt + σ(X,t)√dt * N(0,1)

    This is the simplest stochastic integration scheme with strong order 0.5
    convergence for general SDEs.
    """

    model: SDE

    def __call__(self, integrator_state: IntegratorState, predictor: Predictor) -> IntegratorState:
        """Perform one Euler-Maruyama integration step.

        Args:
            integrator_state: Current state containing:
                - position: Current position X(t)
                - rng_key: JAX random number generator key
                - step: Current integration step
            score: Score function that approximates ∇ₓ log p(x|t)

        Returns:
            Updated IntegratorState containing:
                - New position X(t + dt)
                - Updated RNG key
                - Incremented step count

        Notes:
            The integration step implements:
            dx = drift*dt + diffusion*√dt*ε
            where:
            - drift = β(t) * (0.5 * position + score(position, t))
            - diffusion = √β(t)
            - ε ~ N(0,1)
        """
        position, rng_key, step = integrator_state
        t, t_next = self.timer(step), self.timer(step + 1)
        dt = t - t_next
        drift = self.model.beta(t) * (0.5 * position + predictor.score(position, t))
        diffusion = jnp.sqrt(self.model.beta(t))
        noise = jax.random.normal(rng_key, position.shape) * jnp.sqrt(dt)

        dx = drift * dt + diffusion * noise
        _, rng_key_next = jax.random.split(rng_key)
        return IntegratorState(position + dx, rng_key_next, step + 1)
