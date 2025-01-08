from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.integrator.base import IntegratorState
from diffuse.diffusion.sde import SDE, SDEState


class EulerState(IntegratorState):
    position: Array
    t: float
    dt: float


@dataclass
class Euler:
    """Euler deterministic integrator for ODEs"""

    sde: SDE

    def init(
        self, position: Array, rng_key: PRNGKeyArray, t: float, dt: float
    ) -> EulerState:
        """Initialize integrator state with position, timestep and step size"""
        return EulerState(position, t, dt)

    def __call__(self, integrator_state: EulerState, score: Callable) -> EulerState:
        """Perform one Euler integration step: dx = drift*dt"""
        position, t, dt = integrator_state
        drift = self.sde.reverse_drift(integrator_state, score)
        dx = drift * dt
        return EulerState(position + dx, t + dt, dt)


@dataclass
class DPMpp2sIntegrator:
    """DPM++-P2S integrator for SDEs"""

    sde: SDE

    def init(
        self, position: Array, rng_key: PRNGKeyArray, t: float, dt: float
    ) -> EulerState:
        """Initialize integrator state with position, timestep and step size"""
        return EulerState(position, jnp.array(self.sde.tf - t), jnp.array(-dt))

    def __call__(self, integrator_state: EulerState, score: Callable) -> EulerState:
        """Perform one DPM++-P2S integration step"""
        position, t, dt = integrator_state
        t0 = self.sde.beta.t0
        current_t = t + dt
        mid_t = (t + current_t) / 2

        prev_int_b, current_int_b, mid_int_b = (
            self.sde.beta.integrate(t, t0),
            self.sde.beta.integrate(current_t, t0),
            self.sde.beta.integrate(mid_t, t0),
        )
        prev_alpha, current_alpha, mid_alpha = (
            jnp.exp(-0.5 * prev_int_b),
            jnp.exp(-0.5 * current_int_b),
            jnp.exp(-0.5 * mid_int_b),
        )
        prev_beta, current_beta, mid_beta = (
            jnp.sqrt(1 - jnp.exp(-prev_int_b)),
            jnp.sqrt(1 - jnp.exp(-current_int_b)),
            jnp.sqrt(1 - jnp.exp(-mid_int_b)),
        )

        prev_log_scale = jnp.log(prev_alpha / prev_beta)
        current_log_scale = jnp.log(current_alpha / current_beta)
        mid_log_scale = jnp.log(mid_alpha / mid_beta)

        h = current_log_scale - prev_log_scale
        r = (mid_log_scale - prev_log_scale) / h

        prev_tweedie = self.sde.tweedie(SDEState(position, t), score)
        u = (
            jnp.sqrt(mid_beta) / jnp.sqrt(prev_beta) * position
            + mid_alpha * (jnp.exp(-h * r) - 1) * prev_tweedie.position
        )

        mid_tweedie = self.sde.tweedie(SDEState(u, mid_t), score)
        D = (1 - 1 / (2 * r)) * prev_tweedie.position + (
            1 / (2 * r)
        ) * mid_tweedie.position

        next_position = (
            jnp.sqrt(current_beta) / jnp.sqrt(prev_beta) * position
            - current_alpha * (jnp.exp(-h) - 1) * D
        )
        next_position = jnp.where(current_t < 1e-5, position, next_position)
        return EulerState(next_position, current_t, dt)
