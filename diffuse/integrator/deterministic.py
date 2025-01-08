from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array

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

    def init(self, position: Array, t: float, dt: float) -> EulerState:
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

    def init(self, position: Array, t: float, dt: float) -> EulerState:
        """Initialize integrator state with position, timestep and step size"""
        return EulerState(position, t, dt)
    
    def __call__(self, integrator_state: EulerState, score: Callable) -> EulerState:
        """Perform one DPM++-P2S integration step"""
        position, t, dt = integrator_state
        t0 = self.sde.beta.t0
        next_t = t + dt
        mid_t = jnp.sqrt(t * next_t)

        int_b, next_int_b, mid_int_b = self.sde.beta.integrate(t, t0), self.sde.beta.integrate(next_t, t0), self.sde.beta.integrate(mid_t, t0)
        alpha, next_alpha, mid_alpha = jnp.exp(-0.5 * int_b), jnp.exp(-0.5 * next_int_b), jnp.exp(-0.5 * mid_int_b)
        beta, next_beta, mid_beta = 1 - jnp.exp(-int_b), 1 - jnp.exp(-next_int_b), 1 - jnp.exp(-mid_int_b)
        
        log_scale = jnp.log(alpha / jnp.sqrt(beta))
        log_scale_next = jnp.log(next_alpha / jnp.sqrt(next_beta))
        log_scale_mid = jnp.log(mid_alpha / jnp.sqrt(mid_beta))

        h = log_scale_next- log_scale
        r = (log_scale_mid - log_scale) / h

        init_tweedie = self.sde.tweedie(SDEState(position, t), score)
        u = jnp.sqrt(mid_beta) / jnp.sqrt(beta) * position + mid_alpha * (jnp.exp(-h * r) - 1) * init_tweedie.position

        inter_tweedie = self.sde.tweedie(SDEState(u, mid_t), score)
        D = (1 - 1 / (2 * r)) * init_tweedie.position + (1 / (2 * r)) * inter_tweedie.position

        next_position = jnp.sqrt(next_beta) / jnp.sqrt(beta) * position - next_alpha * (jnp.exp(-h) -1) * D

        return EulerState(next_position, next_t, dt)



