from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.integrator.base import IntegratorState
from diffuse.diffusion.sde import SDE, SDEState


class EulerState(IntegratorState):
    position: Array
    rng_key: PRNGKeyArray
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
        return EulerState(position, rng_key, t, dt)

    def __call__(self, integrator_state: EulerState, score: Callable) -> EulerState:
        """Perform one Euler integration step: dx = drift*dt"""
        position, t, dt, rng_key = integrator_state
        drift = self.sde.reverse_drift(integrator_state, score)
        dx = drift * dt
        return EulerState(position + dx, rng_key, t + dt, dt)


@dataclass
class DPMpp2sIntegrator:
    """DPM++-2S integrator for SDEs"""

    sde: SDE
    stochastic_churn_rate: float = 0.0
    churn_min: float = 0.05
    churn_max: float = 1.95
    noise_inflation_factor: float = 1.0

    def init(
        self, position: Array, rng_key: PRNGKeyArray, t: float, dt: float
    ) -> EulerState:
        """Initialize integrator state with position, timestep and step size"""
        return EulerState(position, rng_key, t, dt)

    def next_churn_noise_level(self, integrator_state: EulerState) -> float:
        """Compute the next churn noise level"""
        _, _, t, dt = integrator_state
        t = self.sde.tf - t
        n_steps = self.sde.tf / dt
        churn_rate = jnp.where(
            self.stochastic_churn_rate / n_steps - jnp.sqrt(2) + 1 > 0,
            jnp.sqrt(2) - 1,
            self.stochastic_churn_rate / n_steps,
        )
        churn_rate = jnp.where(
            t > self.churn_min, jnp.where(t < self.churn_max, churn_rate, 0), 0
        )
        return t * (1 + churn_rate)

    def apply_stochastic_churn(self, integrator_state: EulerState) -> EulerState:
        """Apply stochastic churn to the sample"""
        position, rng_key, noise_level, dt = integrator_state
        noise_level = self.sde.tf - noise_level
        next_noise_level = self.next_churn_noise_level(integrator_state)
        extra_noise_stddev = (
            jnp.sqrt(next_noise_level**2 - noise_level**2) * self.noise_inflation_factor
        )
        new_position = (
            position + jax.random.normal(rng_key, position.shape) * extra_noise_stddev
        )
        _, rng_key_next = jax.random.split(rng_key)

        return EulerState(
            new_position, rng_key_next, self.sde.tf - next_noise_level, dt
        )

    def __call__(self, integrator_state: EulerState, score: Callable) -> EulerState:
        """Perform one DPM++-2S integration step"""
        integrator_state = jax.lax.cond(
            self.stochastic_churn_rate > 0,
            self.apply_stochastic_churn,
            lambda x: x,
            integrator_state,
        )
        position, rng_key, t, dt = integrator_state
        t0 = self.sde.beta.t0
        current_t = t + dt
        mid_t = (t + current_t) / 2

        prev_int_b, current_int_b, mid_int_b = (
            self.sde.beta.integrate(self.sde.tf - t, t0),
            self.sde.beta.integrate(self.sde.tf - current_t, t0),
            self.sde.beta.integrate(self.sde.tf - mid_t, t0),
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

        prev_tweedie = self.sde.tweedie(SDEState(position, self.sde.tf - t), score)
        u = (
            jnp.sqrt(mid_beta) / jnp.sqrt(prev_beta) * position
            + mid_alpha * (jnp.exp(-h * r) - 1) * prev_tweedie.position
        )

        mid_tweedie = self.sde.tweedie(SDEState(u, self.sde.tf - mid_t), score)
        D = (1 - 1 / (2 * r)) * prev_tweedie.position + (
            1 / (2 * r)
        ) * mid_tweedie.position

        next_position = (
            jnp.sqrt(current_beta) / jnp.sqrt(prev_beta) * position
            - current_alpha * (jnp.exp(-h) - 1) * D
        )

        # Handle edge case when t = tf
        next_position = jnp.where(
            (self.sde.tf - current_t) < 1e-5, position, next_position
        )
        _, rng_key_next = jax.random.split(rng_key)
        return EulerState(next_position, rng_key_next, current_t, dt)
