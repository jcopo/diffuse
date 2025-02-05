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
        position, rng_key, t, dt = integrator_state
        drift = self.sde.reverse_drift_ode(integrator_state, score)
        dx = drift * dt
        _, rng_key_next = jax.random.split(rng_key)
        return EulerState(position + dx, rng_key_next, t + dt, dt)


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
        _, _, t_reverse, dt = integrator_state
        t_forward_curr = self.sde.tf - t_reverse
        n_steps = self.sde.tf / dt
        churn_rate = jnp.where(
            self.stochastic_churn_rate / n_steps - jnp.sqrt(2) + 1 > 0,
            jnp.sqrt(2) - 1,
            self.stochastic_churn_rate / n_steps,
        )
        churn_rate = jnp.where(
            t_forward_curr > self.churn_min, jnp.where(t_forward_curr < self.churn_max, churn_rate, 0), 0
        )
        return self.sde.tf - t_forward_curr * (1 + churn_rate)

    def apply_stochastic_churn(self, integrator_state: EulerState) -> EulerState:
        """Apply stochastic churn to the sample"""
        position, rng_key, t_reverse, dt = integrator_state
        t_forward_curr = self.sde.tf - t_reverse
        t_forward_prev = self.sde.tf - self.next_churn_noise_level(integrator_state)
        int_b_curr, int_b_prev = (
            self.sde.beta.integrate(t_forward_curr, self.sde.beta.t0),
            self.sde.beta.integrate(t_forward_prev, self.sde.beta.t0),
        )
        beta_curr, beta_prev = (
            jnp.sqrt(1 - jnp.exp(-int_b_curr)),
            jnp.sqrt(1 - jnp.exp(-int_b_prev)),
        )
        extra_noise_stddev = (
            jnp.sqrt(beta_prev**2 - beta_curr**2) * self.noise_inflation_factor
        )
        new_position = (
            position + jax.random.normal(rng_key, position.shape) * extra_noise_stddev
        )
        _, rng_key_next = jax.random.split(rng_key)
        return EulerState(
            new_position, rng_key_next, self.sde.tf - t_forward_prev, dt
        )

    def __call__(self, integrator_state: EulerState, score: Callable) -> EulerState:
        """Perform one DPM++-2S integration step"""
        integrator_state = jax.lax.cond(
            self.stochastic_churn_rate > 0,
            self.apply_stochastic_churn,
            lambda x: x,
            integrator_state,
        )
        position, rng_key, t_reverse, dt = integrator_state
        t0 = self.sde.beta.t0
        t_forward_curr = self.sde.tf - t_reverse

        t_forward_next = t_forward_curr - dt
        t_forward_mid = jnp.sqrt(t_forward_curr * t_forward_next)# (t_forward_curr + t_forward_next) / 2

        int_b_curr, int_b_next, int_b_mid = (
            self.sde.beta.integrate(t_forward_curr, t0),
            self.sde.beta.integrate(t_forward_next, t0),
            self.sde.beta.integrate(t_forward_mid, t0),
        )
        alpha_curr, alpha_next, alpha_mid = (
            jnp.exp(-0.5 * int_b_curr),
            jnp.exp(-0.5 * int_b_next),
            jnp.exp(-0.5 * int_b_mid),
        )
        sigma_curr, sigma_next, sigma_mid = (
            jnp.sqrt(1 - jnp.exp(-int_b_curr)),
            jnp.sqrt(1 - jnp.exp(-int_b_next)),
            jnp.sqrt(1 - jnp.exp(-int_b_mid)),
        )

        log_scale_curr, log_scale_next, log_scale_mid = (
            jnp.log(alpha_curr / sigma_curr),
            jnp.log(alpha_next / sigma_next),
            jnp.log(alpha_mid / sigma_mid),
        )

        h = log_scale_next - log_scale_curr
        r = (log_scale_mid - log_scale_curr) / h

        noise_pred = self.sde.score_to_noise(score)
        # pred_x0_curr = (position - sigma_curr * noise_pred(position, t_forward_curr)) / alpha_curr
        pred_x0_curr = self.sde.tweedie(SDEState(position, t_forward_curr), score).position
        u = (
            sigma_mid / sigma_curr * position
            - alpha_mid * (jnp.exp(-h * r) - 1) * pred_x0_curr
        )

        # pred_x0_mid = (u - sigma_mid * noise_pred(u, t_forward_mid)) / alpha_mid
        pred_x0_mid = self.sde.tweedie(SDEState(u, t_forward_mid), score).position
        D = (1 - 1 / (2 * r)) * pred_x0_curr + (
            1 / (2 * r)
        ) * pred_x0_mid

        next_position = (
            sigma_next / sigma_curr * position
            - alpha_next * (jnp.exp(-h) - 1) * D
        )

        _, rng_key_next = jax.random.split(rng_key)
        next_state = EulerState(next_position, rng_key_next, t_reverse + dt, dt)

        next_state = jax.lax.cond(
            t_forward_next < 1e-5,
            lambda x, y: self.euler_step(x, rng_key_next, score),
            lambda x, y: y,
            integrator_state,
            next_state,
        )
        return next_state

    def euler_step(
        self, integrator_state: EulerState, rng_key_next: PRNGKeyArray, score: Callable
    ) -> EulerState:
        """Perform one Euler integration step: dx = drift*dt"""
        position, _, t, dt = integrator_state
        drift = self.sde.reverse_drift_ode(integrator_state, score)
        dx = drift * dt
        return EulerState(position + dx, rng_key_next, t + dt, dt)


class DDIMState(IntegratorState):
    """DDIM integrator state"""
    position: Array
    rng_key: PRNGKeyArray
    t: float
    dt: float


@dataclass
class DDIMIntegrator:
    """
    Deterministic DDIM integrator that performs reverse process steps using dt.
    The reverse process starts at t=0 (noise at forward time T) and progresses to t=T (data at forward time 0).
    """
    sde: SDE
    eta: float = 0.0  # Controls stochasticity (η=0 for deterministic DDIM)

    def init(
        self,
        position: Array,
        rng_key: PRNGKeyArray,
        t: float,
        dt: float
    ) -> DDIMState:
        """Initialize integrator state with position, time, and timestep"""
        return DDIMState(position=position, rng_key=rng_key, t=t, dt=dt)

    def __call__(
        self,
        integrator_state: DDIMState,
        score: Callable
    ) -> DDIMState:
        """Perform one DDIM step from t to t+dt in the REVERSE process"""
        position, rng_key, t_reverse, dt = integrator_state
        noise_pred = self.sde.score_to_noise(score)

        # Convert reverse time (t) to forward time (T - t_reverse)
        t_forward_curr = self.sde.tf - t_reverse  # Current forward time
        t_forward_next = t_forward_curr - dt      # Next forward time

        # Compute cumulative noise schedule (β) integrals for α calculation
        # Corrected integration limits: integrate from 0 to t_forward_curr (not curr_t to 0)
        int_b_curr = self.sde.beta.integrate(t_forward_curr, self.sde.beta.t0).squeeze()
        int_b_next = self.sde.beta.integrate(t_forward_next, self.sde.beta.t0).squeeze()

        # Compute α (signal rate) and β (noise variance) for current/next steps
        alpha_curr = jnp.exp(-0.5 * int_b_curr)        # α_t = exp(-0.5 ∫₀^t β(s) ds)
        alpha_next = jnp.exp(-0.5 * int_b_next)
        beta_curr = 1.0 - alpha_curr**2               # β_t = 1 - α_t² (noise variance)

        # Predict noise using the FORWARD time (t_forward_curr)
        eps = noise_pred(position, t_forward_curr)

        # Estimate x₀ from x_t and predicted noise
        pred_x0 = (position - jnp.sqrt(beta_curr) * eps) / alpha_curr

        # Compute stochasticity term (σ) for next step
        sigma_next = self.eta * jnp.sqrt(1 - alpha_next**2)

        # Sample noise if η > 0 (stochastic DDIM)
        noise = jax.random.normal(rng_key, position.shape)
        _, rng_key_next = jax.random.split(rng_key)

        # DDIM update rule (reverse process)
        position_next = (
            alpha_next * pred_x0 +  # Data estimate term
            jnp.sqrt(1 - alpha_next**2 - sigma_next**2) * eps +  # Direction adjustment
            sigma_next * noise  # Stochastic noise (if η > 0)
        )

        return DDIMState(
            position=position_next,
            rng_key=rng_key_next,
            t=t_reverse + dt,  # Advance reverse time by dt
            dt=dt
        )