from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from diffuse.diffusion.sde import SDEState
from diffuse.integrator.base import IntegratorState, ChurnedIntegrator


__all__ = ["EulerIntegrator", "HeunIntegrator", "DPMpp2sIntegrator", "DDIMIntegrator"]


@dataclass
class EulerIntegrator(ChurnedIntegrator):
    def __call__(
        self, integrator_state: IntegratorState, score: Callable
    ) -> IntegratorState:
        _, rng_key, step = integrator_state
        _, rng_key_next = jax.random.split(rng_key)

        position_churned, t_churned = self._churn_fn(integrator_state)
        t_next = self.timer(step + 1)
        dt = t_next - t_churned
        drift = (
            -0.5
            * self.sde.beta(t_churned)
            * (position_churned + score(position_churned, t_churned))
        )
        dx = drift * dt
        _, rng_key_next = jax.random.split(rng_key)
        return IntegratorState(position_churned + dx, rng_key_next, step + 1)


@dataclass
class HeunIntegrator(ChurnedIntegrator):
    def __call__(
        self, integrator_state: IntegratorState, score: Callable
    ) -> IntegratorState:
        _, rng_key, step = integrator_state
        _, rng_key_next = jax.random.split(rng_key)

        position_churned, t_churned = self._churn_fn(integrator_state)

        t_next = self.timer(step + 1)
        dt = t_next - t_churned

        drift_churned = (
            -0.5
            * self.sde.beta(t_churned)
            * (position_churned + score(position_churned, t_churned))
        )
        position_next_churned = position_churned + drift_churned * dt

        drift_next = (
            -0.5
            * self.sde.beta(t_next)
            * (position_next_churned + score(position_next_churned, t_next))
        )
        position_next_heun = position_churned + (drift_churned + drift_next) * dt / 2

        next_state = jax.lax.cond(
            step + 1 == self.timer.n_steps,
            lambda: IntegratorState(position_next_churned, rng_key_next, step + 1),
            lambda: IntegratorState(position_next_heun, rng_key_next, step + 1),
        )

        return next_state


@dataclass
class DPMpp2sIntegrator(ChurnedIntegrator):
    def __call__(
        self, integrator_state: IntegratorState, score: Callable
    ) -> IntegratorState:
        _, rng_key, step = integrator_state
        _, rng_key_next = jax.random.split(rng_key)

        position_churned, t_churned = self._churn_fn(integrator_state)

        t_next = self.timer(step + 1)
        t_mid = (t_churned + t_next) / 2

        t0 = self.sde.beta.t0
        int_b_churned, int_b_next, int_b_mid = (
            self.sde.beta.integrate(t_churned, t0),
            self.sde.beta.integrate(t_next, t0),
            self.sde.beta.integrate(t_mid, t0),
        )
        alpha_churned, alpha_next, alpha_mid = (
            jnp.exp(-0.5 * int_b_churned),
            jnp.exp(-0.5 * int_b_next),
            jnp.exp(-0.5 * int_b_mid),
        )
        sigma_churned, sigma_next, sigma_mid = (
            jnp.sqrt(-jnp.expm1(-int_b_churned)),
            jnp.sqrt(-jnp.expm1(-int_b_next)),
            jnp.sqrt(-jnp.expm1(-int_b_mid)),
        )

        log_scale_churned, log_scale_next, log_scale_mid = (
            jnp.log(alpha_churned / sigma_churned),
            jnp.log(alpha_next / sigma_next),
            jnp.log(alpha_mid / sigma_mid),
        )

        h = log_scale_next - log_scale_churned
        r = (log_scale_mid - log_scale_churned) / h

        pred_x0_churned = self.sde.tweedie(
            SDEState(position_churned, t_churned), score
        ).position
        u = (
            sigma_mid / sigma_churned * position_churned
            - alpha_mid * jnp.expm1(-h * r) * pred_x0_churned
        )

        pred_x0_mid = self.sde.tweedie(SDEState(u, t_mid), score).position
        D = (1 - 1 / (2 * r)) * pred_x0_churned + (1 / (2 * r)) * pred_x0_mid

        next_position = (
            sigma_next / sigma_churned * position_churned
            - alpha_next * jnp.expm1(-h) * D
        )

        _, rng_key_next = jax.random.split(rng_key)
        next_state = IntegratorState(next_position, rng_key_next, step + 1)

        return next_state


@dataclass
class DDIMIntegrator(ChurnedIntegrator):
    """
    Deterministic DDIM integrator that performs reverse process steps using dt.
    The reverse process starts at t=0 (noise at forward time T) and progresses to t=T (data at forward time 0).
    """

    eta: float = 0.0  # Controls stochasticity (η=0 for deterministic DDIM)

    def __call__(
        self, integrator_state: IntegratorState, score: Callable
    ) -> IntegratorState:
        """Perform one DDIM step from t to t+dt in the REVERSE process"""
        _, rng_key, step = integrator_state
        _, rng_key_next = jax.random.split(rng_key)

        position_churned, t_churned = self._churn_fn(integrator_state)

        noise_pred = self.sde.score_to_noise(score)

        t_next = self.timer(step + 1)

        # Compute cumulative noise schedule (β) integrals for α calculation
        # Corrected integration limits: integrate from 0 to t_forward_curr (not curr_t to 0)
        int_b_churned = self.sde.beta.integrate(t_churned, self.sde.beta.t0).squeeze()
        int_b_next = self.sde.beta.integrate(t_next, self.sde.beta.t0).squeeze()

        # Compute α (signal rate) and β (noise variance) for current/next steps
        alpha_churned = jnp.exp(-0.5 * int_b_churned)  # α_t = exp(-0.5 ∫₀^t β(s) ds)
        alpha_next = jnp.exp(-0.5 * int_b_next)
        beta_churned = 1.0 - alpha_churned**2  # β_t = 1 - α_t² (noise variance)

        # Predict noise using the FORWARD time (t_forward_curr)
        eps = noise_pred(position_churned, t_churned)

        # Estimate x₀ from x_t and predicted noise
        pred_x0 = (position_churned - jnp.sqrt(beta_churned) * eps) / alpha_churned

        # Compute stochasticity term (σ) for next step
        sigma_next = self.eta * jnp.sqrt(1 - alpha_next**2)

        # Sample noise if η > 0 (stochastic DDIM)
        noise = jax.random.normal(rng_key, position_churned.shape)
        _, rng_key_next = jax.random.split(rng_key)

        # DDIM update rule (reverse process)
        position_next = (
            alpha_next * pred_x0  # Data estimate term
            + jnp.sqrt(1 - alpha_next**2 - sigma_next**2) * eps  # Direction adjustment
            + sigma_next * noise  # Stochastic noise (if η > 0)
        )

        return IntegratorState(position_next, rng_key_next, step + 1)
