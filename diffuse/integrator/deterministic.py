from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from diffuse.diffusion.sde import SDEState
from diffuse.integrator.base import IntegratorState, ChurnedIntegrator


__all__ = ["EulerIntegrator", "HeunIntegrator", "DPMpp2sIntegrator", "DDIMIntegrator"]


@dataclass
class EulerIntegrator(ChurnedIntegrator):
    """Euler deterministic integrator for reverse-time diffusion processes.

    Implements the basic Euler method for numerical integration of the reverse-time SDE:
    dx = [-0.5 * β(t) * (x + s(x,t))] * dt

    where β(t) is the noise schedule and s(x,t) is the score function.
    """

    def __call__(self, integrator_state: IntegratorState, score: Callable) -> IntegratorState:
        """Perform one Euler integration step in reverse time.

        Args:
            integrator_state: Current state containing (position, rng_key, step)
            score: Score function s(x,t) that approximates ∇ₓ log p(x|t)

        Returns:
            Updated IntegratorState with the next position
        """
        _, rng_key, step = integrator_state
        _, rng_key_next = jax.random.split(rng_key)

        position_churned, t_churned = self._churn_fn(integrator_state)

        t_next = self.timer(step + 1)
        dt = t_next - t_churned
        noise_level_churned = self.sde.noise_level(t_churned)
        alpha_churned = 1 - noise_level_churned
        beta_churned = self.sde.beta(t_churned)
        drift = -0.5 * beta_churned * (position_churned + score(position_churned, t_churned))
        dx = drift * dt
        _, rng_key_next = jax.random.split(rng_key)

        return IntegratorState(position_churned + dx, rng_key_next, step + 1)


@dataclass
class HeunIntegrator(ChurnedIntegrator):
    """Heun's method integrator for reverse-time diffusion processes.

    Implements a second-order Runge-Kutta method (Heun's method) that uses an
    intermediate Euler step to improve accuracy. The final step is computed as:

    x_{n+1} = x_n + (k₁ + k₂)/2 * dt

    where:
    k₁ = drift(x_n, t_n)
    k₂ = drift(x_n + k₁*dt, t_{n+1})
    """

    def __call__(self, integrator_state: IntegratorState, score: Callable) -> IntegratorState:
        """Perform one Heun integration step in reverse time.

        Args:
            integrator_state: Current state containing (position, rng_key, step)
            score: Score function s(x,t) that approximates ∇ₓ log p(x|t)

        Returns:
            Updated IntegratorState with the next position. For the final step,
            returns the Euler prediction instead of the Heun correction.
        """
        _, rng_key, step = integrator_state
        _, rng_key_next = jax.random.split(rng_key)

        position_churned, t_churned = self._churn_fn(integrator_state)

        t_next = self.timer(step + 1)
        dt = t_next - t_churned
        noise_level_churned = self.sde.noise_level(t_churned)
        alpha_churned = 1 - noise_level_churned
        beta_churned = self.sde.beta(t_churned)
        drift_churned = -0.5 * beta_churned * (position_churned + score(position_churned, t_churned))
        position_next_churned = position_churned + drift_churned * dt

        drift_next = -0.5 * self.sde.beta(t_next) * (position_next_churned + score(position_next_churned, t_next))
        position_next_heun = position_churned + (drift_churned + drift_next) * dt / 2

        next_state = jax.lax.cond(
            step + 1 == self.timer.n_steps,
            lambda: IntegratorState(position_next_churned, rng_key_next, step + 1),
            lambda: IntegratorState(position_next_heun, rng_key_next, step + 1),
        )

        return next_state


@dataclass
class DPMpp2sIntegrator(ChurnedIntegrator):
    """DPM-Solver++ (2S) integrator for reverse-time diffusion processes.

    Implements the 2nd-order DPM-Solver++ algorithm which uses a midpoint
    prediction step and dynamic thresholding. This method provides improved
    stability and accuracy compared to basic Euler integration.

    The method uses log-space computations and midpoint predictions to
    better handle the diffusion process dynamics.
    """

    def __call__(self, integrator_state: IntegratorState, score: Callable) -> IntegratorState:
        """Perform one DPM-Solver++ (2S) integration step in reverse time.

        Args:
            integrator_state: Current state containing (position, rng_key, step)
            score: Score function s(x,t) that approximates ∇ₓ log p(x|t)

        Returns:
            Updated IntegratorState with the next position computed using
            the DPM-Solver++ (2S) algorithm
        """
        _, rng_key, step = integrator_state
        _, rng_key_next = jax.random.split(rng_key)

        position_churned, t_churned = self._churn_fn(integrator_state)

        t_next = self.timer(step + 1)
        t_mid = (t_churned + t_next) / 2

        noise_level_churned = self.sde.noise_level(t_churned)
        noise_level_next = self.sde.noise_level(t_next)
        noise_level_mid = self.sde.noise_level(t_mid)
        alpha_churned = 1 - noise_level_churned
        alpha_next = 1 - noise_level_next
        alpha_mid = 1 - noise_level_mid

        sigma_churned, sigma_next, sigma_mid = (
            jnp.sqrt(1 - alpha_churned),
            jnp.sqrt(1 - alpha_next),
            jnp.sqrt(1 - alpha_mid),
        )

        log_scale_churned, log_scale_next, log_scale_mid = (
            jnp.log(jnp.sqrt(alpha_churned) / sigma_churned),
            jnp.log(jnp.sqrt(alpha_next) / sigma_next),
            jnp.log(jnp.sqrt(alpha_mid) / sigma_mid),
        )

        h = jnp.clip(log_scale_next - log_scale_churned, 1e-6)
        r = jnp.clip((log_scale_mid - log_scale_churned) / h, 1e-6)

        pred_x0_churned = self.sde.tweedie(SDEState(position_churned, t_churned), score).position

        u = sigma_mid / sigma_churned * position_churned - jnp.sqrt(alpha_mid) * jnp.expm1(-h * r) * pred_x0_churned

        pred_x0_mid = self.sde.tweedie(SDEState(u, t_mid), score).position
        D = (1 - 1 / (2 * r)) * pred_x0_churned + (1 / (2 * r)) * pred_x0_mid

        next_position = sigma_next / sigma_churned * position_churned - jnp.sqrt(alpha_next) * jnp.expm1(-h) * D

        _, rng_key_next = jax.random.split(rng_key)
        next_state = IntegratorState(next_position, rng_key_next, step + 1)
        return next_state


@dataclass
class DDIMIntegrator(ChurnedIntegrator):
    """Denoising Diffusion Implicit Models (DDIM) integrator.

    Implements the DDIM sampling procedure which enables fast, high-quality sampling
    through a non-Markovian deterministic process. DDIM can generate samples in
    significantly fewer steps than DDPM while maintaining sample quality.

    The update rule follows:
    x_{t-1} = √α_{t-1} * x̂₀ + √(1 - α_{t-1}) * ε_θ

    where:
    - x̂₀ is the predicted denoised sample: (x_t - √(1-α_t) * ε_θ) / √α_t
    - ε_θ is the predicted noise from the model
    - α_t represents the cumulative product of (1 - β_t)
    - β_t is the forward process noise schedule

    This implementation uses a deterministic sampler (η=0) which provides a good
    balance between speed and quality. The deterministic nature allows for exact
    reconstruction of the reverse process given the same noise predictions.

    References:
        Song, J., Meng, C., Ermon, S. (2020). "Denoising Diffusion Implicit Models"
        https://arxiv.org/abs/2010.02502

    """

    def __call__(self, integrator_state: IntegratorState, score: Callable) -> IntegratorState:
        """Perform one DDIM step in reverse time.

        Args:
            integrator_state: Current state containing (position, rng_key, step)
            score: Score function s(x,t) that approximates ∇ₓ log p(x|t)

        Returns:
            Updated IntegratorState with the next position computed using
            the DDIM update rule:
            x_{t-1} = √α_{t-1} * x̂₀ + √(1 - α_{t-1}) * ε_θ
            where:
            - x̂₀ is the predicted denoised sample: (x_t - √(1-α_t) * ε_θ) / √α_t
            - ε_θ is the predicted noise from the model
            - α_t represents the cumulative product of (1 - β_t)
            - β_t is the forward process noise schedule
        """
        _, rng_key, step = integrator_state
        _, rng_key_next = jax.random.split(rng_key)

        position_churned, t_churned = self._churn_fn(integrator_state)

        noise_pred = self.sde.score_to_noise(score)

        t_next = self.timer(step + 1)

        noise_level_churned = self.sde.noise_level(t_churned)
        noise_level_next = self.sde.noise_level(t_next)
        alpha_churned = 1 - noise_level_churned
        alpha_next = 1 - noise_level_next

        eps = noise_pred(position_churned, t_churned)

        pred_x0 = (position_churned - jnp.sqrt(1 - alpha_churned) * eps) / jnp.sqrt(alpha_churned)

        position_next = jnp.sqrt(alpha_next) * pred_x0 + jnp.sqrt(1 - alpha_next) * eps

        return IntegratorState(position_next, rng_key_next, step + 1)
