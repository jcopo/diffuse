# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from diffuse.diffusion.sde import SDEState, DiffusionModel
from diffuse.integrator.base import IntegratorState, ChurnedIntegrator
from diffuse.predictor import Predictor


__all__ = ["EulerIntegrator", "HeunIntegrator", "DPMpp2sIntegrator", "DDIMIntegrator"]


@dataclass
class EulerIntegrator(ChurnedIntegrator):
    r"""Euler integrator for probability flow ODEs in diffusion models.

    Implements the basic Euler method for numerical integration:

    .. math::
        dx = v(x,t) \cdot dt

    where :math:`v(x,t)` is the velocity field from the probability flow ODE.
    Works with all diffusion models (SDE, Flow, EDM) using the velocity parameterization.

    Args:
        model: Diffusion model defining the diffusion process
        timer: Timer object managing the discretization of the time interval
        stochastic_churn_rate: Rate of applying stochastic churning (default: 0.0)
        churn_min: Minimum time threshold for churning (default: 0.0)
        churn_max: Maximum time threshold for churning (default: 0.0)
        noise_inflation_factor: Factor to scale injected noise (default: 1.0)
    """

    model: DiffusionModel

    def __call__(self, integrator_state: IntegratorState, predictor: Predictor) -> IntegratorState:
        """Perform one Euler integration step.

        Args:
            integrator_state: Current state containing (position, rng_key, step)
            predictor: Predictor providing velocity field v(x,t)

        Returns:
            Updated IntegratorState with the next position
        """
        _, rng_key, step = integrator_state
        _, rng_key_next = jax.random.split(rng_key)

        position_churned, t_churned = self._churn_fn(integrator_state)

        t_next = self.timer(step + 1)
        dt = t_next - t_churned

        # Use velocity directly for probability flow ODE
        velocity = predictor.velocity(position_churned, t_churned)
        dx = velocity * dt
        _, rng_key_next = jax.random.split(rng_key)

        return IntegratorState(position_churned + dx, rng_key_next, step + 1)


@dataclass
class HeunIntegrator(ChurnedIntegrator):
    r"""Heun's method integrator for probability flow ODEs in diffusion models.

    Implements a second-order Runge-Kutta method (Heun's method) that uses an
    intermediate Euler step to improve accuracy:

    .. math::
        x_{n+1} = x_n + \frac{v_1 + v_2}{2} \cdot dt

    where:

    - :math:`v_1 = v(x_n, t_n)`
    - :math:`v_2 = v(x_n + v_1 \cdot dt, t_{n+1})`

    Works with all diffusion models (SDE, Flow, EDM) using the velocity parameterization.

    Args:
        model: Diffusion model defining the diffusion process
        timer: Timer object managing the discretization of the time interval
        stochastic_churn_rate: Rate of applying stochastic churning (default: 0.0)
        churn_min: Minimum time threshold for churning (default: 0.0)
        churn_max: Maximum time threshold for churning (default: 0.0)
        noise_inflation_factor: Factor to scale injected noise (default: 1.0)
    """

    model: DiffusionModel

    def __call__(self, integrator_state: IntegratorState, predictor: Predictor) -> IntegratorState:
        """Perform one Heun integration step.

        Args:
            integrator_state: Current state containing (position, rng_key, step)
            predictor: Predictor providing velocity field v(x,t)

        Returns:
            Updated IntegratorState with the next position using Heun's method
        """
        _, rng_key, step = integrator_state
        _, rng_key_next = jax.random.split(rng_key)

        position_churned, t_churned = self._churn_fn(integrator_state)

        t_next = self.timer(step + 1)
        dt = t_next - t_churned

        # Heun's method using velocity (probability flow ODE)
        # k1 = velocity at current point
        velocity_churned = predictor.velocity(position_churned, t_churned)
        position_next_euler = position_churned + velocity_churned * dt

        # k2 = velocity at Euler prediction
        velocity_next = predictor.velocity(position_next_euler, t_next)

        # Heun correction: average of the two velocities
        position_next_heun = position_churned + (velocity_churned + velocity_next) * dt / 2

        return IntegratorState(position_next_heun, rng_key_next, step + 1)


@dataclass
class DPMpp2sIntegrator(ChurnedIntegrator):
    """DPM-Solver++ (2S) integrator for reverse-time diffusion processes.

    Implements the 2nd-order DPM-Solver++ algorithm which uses a midpoint
    prediction step and dynamic thresholding. This method provides improved
    stability and accuracy compared to basic Euler integration.

    The method uses log-space computations and midpoint predictions to
    better handle the diffusion process dynamics.
    """

    model: DiffusionModel

    def __call__(self, integrator_state: IntegratorState, predictor: Predictor) -> IntegratorState:
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

        signal_level_churned = self.model.signal_level(t_churned)
        signal_level_mid = self.model.signal_level(t_mid)
        signal_level_next = self.model.signal_level(t_next)

        sigma_churned = self.model.noise_level(t_churned)
        sigma_next = self.model.noise_level(t_next)
        sigma_mid = self.model.noise_level(t_mid)

        log_scale_churned, log_scale_next, log_scale_mid = (
            jnp.log(signal_level_churned / sigma_churned),
            jnp.log(signal_level_next / sigma_next),
            jnp.log(signal_level_mid / sigma_mid),
        )

        h = jnp.clip(log_scale_next - log_scale_churned, 1e-6)
        r = jnp.clip((log_scale_mid - log_scale_churned) / h, 1e-6)

        pred_x0_churned = self.model.tweedie(SDEState(position_churned, t_churned), predictor.score).position

        u = sigma_mid / sigma_churned * position_churned - signal_level_mid * jnp.expm1(-h * r) * pred_x0_churned

        pred_x0_mid = self.model.tweedie(SDEState(u, t_mid), predictor.score).position
        D = (1 - 1 / (2 * r)) * pred_x0_churned + (1 / (2 * r)) * pred_x0_mid

        next_position = sigma_next / sigma_churned * position_churned - signal_level_next * jnp.expm1(-h) * D

        _, rng_key_next = jax.random.split(rng_key)
        next_state = IntegratorState(next_position, rng_key_next, step + 1)
        return next_state


@dataclass
class DDIMIntegrator(ChurnedIntegrator):
    r"""Denoising Diffusion Implicit Models (DDIM) integrator.

    DDIM assumes the same latent noise :math:`\varepsilon` along the entire path. The update rule is:

    .. math::
        x_s = \frac{\alpha_s}{\alpha_t} x_t - \left(\frac{\alpha_s \sigma_t}{\alpha_t} - \sigma_s\right) \varepsilon_\theta(x_t, t)

    where :math:`s < t`, and :math:`\varepsilon_\theta(x_t, t)` is the predicted noise.

    Args:
        model: Diffusion model defining the diffusion process
        timer: Timer object managing the discretization of the time interval
        stochastic_churn_rate: Rate of applying stochastic churning (default: 0.0)
        churn_min: Minimum time threshold for churning (default: 0.0)
        churn_max: Maximum time threshold for churning (default: 0.0)
        noise_inflation_factor: Factor to scale injected noise (default: 1.0)

    References:
        Song, J., Meng, C., Ermon, S. (2020). "Denoising Diffusion Implicit Models"
        arXiv:2010.02502
    """

    model: DiffusionModel

    def __call__(self, integrator_state: IntegratorState, predictor: Predictor) -> IntegratorState:
        r"""Perform one DDIM step in reverse time.

        Args:
            integrator_state: Current state containing (position, rng_key, step)
            predictor: Predictor providing noise prediction :math:`\varepsilon_\theta(x_t, t)`

        Returns:
            Updated IntegratorState with the next position computed using the DDIM update rule:

            .. math::
                x_{t-1} = \sqrt{\alpha_{t-1}} \hat{x}_0 + \sqrt{1 - \alpha_{t-1}} \varepsilon_\theta

            where:

            - :math:`\hat{x}_0 = (x_t - \sqrt{1-\alpha_t} \varepsilon_\theta) / \sqrt{\alpha_t}` is the predicted denoised sample
            - :math:`\varepsilon_\theta` is the predicted noise from the model
            - :math:`\alpha_t` represents the signal level (cumulative product of :math:`1 - \beta_t`)
            - :math:`\beta_t` is the forward process noise schedule
        """
        _, rng_key, step = integrator_state
        _, rng_key_next = jax.random.split(rng_key)

        position_churned, t_churned = self._churn_fn(integrator_state)

        t_next = self.timer(step + 1)

        signal_level_churned = self.model.signal_level(t_churned)
        signal_level_next = self.model.signal_level(t_next)
        sigma_churned = self.model.noise_level(t_churned)
        sigma_next = self.model.noise_level(t_next)

        eps = predictor.noise(position_churned, t_churned)

        pred_x0 = (position_churned - sigma_churned * eps) / signal_level_churned

        position_next = signal_level_next * pred_x0 + sigma_next * eps

        return IntegratorState(position_next, rng_key_next, step + 1)
