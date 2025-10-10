# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from dataclasses import dataclass

from einops import reduce
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.denoisers.utils import resample_particles, normalize_log_weights


@dataclass
class FPSDenoiser(CondDenoiser):
    """Filtering Posterior Sampling (FPS) Denoiser.

    Implements continuous-time SDE version of FPS for conditional generation
    with particle filtering and resampling.

    Args:
        integrator: Numerical integrator for solving the reverse SDE
        model: Diffusion model defining the forward process
        predictor: Predictor for computing score/noise/velocity
        forward_model: Forward measurement operator

    Attributes:
        resample: Whether to use particle resampling (set in __post_init__)
        ess_low: Low threshold for effective sample size (0.2)
        ess_high: High threshold for effective sample size (0.6)

    References:
        Dou, Z., & Song, Y. (2024). Diffusion Posterior Sampling for Linear Inverse
        Problem Solving: A Filtering Perspective. arXiv:2407.03981
    """

    def __post_init__(self):
        self.resample = True
        self.ess_low = 0.2
        self.ess_high = 0.6

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Single step of continuous-time FPS sampling.

        Implements FPS by separating:
        1. Computation of guidance term at current position
        2. Unconditional diffusion step with integrator
        3. Guidance correction applied to result

        This approach works correctly with second-order integrators (Heun, DPM++, etc.)
        because the integrator sees the true unconditional score/velocity.

        Args:
            rng_key: Random number generator key
            state: Current conditional denoiser state
            measurement_state: Measurement information

        Returns:
            Updated conditional denoiser state
        """
        position_current = state.integrator_state.position
        t_current = self.integrator.timer(state.integrator_state.step)
        t_next = self.integrator.timer(state.integrator_state.step + 1)
        dt = t_next - t_current

        # Compute guidance score at current position
        y_t = self.y_noiser(rng_key, t_current, measurement_state).position
        sigma_t = self.model.noise_level(t_current)
        y_pred = self.forward_model.apply(position_current, measurement_state)
        residual = y_t - y_pred
        guidance_score = self.forward_model.restore(residual, measurement_state) / (self.forward_model.std * sigma_t)

        # Take unconditional integrator step (works with any integrator)
        integrator_state_uncond = self.integrator(state.integrator_state, self.predictor)

        # Apply guidance correction
        # In the probability flow ODE, score modifications affect position through g(t)² factor
        # For numerical stability, we clip g_t^2 to prevent overflow with Flow models
        _, g_t = self.model.sde_coefficients(t_current)
        g_t_squared = jnp.clip(g_t**2, 0.0, 100.0)  # Clip to prevent overflow
        correction = -g_t_squared * dt * guidance_score
        position_corrected = integrator_state_uncond.position + correction

        # Create next state with corrected position
        integrator_state_next = integrator_state_uncond._replace(position=position_corrected)
        state_next = state._replace(integrator_state=integrator_state_next)

        return state_next

    def y_noiser(self, key: PRNGKeyArray, t: float, measurement_state: MeasurementState) -> SDEState:
        r"""Generate noisy measurement at time t.

        Computes :math:`y^{(t)} = \sqrt{\bar{\alpha}_t} y + \sqrt{1-\bar{\alpha}_t} A_\xi \epsilon`

        Args:
            key: Random number generator key
            t: Current time
            measurement_state: Measurement information

        Returns:
            SDEState containing the noised measurement
        """
        y_0 = measurement_state.y
        alpha_t = self.model.signal_level(t)

        # Noise y_t as the mean to keep deterministic sampling methods deterministic
        # rndm = jax.random.normal(key, y_0.shape)
        res = alpha_t * y_0  # + noise_level * rndm

        return SDEState(res, t)

    def resampler(
        self,
        state_next: CondDenoiserState,
        measurement_state: MeasurementState,
        rng_key: PRNGKeyArray,
    ) -> CondDenoiserState:
        """
        Resample particles based on the current state and measurement.

        This method resamples particles if the Effective Sample Size (ESS) falls below
        the specified thresholds, ensuring the quality of the particle set.

        Args:
            state_next: Next state of the denoiser. Shape: (n_particles, ...)
            measurement_state: Current measurement state.
            rng_key: Random number generator key.

        Returns:
            CondDenoiserState: Updated state after resampling.
        """
        integrator_state = state_next.integrator_state
        x_t = state_next.integrator_state.position
        rng_key, rng_key_resample = jax.random.split(rng_key)

        t = self.integrator.timer(state_next.integrator_state.step)

        keys = jax.random.split(rng_key, x_t.shape[0])
        y_t = jax.vmap(self.y_noiser, in_axes=(0, 0, None))(keys, t, measurement_state).position
        f_x_t = jax.vmap(self.forward_model.apply, in_axes=(0, None))(x_t, measurement_state)

        # Compute ||y_t - A(x_t)||² for each particle (shape: n_particles)
        residual_squared = reduce((y_t - f_x_t) ** 2, "b ... -> b", "sum")

        # Compute log weights from measurement likelihood: log p(y_t|x_t)
        # For Gaussian noise: log p(y|x) = -||y - Ax||² / (2σ²)
        # Note: Compute fresh weights at each step (no accumulation) to prevent degeneracy
        log_weights = -residual_squared / (2 * self.forward_model.std**2)
        log_weights = normalize_log_weights(log_weights)
        position, log_weights = resample_particles(integrator_state.position, log_weights, rng_key_resample, self.ess_low, self.ess_high)

        integrator_state_next = state_next.integrator_state._replace(position=position)
        return CondDenoiserState(integrator_state_next, log_weights)
