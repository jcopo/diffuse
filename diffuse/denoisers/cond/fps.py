from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.denoisers.utils import resample_particles
from diffuse.predictor import Predictor


@dataclass
class FPSDenoiser(CondDenoiser):
    """Filtering Posterior Sampling Denoiser implementing continuous-time SDE version."""

    def __post_init__(self):
        self.resample = False
        self.ess_low = 0.05
        self.ess_high = 0.9

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
        guidance_score = self.forward_model.adjoint(residual, measurement_state) / (self.forward_model.std * sigma_t)

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
        state: CondDenoiserState,
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

        # weight computation
        t = self.integrator.timer(state.integrator_state.step)
        alpha_t, sigma_t = self.model.signal_level(t), self.model.noise_level(t)
        y_t = alpha_t * jnp.asarray(measurement_state.y)

        tweedies = jax.vmap(self.model.tweedie, in_axes=(0, None))(
            SDEState(state.integrator_state.position, t), self.predictor.score
        )
        forwards = jax.vmap(self.forward_model.apply, in_axes=(0, None))(tweedies.position, measurement_state)

        residual_norm = jnp.linalg.norm(y_t - alpha_t * forwards)
        log_weights = -0.5 * residual_norm**2 / (self.forward_model.std**2 * sigma_t**2)

        # resample particles
        position, log_weights = resample_particles(
            state.integrator_state.position, log_weights, rng_key, self.ess_low, self.ess_high
        )
        integrator_state_next = state.integrator_state._replace(position=position)
        return CondDenoiserState(integrator_state_next, log_weights)
