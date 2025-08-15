from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.denoisers.utils import resample_particles, normalize_log_weights


@dataclass
class FPSDenoiser(CondDenoiser):
    """Filtering Posterior Sampling Denoiser implementing continuous-time SDE version."""

    def __post_init__(self):
        self.resample = True
        self.ess_low = 0.05
        self.ess_high = 0.9

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Single step of continuous-time FPS sampling.

        Modifies the score to include measurement term and uses integrator for the update.
        """

        # Define modified score function that includes measurement term
        def modified_score(x: Array, t: Array) -> Array:
            # import pdb; pdb.set_trace()
            # noise y
            y_t = self.y_noiser(rng_key, t, measurement_state).position

            # Compute guidance term
            sigma_t = self.sde.noise_level(t)
            y_pred = self.forward_model.apply(x, measurement_state)
            residual = y_t - y_pred
            guidance_term = self.forward_model.restore(residual, measurement_state) / (self.forward_model.std * sigma_t)

            return self.score(x, t) + guidance_term

        # Use integrator to compute next state
        integrator_state_next = self.integrator(state.integrator_state, modified_score)
        state_next = CondDenoiserState(integrator_state_next, state.log_weights)

        return state_next

    def y_noiser(self, key: PRNGKeyArray, t: float, measurement_state: MeasurementState) -> SDEState:
        r"""
        Generate y^{(t)} = \sqrt{\bar{\alpha}_t} y + \sqrt{1-\bar{\alpha}_t} A_\xi \epsilon
        """
        y_0 = measurement_state.y
        alpha_t = self.sde.signal_level(t)

        # Noise y_t as the mean to keep deterministic sampling methods deterministic
        # rndm = jax.random.normal(key, y_0.shape)
        res = alpha_t * y_0 #+ noise_level * rndm

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
        integrator_state, log_weights = state_next.integrator_state, state_next.log_weights
        x_t = state_next.integrator_state.position
        rng_key, rng_key_resample = jax.random.split(rng_key)

        t = self.integrator.timer(state_next.integrator_state.step)
        sigma_t_squared = self.sde.noise_level(t) ** 2

        keys = jax.random.split(rng_key, x_t.shape[0])
        y_t = jax.vmap(self.y_noiser, in_axes=(0, 0, None))(keys, t, measurement_state).position
        f_x_t = jax.vmap(self.forward_model.apply, in_axes=(0, None))(x_t, measurement_state)
        residual = jnp.linalg.norm(y_t - f_x_t, axis=-1)

        t_prev = self.integrator.timer(state_next.integrator_state.step - 1)
        sigma_prev_squared = self.sde.noise_level(t_prev) ** 2
        y_t_prev = jax.vmap(self.y_noiser, in_axes=(0, 0, None))(keys, t_prev, measurement_state).position
        f_x_t_prev = jax.vmap(self.forward_model.apply, in_axes=(0, None))(x_t, measurement_state)
        residual_prev = jnp.linalg.norm(y_t_prev - f_x_t_prev, axis=-1)
        # compute log weights
        # log_weights = - 0.5 * residual**2 / (self.forward_model.std**2 * alpha_t**2)  - 0.5 * (residual_prev**2) / (self.forward_model.std**2 * alpha_prev**2)
        log_weights = - 0.5 * residual**2  - 0.5 * (residual_prev**2)
        log_weights = normalize_log_weights(log_weights)
        position, log_weights = resample_particles(
            integrator_state.position, log_weights, rng_key_resample, self.ess_low, self.ess_high
        )

        integrator_state_next = state_next.integrator_state._replace(position=position)
        return CondDenoiserState(integrator_state_next, log_weights)
