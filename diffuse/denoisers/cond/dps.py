from dataclasses import dataclass

import jax
import jax.experimental
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.base_forward_model import MeasurementState


@dataclass
class DPSDenoiser(CondDenoiser):
    """Conditional denoiser using Diffusion Posterior Sampling with Tweedie's formula"""

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Single step of DPS sampling.

        Modifies the score to include measurement term and uses integrator for the update.
        """
        y_meas = measurement_state.y

        # Define modified score function that includes measurement term
        def modified_score(x: Array, t: float) -> Array:
            # Apply Tweedie's formula
            denoised = self.sde.tweedie(SDEState(x, t), self.score).position

            # Compute residual and guidance
            residual = jnp.linalg.norm(y_meas - self.forward_model.apply(denoised, measurement_state)) ** 2
            _, guidance = jax.value_and_grad(lambda x: jnp.linalg.norm(y_meas - self.forward_model.apply(x, measurement_state)) ** 2)(x)

            # Compute guidance scale
            xi = 1 / (jnp.sqrt(residual) + 1e-3)

            # Return modified score
            return self.score(x, t) - xi * guidance

        # Use integrator to compute next state
        integrator_state_next = self.integrator(state.integrator_state, modified_score)
        state_next = CondDenoiserState(integrator_state_next, state.log_weights)

        return state_next
