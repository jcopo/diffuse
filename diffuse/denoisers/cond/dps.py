from dataclasses import dataclass

import jax
import jax.experimental
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.base_forward_model import MeasurementState
from diffuse.utils.plotting import sigle_plot


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

        # Define modified score function that includes guidance term
        def modified_score(x: Array, t: float) -> Array:
            def norm_tweedie(x: Array):
                # Apply Tweedie's formula
                denoised = self.sde.tweedie(SDEState(x, t), self.score).position
                norm = jnp.linalg.norm(y_meas - self.forward_model.apply(denoised, measurement_state)) ** 2
                return norm, denoised  # Return both norm and denoised

            # Compute residual and guidance
            (val, denoised), guidance = jax.value_and_grad(norm_tweedie, has_aux=True)(x)

            # Plot outside the differentiated function
            jax.experimental.io_callback(sigle_plot, None, denoised, t)

            # Compute guidance scale
            zeta = 3. / (jnp.sqrt(val) + 1e-3)

            return self.score(x, t) - zeta * guidance

        # Use integrator to compute next state
        integrator_state_next = self.integrator(state.integrator_state, modified_score)
        state_next = CondDenoiserState(integrator_state_next, state.log_weights)

        return state_next
