from dataclasses import dataclass
from typing import Optional

import jax
import jax.experimental
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.base_forward_model import MeasurementState
from diffuse.predictor import Predictor


@dataclass
class DPSDenoiser(CondDenoiser):
    """Conditional denoiser using Diffusion Posterior Sampling with Tweedie's formula

    Args:
        zeta: Guidance scale. If None, uses adaptive scaling based on residual norm.
              If float, uses fixed guidance scale.
        epsilon: Small constant for numerical stability (default: 1e-8)
    """
    zeta: Optional[float] = None
    epsilon: float = 1e-8

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
        def modified_score(x: Array, t: Array) -> Array:
            def norm_tweedie(x: Array):
                # Apply Tweedie's formula
                denoised = self.sde.tweedie(SDEState(x, t), self.predictor.score).position
                residual = y_meas - self.forward_model.apply(denoised, measurement_state)

                norm = jnp.sqrt(jnp.sum(residual ** 2) + self.epsilon)
                return norm, denoised  # Return both norm and denoised

            # Compute residual and guidance
            (val, denoised), guidance = jax.value_and_grad(norm_tweedie, has_aux=True)(x)

            # Compute guidance scale
            zeta = 1.0 / (val + 1e-3)

            return self.predictor.score(x, t) - zeta * guidance

        # Create modified predictor for guidance
        modified_predictor = Predictor(self.sde, modified_score, "score")

        # Use integrator to compute next state
        integrator_state_next = self.integrator(state.integrator_state, modified_predictor)
        state_next = CondDenoiserState(integrator_state_next, state.log_weights)

        return state_next
