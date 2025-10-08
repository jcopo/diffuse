from dataclasses import dataclass

import jax
import jax.experimental
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.base_forward_model import MeasurementState
from diffuse.predictor import Predictor
from diffuse.integrator.base import IntegratorState


@dataclass
class DPSDenoiser(CondDenoiser):
    """Conditional denoiser using Diffusion Posterior Sampling with Tweedie's formula"""
    epsilon: float = 1e-3  # Small constant to avoid division by zero in norm calculation
    zeta: float = .5
    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Single step of DPS sampling.

        Implements the DPS algorithm as originally intended:
        1. Compute Tweedie estimate at current position
        2. Take unconditional diffusion step with integrator
        3. Apply measurement-consistency gradient correction

        This approach works correctly with second-order integrators (Heun, DPM++, etc.)
        because the integrator sees the true unconditional score/velocity.
        """
        y_meas = measurement_state.y
        position_current = state.integrator_state.position
        t_current = self.integrator.timer(state.integrator_state.step)

        # Compute measurement loss and gradient at current position
        # The gradient includes the chain rule through Tweedie's formula
        def measurement_loss(x: Array) -> Array:
            # Tweedie estimate: x̂_0 from x at time t
            denoised = self.model.tweedie(SDEState(x, t_current), self.predictor.score).position
            # Measurement consistency loss: ||y - A(x̂_0)||²
            residual = y_meas - self.forward_model.apply(denoised, measurement_state)
            return jnp.sum(residual ** 2)

        # Compute loss value and gradient ∇_x ||y - A(x̂_0)||²
        loss_val, gradient = jax.value_and_grad(measurement_loss)(position_current)

        # Adaptive guidance scale (normalized by residual magnitude)
        zeta = self.zeta / (jnp.sqrt(loss_val) + self.epsilon)

        # Take unconditional integrator step (works with any integrator)
        integrator_state_uncond = self.integrator(state.integrator_state, self.predictor)

        # Apply measurement correction: x_{i-1} = x'_{i-1} - ζ ∇_x ||y - A(x̂_0)||²
        position_corrected = integrator_state_uncond.position - zeta * gradient

        # Create next state with corrected position
        integrator_state_next = integrator_state_uncond._replace(position=position_corrected)
        state_next = state._replace(integrator_state=integrator_state_next)

        return state_next
