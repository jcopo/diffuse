from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.base_forward_model import MeasurementState


@dataclass
class DPSDenoiser(CondDenoiser):
    """Conditional denoiser using Diffusion Posterior Sampling with Tweedie's formula"""

    epsilon: float = 1e-3
    zeta: float = 1e-2

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Single step of DPS sampling.

        Implements the DPS algorithm:
        1. Compute Tweedie estimate at current position
        2. Take unconditional diffusion step with integrator
        3. Apply measurement-consistency gradient correction

        This approach works correctly with second-order integrators (Heun, DPM++, etc.)
        """
        y_meas = measurement_state.y
        position_current = state.integrator_state.position
        t_current = self.integrator.timer(state.integrator_state.step)

        def measurement_loss(x: Array) -> Array:
            denoised = self.model.tweedie(SDEState(x, t_current), self.predictor.score).position
            # Measurement consistency loss: ||y - A(x̂_0)||²
            residual = y_meas - self.forward_model.apply(denoised, measurement_state)
            return jnp.sum(residual**2)

        loss_val, gradient = jax.value_and_grad(measurement_loss)(position_current)
        zeta = self.zeta / (jnp.sqrt(loss_val) + self.epsilon)

        integrator_state_uncond = self.integrator(state.integrator_state, self.predictor)
        position_corrected = integrator_state_uncond.position - zeta * gradient

        integrator_state_next = integrator_state_uncond._replace(position=position_corrected)
        state_next = state._replace(integrator_state=integrator_state_next)

        return state_next
