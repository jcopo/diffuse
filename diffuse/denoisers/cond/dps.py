# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.base_forward_model import MeasurementState


@dataclass
class DPSDenoiser(CondDenoiser):
    """Conditional denoiser using Diffusion Posterior Sampling (DPS).

    Implements DPS which uses Tweedie's formula for denoising and applies
    measurement-consistency gradient corrections at each sampling step.

    Args:
        integrator: Numerical integrator for solving the reverse SDE
        model: Diffusion model defining the forward process
        predictor: Predictor for computing score/noise/velocity
        forward_model: Forward measurement operator
        epsilon: Numerical stability parameter (default: 1e-3)
        zeta: Gradient step size parameter (default: 1e-2)

    References:
        Chung, H., Kim, J., Mccann, M. T., Klasky, M. L., & Ye, J. C. (2022).
        Diffusion posterior sampling for general noisy inverse problems. arXiv:2209.14687
    """

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

        Args:
            rng_key: Random number generator key
            state: Current conditional denoiser state
            measurement_state: Measurement information

        Returns:
            Updated conditional denoiser state
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
