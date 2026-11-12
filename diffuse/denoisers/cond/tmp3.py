# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from dataclasses import dataclass

import jax
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.diffusion.sde import SDEState


@dataclass
class TMPDenoiser(CondDenoiser):
    """Conditional denoiser using Tweedie's Moments from https://arxiv.org/pdf/2310.06721v3

    Implements TMP by computing the Tweedie-based measurement correction at the current
    position, then applying it as an additive correction to the unconditional integrator step.
    """

    zeta: float = 0.1  # Correction strength parameter

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Single step of TMP sampling.

        Implements the TMP algorithm as originally intended:
        1. Compute Tweedie-based measurement correction at current position
        2. Take unconditional diffusion step with integrator
        3. Apply measurement correction to result

        This approach works correctly with second-order integrators (Heun, DPM++, etc.)
        because the integrator sees the true unconditional score/velocity.
        """
        y_meas = measurement_state.y
        position_current = state.integrator_state.position
        t_current = self.integrator.timer(state.integrator_state.step)

        # Get noise and signal levels at current time
        sigma_t = self.model.noise_level(t_current)
        alpha_t = self.model.signal_level(t_current)
        scale = sigma_t**2 / alpha_t  # Variance scaling for the linear system
        scale = 1

        # Compute Tweedie estimate at current position
        def tweedie_fn(x: Array) -> Array:
            """Tweedie's formula: E[x_0 | x_t]"""
            return self.model.tweedie(SDEState(x, t_current), self.predictor.score).position

        # Set up the measurement-corrected Tweedie estimation problem
        # We solve: (H * scale * J * H^T + σ_y² I) * delta = y - H*x̂_0
        def matvec(v: Array) -> Array:
            """Matrix-vector product for CG solver."""
            restored_v = self.forward_model.restore(v, measurement_state)
            _, jvp_tangent = jax.jvp(tweedie_fn, (position_current,), (restored_v,))
            measured_tangent = self.forward_model.apply(jvp_tangent, measurement_state)
            return scale * measured_tangent + self.forward_model.std**2 * v

        # Compute unconditional Tweedie estimate and measurement residual
        denoised = tweedie_fn(position_current)
        residual = y_meas - self.forward_model.apply(denoised, measurement_state)

        # Solve for measurement correction in measurement space
        delta_meas, _ = jax.scipy.sparse.linalg.cg(matvec, residual, maxiter=10)

        # Map correction back to image space via Jacobian
        restored_delta = self.forward_model.restore(delta_meas, measurement_state)
        _, guidance_correction = jax.jvp(tweedie_fn, (position_current,), (restored_delta,))

        # Take unconditional integrator step (works with any integrator)
        integrator_state_uncond = self.integrator(state.integrator_state, self.predictor)

        # Apply measurement correction with adaptive scaling
        position_corrected = integrator_state_uncond.position + self.zeta * guidance_correction

        # Create next state with corrected position
        integrator_state_next = integrator_state_uncond._replace(position=position_corrected)
        state_next = state._replace(integrator_state=integrator_state_next)

        return state_next
