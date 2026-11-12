# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState


@dataclass
class TMPDenoiser(CondDenoiser):
    """TMPD-D step implemented as: (1) unconditional integrator step (ancestral),
    (2) compute Tweedie mean and measurement correction, (3) linear correction
    x_prev = x_prev_uncond + B * (m0_y - m0).
    """

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Single step of TMPD-D sampling.

        Runs the integrator unconditionally, then applies a linear data-fidelity correction.
        """
        y = measurement_state.y

        # 1) extract current sample x_n and current time t_n
        x_n = state.integrator_state.position
        t_n = self.integrator.timer(state.integrator_state.step)

        # Note: In this codebase, signal_level returns sqrt(alpha_bar), noise_level returns sqrt(1-alpha_bar)
        # For interpolation: x_t = alpha_t * x_0 + sigma_t * epsilon
        alpha_n = self.model.signal_level(t_n)  # sqrt(alpha_bar_n)
        sigma_n = self.model.noise_level(t_n)  # sqrt(1 - alpha_bar_n)

        # 2) compute unconditional Tweedie mean m0|n using the codebase's convention
        # m0 = (x_t + sigma_t^2 * score) / alpha_t
        score_xn = self.predictor.score(x_n, t_n)
        m0 = (x_n + sigma_n**2 * score_xn) / alpha_n

        # 3) run the integrator UNCONDITIONALLY using the base predictor (no measurement)
        #    => this yields x_{n-1}^uncond and its time (we need alpha_{n-1})
        uncond_integrator_state = self.integrator(state.integrator_state, self.predictor)
        x_prev_uncond = uncond_integrator_state.position
        t_prev = self.integrator.timer(uncond_integrator_state.step)

        # compute alpha_{n-1} from returned time
        alpha_prev = self.model.signal_level(t_prev)  # α_{n-1}

        # 4) compute the measurement-corrected mean m0^y (Eq. 13) by solving a linear system
        #    We follow the same CG + jvp pattern as in the original code.

        # tweedie mapping m0(x) — used only for jvp to compute Jacobian-vector products
        def tweedie_fn(x):
            # returns m0|t(x) for *the same* t = t_n (we linearize around x_n)
            # Using the codebase's convention: m0 = (x + sigma^2 * score) / alpha
            return (x + sigma_n**2 * self.predictor.score(x, t_n)) / alpha_n

        # Build matvec on measurement-space vectors for CG: (H * (sigma_n^2/alpha_n * J) * H^T + sigma_y^2 I) v
        # Implementation detail: we compute J @ restored_v (J is Jacobian of tweedie_fn at x_n)
        # where restored_v is H^T-space -> image-space mapping (via forward_model.restore).
        def matvec_meas(v_meas):
            # v_meas is in measurement space (dy)
            # restore maps measurement-space vector to image-space; expected to be H^T(v_meas) or similar
            restored = self.forward_model.restore(v_meas, measurement_state)  # image-space vector
            # compute J@restored  (jvp returns (tweedie_fn(x_n), J@restored))
            _, jvp_tangent = jax.jvp(tweedie_fn, (x_n,), (restored,))
            # measured_tangent := H @ (J @ restored)
            measured_tangent = self.forward_model.apply(jvp_tangent, measurement_state)  # measurement-space
            # scale factor appearing in the paper's linear system: (1 - alpha_bar_n) / sqrt(alpha_bar_n) = sigma_n^2 / alpha_n
            scale = sigma_n**2 / alpha_n
            return scale * measured_tangent + (self.forward_model.std**2) * v_meas

        # RHS = y - H @ m0  (measurement residual)
        rhs = y - self.forward_model.apply(m0, measurement_state)

        # Solve linear system with CG in measurement space: mat * delta = rhs
        # Use a small maxiter (paper and your code used e.g. 3), adjust if needed.
        delta_meas, info = jax.scipy.sparse.linalg.cg(matvec_meas, rhs, maxiter=10, tol=1e-5)

        # Check for NaN in CG solution and fall back to zero correction if needed
        delta_meas = jnp.where(jnp.isnan(delta_meas), 0.0, delta_meas)

        # convert delta_meas to image-space direction via restore (this is ~ H^T @ delta scaled appropriately)
        restored_delta = self.forward_model.restore(delta_meas, measurement_state)
        # J @ restored_delta  (this will be equal to the additive correction to m0)
        _, jvp_correction = jax.jvp(tweedie_fn, (x_n,), (restored_delta,))

        # Guard against NaN in jvp
        jvp_correction = jnp.where(jnp.isnan(jvp_correction), 0.0, jvp_correction)
        m0_y = m0 + jvp_correction  # measurement corrected mean

        # 5) compute B = sqrt(alpha_bar_{n-1} - alpha_bar_n) / (1 - alpha_bar_n)
        #    where alpha_bar = alpha^2 in our notation
        #    (uses identity beta_n = 1 - alpha_bar_n / alpha_bar_{n-1})
        numerator = jnp.sqrt(jnp.clip(alpha_prev**2 - alpha_n**2, min=0.0))  # guard numerical negative
        denom = jnp.maximum(1.0 - alpha_n**2, 1e-8)  # prevent division by zero near t=0
        B = numerator / denom

        # 6) linear correction of the unconditional integrator result
        correction = B * (m0_y - m0)
        # Guard against NaN or inf in correction
        correction = jnp.where(jnp.isnan(correction) | jnp.isinf(correction), 0.0, correction)
        x_prev_cond = x_prev_uncond + correction

        # Final safety check
        x_prev_cond = jnp.where(jnp.isnan(x_prev_cond), x_prev_uncond, x_prev_cond)

        # 7) pack into state and return
        next_integrator_state = uncond_integrator_state._replace(position=x_prev_cond)
        state_next = CondDenoiserState(next_integrator_state, state.log_weights)
        return state_next
