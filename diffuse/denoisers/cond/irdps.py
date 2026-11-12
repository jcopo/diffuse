# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Integrated Riemannian Diffusion Posterior Sampling (iRDPS).

This module implements iRDPS, a variant of RDPS that uses full ODE/SDE integration
to estimate the clean sample x₀ instead of Tweedie's formula.

The key difference from standard RDPS:
- RDPS: x̂₀ = Tweedie(xₜ) = (xₜ - σₜ·ε_θ(xₜ,t)) / αₜ
- iRDPS: x̂₀ = ODE_integrate(xₜ, t → 0)

This is more expensive but can be more accurate, especially at high noise levels
where Tweedie is a poor approximation. The ODE integration provides a multi-step
refinement of the clean estimate.

After obtaining x̂₀ via integration, we apply the same natural gradient correction
as RDPS: x̂₀_dc = x̂₀ + ζ · (AᵀA + λI)⁻¹ Aᵀ(y - Ax̂₀)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.integrator.base import IntegratorState
from diffuse.timer.base import Timer, VpTimer, HeunTimer


@dataclass
class IntegratedRDPSDenoiser(CondDenoiser):
    """Integrated Riemannian Diffusion Posterior Sampling (iRDPS).

    Uses full ODE integration (instead of Tweedie) to estimate x₀, then applies
    natural gradient correction for measurement guidance.

    The algorithm:
    1. Integrate: x̂₀ = ODE(xₜ, t → 0) using n_inner_steps
    2. Correct: x̂₀_dc = x̂₀ + ζ · (AᵀA + λI)⁻¹ Aᵀ(y - Ax̂₀)
    3. Re-noise: x_{t-1} = α_{t-1} · x̂₀_dc + σ_{t-1} · ε

    Args:
        diffusion_steps: Number of ODE steps for reverse diffusion to x₀. Default: 50.
        lambda_reg: Fixed regularization for CG solver. Default: 1e-2.
        zeta: Step size for data-fidelity correction. Default: 1.0.
        cg_maxiter: Maximum CG iterations. Default: 10.
        cg_tol: CG convergence tolerance. Default: 1e-5.
    """

    diffusion_steps: int = 50
    lambda_reg: float = 1e-2
    zeta: float = 1.0
    cg_maxiter: int = 10
    cg_tol: float = 1e-5

    def _integrate_to_clean(
        self,
        x_t: Array,
        t_start: Array,
        rng_key: PRNGKeyArray,
    ) -> Array:
        """Integrate the ODE from t_start to ~0 using a proper mini-timer."""
        from diffuse.integrator.deterministic import EulerIntegrator

        base_timer = self.integrator.timer
        eps = getattr(base_timer, "eps", 0.001)
        t_start_clipped = jnp.clip(t_start, eps * 1.5, 1.0 - 1e-4)

        mini_timer = VpTimer(n_steps=self.diffusion_steps, eps=eps, tf=t_start_clipped)
        mini_integrator = EulerIntegrator(model=self.model, timer=mini_timer)
        mini_state = mini_integrator.init(x_t, rng_key)

        def ode_step(state: IntegratorState, _):
            return mini_integrator(state, self.predictor), None

        final_state, _ = jax.lax.scan(ode_step, mini_state, xs=None, length=self.diffusion_steps)
        return final_state.position

    def _natural_gradient_correction(
        self,
        x0_hat: Array,
        measurement_state: MeasurementState,
    ) -> Array:
        """Compute natural gradient correction: (AᵀA + λI)⁻¹ Aᵀ(y - Ax̂₀)."""
        y = measurement_state.y
        y_pred = self.forward_model.apply(x0_hat, measurement_state)
        residual = y - y_pred
        rhs = self.forward_model.adjoint(residual, measurement_state)

        def matvec(v: Array) -> Array:
            Av = self.forward_model.apply(v, measurement_state)
            AtAv = self.forward_model.adjoint(Av, measurement_state)
            return AtAv + self.lambda_reg * v

        direction, _ = jax.scipy.sparse.linalg.cg(
            matvec,
            rhs,
            x0=rhs / (self.lambda_reg + 1e-8),
            tol=self.cg_tol,
            maxiter=self.cg_maxiter,
        )
        return jnp.where(jnp.isfinite(direction), direction, rhs / (self.lambda_reg + 1e-8))

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Single step of iRDPS sampling.

        1. Integrate ODE from t to ~0 to get x̂₀
        2. Apply natural gradient correction
        3. Forward diffuse to next noise level (like DAPS)
        """
        x_t = state.integrator_state.position
        step_idx = state.integrator_state.step
        t_current = self.integrator.timer(step_idx)
        t_next = self.integrator.timer(step_idx + 1)

        key_diffuse, key_forward = jax.random.split(rng_key)

        # 1. Reverse diffuse: ODE from t to ~0 to get x̂₀
        x0_hat = self._integrate_to_clean(x_t, t_current, key_diffuse)

        # 2. Natural gradient correction
        correction = self._natural_gradient_correction(x0_hat, measurement_state)
        x0_corrected = x0_hat + self.zeta * correction

        # 3. Forward diffuse: add fresh noise to reach t_next
        alpha_next = self.model.signal_level(t_next)
        sigma_next = self.model.noise_level(t_next)
        noise = jax.random.normal(key_forward, x_t.shape, dtype=x_t.dtype)
        x_next = alpha_next * x0_corrected + sigma_next * noise

        # Handle final step (no noise)
        is_final = sigma_next < 1e-6
        x_next = jnp.where(is_final, x0_corrected, x_next)

        integrator_state_next = IntegratorState(
            position=x_next,
            rng_key=key_forward,
            step=step_idx + 1,
        )
        return state._replace(integrator_state=integrator_state_next)
