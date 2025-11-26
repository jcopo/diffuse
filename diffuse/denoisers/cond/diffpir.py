# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Plug-and-Play Diffusion for Image Restoration (DiffPIR)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.base_forward_model import MeasurementState
from diffuse.integrator.base import IntegratorState


@dataclass
class DiffPIRDenoiser(CondDenoiser):
    """Plug-and-Play Diffusion for Image Restoration (DiffPIR).

    Alternates between diffusion denoising and data-fidelity steps:
    1. Denoise to get x̂₀
    2. Data-fidelity step (proximal for linear, gradient for nonlinear)
    3. Add noise for next step

    Data-fidelity regularization: ρ = 2λσₙ² / σₜ²
    - Linear:    x̂₀ = (AᵀA + ρI)⁻¹(Aᵀy + ρx̂₀_diffusion)
    - Nonlinear: x̂₀ = x̂₀_diffusion - (1/ρ)∇||Ax̂₀ - y||²

    Args:
        integrator: Numerical integrator for the reverse SDE
        model: Diffusion model defining the forward process
        predictor: Score/noise/velocity predictor
        forward_model: Forward measurement operator
        sigma_n: Measurement noise std
        lamb: Regularization strength (prior vs likelihood balance)
        xi: Noise injection ratio (0=deterministic, 1=stochastic)
        linear: Use proximal step (linear operator) or gradient step
        cg_tol: CG solver tolerance (linear mode)
        cg_maxiter: CG max iterations (linear mode)
    """

    sigma_n: float = 0.05
    lamb: float = 1.0
    xi: float = 0.5
    linear: bool = False
    cg_tol: float = 1e-5
    cg_maxiter: int = 32

    def __post_init__(self):
        if self.sigma_n <= 0:
            raise ValueError("sigma_n must be strictly positive.")
        if self.lamb <= 0:
            raise ValueError("lamb must be strictly positive.")
        if not 0.0 <= self.xi <= 1.0:
            raise ValueError("xi must be in [0, 1].")

    def _compute_rho(self, sigma_t: Array) -> Array:
        """ρ = 2λσₙ² / σₜ², clamped to [0.01, 100] for stability."""
        rho = 2.0 * self.lamb * (self.sigma_n**2) / (sigma_t**2 + 1e-8)
        return jnp.clip(rho, 0.01, 100.0)

    def _linear_proximal_step(
        self,
        x0_diffusion: Array,
        measurement_state: MeasurementState,
        rho: Array,
    ) -> Array:
        """Solve (AᵀA + ρI)x = Aᵀy + ρx̂₀_diffusion via CG."""
        y = measurement_state.y
        Aty = self.forward_model.adjoint(y, measurement_state)
        rhs = Aty + rho * x0_diffusion

        def matvec(v: Array) -> Array:
            Av = self.forward_model.apply(v, measurement_state)
            AtAv = self.forward_model.adjoint(Av, measurement_state)
            return AtAv + rho * v

        sol, info = cg(matvec, rhs, x0=x0_diffusion, tol=self.cg_tol, maxiter=self.cg_maxiter)

        # Fallback if CG fails
        sol = jnp.where(jnp.isfinite(sol), sol, x0_diffusion)
        sol = jax.lax.cond(info == 0, lambda _: sol, lambda _: x0_diffusion, operand=None)
        return sol

    def _nonlinear_gradient_step(
        self,
        x0_diffusion: Array,
        measurement_state: MeasurementState,
        rho: Array,
    ) -> Array:
        """x̂₀ = x̂₀_diffusion - (1/ρ)∇||Ax̂₀ - y||²."""
        y = measurement_state.y

        def data_fidelity_loss(x: Array) -> Array:
            residual = self.forward_model.apply(x, measurement_state) - y
            return 0.5 * jnp.sum(residual**2)

        gradient = jax.grad(data_fidelity_loss)(x0_diffusion)
        update = gradient / rho

        # Clip update to at most 2x signal magnitude
        update_norm = jnp.linalg.norm(update) + 1e-8
        x0_norm = jnp.linalg.norm(x0_diffusion) + 1e-8
        update = update * jnp.minimum(1.0, 2.0 * x0_norm / update_norm)

        return x0_diffusion - update

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        x_t = state.integrator_state.position
        step_idx = state.integrator_state.step
        t_current = self.integrator.timer(step_idx)
        t_next = self.integrator.timer(step_idx + 1)

        alpha_t = self.model.signal_level(t_current)
        sigma_t = self.model.noise_level(t_current)
        alpha_next = self.model.signal_level(t_next)
        sigma_next = self.model.noise_level(t_next)

        x_t_normalized = x_t / (alpha_t + 1e-8)
        x0_diffusion = self.model.tweedie(SDEState(x_t, t_current), self.predictor.score).position

        # Data-fidelity with rho based on effective noise level
        rho = self._compute_rho(sigma_t * alpha_t)
        if self.linear:
            x0_hat = self._linear_proximal_step(x0_diffusion, measurement_state, rho)
        else:
            x0_hat = self._nonlinear_gradient_step(x0_diffusion, measurement_state, rho)

        # DiffPIR update: x_next = x̂₀ + σ_next * (√ξ * noise + √(1-ξ) * effect)
        effect = (x_t_normalized - x0_hat) / (sigma_t + 1e-8)
        noise = jax.random.normal(rng_key, x0_hat.shape, dtype=x0_hat.dtype)
        x_next = x0_hat + sigma_next * (jnp.sqrt(self.xi) * noise + jnp.sqrt(1.0 - self.xi) * effect)

        # Scale by α_next except at final step
        is_final_step = sigma_next < 1e-4
        x_next = jnp.where(is_final_step, x_next, x_next * alpha_next)

        integrator_state_next = IntegratorState(position=x_next, rng_key=rng_key, step=step_idx + 1)
        return state._replace(integrator_state=integrator_state_next)


__all__ = ["DiffPIRDenoiser"]
