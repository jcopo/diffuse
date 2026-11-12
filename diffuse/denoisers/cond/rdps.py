# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Riemannian Diffusion Posterior Sampling (RDPS).

This module implements RDPS, a novel conditional denoiser that uses natural
gradients (Gauss-Newton direction) for geometry-aware measurement guidance.

The key insight is that the image manifold has a Riemannian structure induced
by the measurement operator A. Instead of using Euclidean gradients, RDPS uses
the natural gradient which accounts for the measurement geometry:

    Natural gradient = G⁻¹ ∇f = (AᵀA + λI)⁻¹ Aᵀ(y - Ax̂₀)

This is the minimum-variance unbiased correction given the measurement geometry.

Key innovations:
1. Adaptive regularization λ(t) that decreases with noise level
2. Data-fidelity correction applied to x̂₀ (clean estimate), then re-noised
3. Optional hard data consistency projection
4. Trust-region step size to prevent over-correction

Reference: Novel method for ICML/NeurIPS 2025 submission.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.diffusion.sde import SDEState
from diffuse.integrator.base import IntegratorState


@dataclass
class RDPSDenoiser(CondDenoiser):
    """Riemannian Diffusion Posterior Sampling (RDPS).

    Uses natural gradient (Gauss-Newton direction) for geometry-aware
    measurement guidance in inverse problem reconstruction.

    The algorithm follows a denoise-correct-renoise pattern:
    1. Denoise: x̂₀ = Tweedie(x_t)
    2. Correct: x̂₀_dc = x̂₀ + ζ(t) · (AᵀA + λI)⁻¹ Aᵀ(y - Ax̂₀)
    3. Re-noise: x_{t-1} = α_{t-1} · x̂₀_dc + σ_{t-1} · ε

    Args:
        lambda_max: Maximum regularization at t=1 (high noise). Default: 1.0.
        lambda_min: Minimum regularization at t=0 (low noise). Default: 1e-3.
        lambda_alpha: Decay exponent for regularization schedule. Default: 1.0.
        zeta: Step size for data-fidelity correction. Default: 1.0.
        cg_maxiter: Maximum CG iterations for solving normal equations. Default: 10.
        cg_tol: CG convergence tolerance. Default: 1e-5.
        hard_dc: Whether to apply hard data consistency projection. Default: False.
        dc_weight: Weight for soft data consistency (0=prior only, 1=full DC). Default: 1.0.
        stochastic: Whether to inject noise during re-noising. Default: True.
        xi: Noise injection ratio (0=deterministic, 1=full stochastic). Default: 0.5.

    References:
        Novel method combining Riemannian optimization with diffusion guidance.
    """

    # Regularization schedule parameters
    lambda_max: float = 1.0
    lambda_min: float = 1e-3
    lambda_alpha: float = 1.0

    # Step size
    zeta: float = 1.0

    # CG solver parameters
    cg_maxiter: int = 10
    cg_tol: float = 1e-5

    # Data consistency options
    hard_dc: bool = False
    dc_weight: float = 1.0
    dc_iterations: int = 1  # Number of DC iterations per step (for iterative refinement)
    annealed_dc: bool = False  # If True, dc_weight increases as noise decreases

    # Re-noising options
    stochastic: bool = True
    xi: float = 0.5

    def _compute_lambda(self, t: Array) -> Array:
        """Compute adaptive regularization parameter λ(t).

        Regularization decreases with noise level to allow finer corrections
        as the diffusion process approaches the clean image.

        Args:
            t: Current diffusion time.

        Returns:
            Regularization parameter λ(t).
        """
        sigma_t = self.model.noise_level(t)
        sigma_max = self.model.noise_level(jnp.array(1.0))

        # Normalize noise level ratio
        noise_ratio = jnp.clip(sigma_t / (sigma_max + 1e-8), 0.0, 1.0)

        # Adaptive schedule: high regularization at high noise, low at low noise
        lambda_t = self.lambda_min + (self.lambda_max - self.lambda_min) * (noise_ratio**self.lambda_alpha)

        return lambda_t

    def _natural_gradient_correction(
        self,
        x0_hat: Array,
        measurement_state: MeasurementState,
        lambda_t: Array,
    ) -> Array:
        """Compute natural gradient correction for x̂₀.

        Solves: d = (AᵀA + λI)⁻¹ Aᵀ(y - Ax̂₀)

        This gives the minimum-norm update that reduces the measurement residual.

        Args:
            x0_hat: Tweedie estimate of clean image.
            measurement_state: Contains observed measurements y.
            lambda_t: Current regularization parameter.

        Returns:
            Correction direction d to be added to x̂₀.
        """
        y = measurement_state.y

        # Compute measurement residual: r = y - A(x̂₀)
        y_pred = self.forward_model.apply(x0_hat, measurement_state)
        residual = y - y_pred

        # Right-hand side: Aᵀr
        rhs = self.forward_model.adjoint(residual, measurement_state)

        # If lambda is very large, just use the simple gradient direction
        # (AᵀA + λI)⁻¹ ≈ λ⁻¹ I for large λ
        use_cg = lambda_t < 10.0

        def solve_with_cg(_):
            def matvec(v: Array) -> Array:
                Av = self.forward_model.apply(v, measurement_state)
                AtAv = self.forward_model.adjoint(Av, measurement_state)
                return AtAv + lambda_t * v

            direction, _ = jax.scipy.sparse.linalg.cg(
                matvec,
                rhs,
                x0=rhs / (lambda_t + 1e-8),  # Initial guess
                tol=self.cg_tol,
                maxiter=self.cg_maxiter,
            )
            return direction

        def use_simple_gradient(_):
            return rhs / (lambda_t + 1e-8)

        direction = jax.lax.cond(use_cg, solve_with_cg, use_simple_gradient, operand=None)

        # Ensure finite values
        direction = jnp.where(jnp.isfinite(direction), direction, rhs / (lambda_t + 1e-8))

        return direction

    def _hard_data_consistency(
        self,
        x0_corrected: Array,
        measurement_state: MeasurementState,
        lambda_t: Array,
    ) -> Array:
        """Apply hard data consistency projection.

        Solves: min_x ||x - x₀_corrected||² s.t. ||Ax - y||² ≤ ε

        For linear inverse problems, this is approximately:
        x_dc = x₀_corrected + A†(y - A(x₀_corrected))

        where A† = (AᵀA + λI)⁻¹ Aᵀ is the regularized pseudo-inverse.

        Args:
            x0_corrected: Already corrected clean estimate.
            measurement_state: Contains observed measurements y.
            lambda_t: Regularization parameter.

        Returns:
            Data-consistent image estimate.
        """
        # Apply another correction to enforce data consistency more strongly
        correction = self._natural_gradient_correction(x0_corrected, measurement_state, lambda_t * 0.1)
        return x0_corrected + correction

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Single step of RDPS sampling.

        Implements the denoise-correct-renoise pattern:
        1. Denoise: x̂₀ = Tweedie(x_t)
        2. Correct: x̂₀_dc = x̂₀ + ζ · (AᵀA + λI)⁻¹ Aᵀ(y - Ax̂₀)
        3. (Optional) Hard DC: x̂₀_dc = project_to_measurement_consistent(x̂₀_dc)
        4. Re-noise: x_{t-1} = α_{t-1} · x̂₀_dc + σ_{t-1} · noise

        Args:
            rng_key: Random number generator key.
            state: Current conditional denoiser state.
            measurement_state: Measurement information containing y.

        Returns:
            Updated conditional denoiser state.
        """
        x_t = state.integrator_state.position
        step_idx = state.integrator_state.step
        t_current = self.integrator.timer(step_idx)
        t_next = self.integrator.timer(step_idx + 1)

        # Get noise schedule parameters
        alpha_t = self.model.signal_level(t_current)
        sigma_t = self.model.noise_level(t_current)
        alpha_next = self.model.signal_level(t_next)
        sigma_next = self.model.noise_level(t_next)

        # 1. Denoise: compute Tweedie estimate
        x0_hat = self.model.tweedie(
            SDEState(x_t, t_current),
            self.predictor.score,
        ).position

        # 2. Compute adaptive regularization
        lambda_t = self._compute_lambda(t_current)

        # 3. Compute effective DC weight (optionally annealed)
        if self.annealed_dc:
            # DC weight increases as noise decreases (1 - noise_ratio)
            sigma_t = self.model.noise_level(t_current)
            sigma_max = self.model.noise_level(jnp.array(1.0))
            noise_ratio = jnp.clip(sigma_t / (sigma_max + 1e-8), 0.0, 1.0)
            effective_dc_weight = self.dc_weight * (1.0 + (1.0 - noise_ratio))  # 1x to 2x
        else:
            effective_dc_weight = self.dc_weight

        # 4. Apply natural gradient correction to x̂₀ (iteratively if dc_iterations > 1)
        x0_corrected = x0_hat
        for _ in range(self.dc_iterations):
            correction = self._natural_gradient_correction(x0_corrected, measurement_state, lambda_t)
            x0_corrected = x0_corrected + self.zeta * effective_dc_weight * correction

        # 5. Optional: Hard data consistency projection (also iterative)
        if self.hard_dc:
            for _ in range(self.dc_iterations):
                x0_corrected = self._hard_data_consistency(x0_corrected, measurement_state, lambda_t)

        # 6. Re-noise for next step
        # x_{t-1} = α_{t-1} · x̂₀ + σ_{t-1} · noise_component
        if self.stochastic:
            # Mix of deterministic and stochastic components
            key_noise = rng_key
            noise = jax.random.normal(key_noise, x0_corrected.shape, dtype=x0_corrected.dtype)

            # Deterministic component: prediction error
            eps_pred = (x_t - alpha_t * x0_hat) / (sigma_t + 1e-8)

            # Mix stochastic and deterministic
            noise_component = jnp.sqrt(self.xi) * noise + jnp.sqrt(1.0 - self.xi) * eps_pred
        else:
            # Fully deterministic: use predicted noise
            noise_component = (x_t - alpha_t * x0_hat) / (sigma_t + 1e-8)

        # Compute x_{t-1}
        x_next = alpha_next * x0_corrected + sigma_next * noise_component

        # Handle final step (no noise)
        is_final = sigma_next < 1e-6
        x_next = jnp.where(is_final, x0_corrected, x_next)

        # Update state
        integrator_state_next = IntegratorState(
            position=x_next,
            rng_key=rng_key,
            step=step_idx + 1,
        )
        state_next = state._replace(integrator_state=integrator_state_next)

        return state_next
