# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import ForwardModel, MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.diffusion.sde import DiffusionModel, SDEState

ArrayFn = Callable[[Array, Array], Array]


@dataclass
class _RectifiedPredictor:
    """Predictor wrapper that always feeds the integrator a rectified x₀ estimate."""

    model: DiffusionModel
    rectified_x0: ArrayFn

    def x0(self, x: Array, t: Array) -> Array:
        return self.rectified_x0(x, t)

    def score(self, x: Array, t: Array) -> Array:
        x0_hat = self.x0(x, t)
        alpha = self.model.signal_level(t)
        sigma = self.model.noise_level(t)
        return (alpha * x0_hat - x) / (sigma * sigma + 1e-8)

    def noise(self, x: Array, t: Array) -> Array:
        x0_hat = self.x0(x, t)
        alpha = self.model.signal_level(t)
        sigma = self.model.noise_level(t)
        return (x - alpha * x0_hat) / (sigma + 1e-8)

    def velocity(self, x: Array, t: Array) -> Array:
        score = self.score(x, t)
        f_t, g_t = self.model.sde_coefficients(t)
        return f_t * x - 0.5 * (g_t * g_t) * score


@dataclass
class DDNMDenoiser(CondDenoiser):
    """Data-Consistent Diffusion Noise Matching (DDNM) conditional denoiser.

    Implements the range/nullspace decomposition from "Plug-and-Play Denoising
    Diffusion Models: DDNM" by modifying the Tweedie estimate at every step:

        x̂₀|t = A^† y + (I - A^† A) x₀|t

    The rectified x̂₀|t is fed back to the integrator via a predictor wrapper so
    that any integrator (Euler, Heun, DPM++, DDIM, ...) automatically receives a
    data-consistent clean estimate.
    """

    cg_tol: float = 1e-5
    cg_maxiter: int = 32
    pinv_regularization: float = 0.0

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Perform one DDNM-guided diffusion step."""
        del rng_key  # Deterministic guidance – randomness handled by integrator if needed.

        range_component = self._data_consistent_component(measurement_state)

        def rectified_x0_fn(x_t: Array, t: Array) -> Array:
            tweedie = self.model.tweedie(SDEState(x_t, t), self.predictor.score).position
            range_projection = self._project_to_range(tweedie, measurement_state)
            nullspace_part = tweedie - range_projection
            return range_component + nullspace_part

        predictor = _RectifiedPredictor(self.model, rectified_x0_fn)
        integrator_state_next = self.integrator(state.integrator_state, predictor)
        return state._replace(integrator_state=integrator_state_next)

    # ------------------------------------------------------------------ #
    # Range/nullspace helpers
    # ------------------------------------------------------------------ #
    def _data_consistent_component(self, measurement_state: MeasurementState) -> Array:
        """Compute A^† y via a CG solve on the normal equations."""
        measurement = measurement_state.y
        fallback = self.forward_model.adjoint(measurement, measurement_state)
        return self._solve_normal_equation(self.forward_model, measurement_state, measurement, fallback)

    def _project_to_range(self, img: Array, measurement_state: MeasurementState) -> Array:
        """Project an image onto range(Aᵀ) using the same pseudoinverse solver."""
        measurement = self.forward_model.apply(img, measurement_state)
        fallback = self.forward_model.adjoint(measurement, measurement_state)
        return self._solve_normal_equation(self.forward_model, measurement_state, measurement, fallback)

    def _solve_normal_equation(
        self,
        forward_model: ForwardModel,
        measurement_state: MeasurementState,
        measurement: Array,
        fallback: Array,
    ) -> Array:
        """Solve (AᵀA + λI) x = Aᵀ measurement with CG, falling back to the adjoint."""
        rhs = forward_model.adjoint(measurement, measurement_state)

        def matvec(v: Array) -> Array:
            applied = forward_model.apply(v, measurement_state)
            adjoint_applied = forward_model.adjoint(applied, measurement_state)
            if self.pinv_regularization > 0:
                adjoint_applied = adjoint_applied + self.pinv_regularization * v
            return adjoint_applied

        sol, info = cg(matvec, rhs, x0=fallback, tol=self.cg_tol, maxiter=self.cg_maxiter)

        sol = jnp.where(jnp.isfinite(sol), sol, fallback)
        sol = jax.lax.cond(info == 0, lambda _: sol, lambda _: fallback, operand=None)
        return sol


__all__ = ["DDNMDenoiser"]
