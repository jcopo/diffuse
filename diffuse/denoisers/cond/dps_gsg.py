# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Zero-order DPS using Gaussian Smoothed Gradient estimation."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.base_forward_model import MeasurementState


@dataclass
class DPSGSGDenoiser(CondDenoiser):
    """Zero-order DPS using Gaussian Smoothed Gradient (GSG) estimation.

    Derivative-free DPS for black-box or non-differentiable forward models.
    Estimates gradients via finite differences with Gaussian perturbations.

    Args:
        num_queries: Number of perturbation samples for gradient estimation
        mu: Perturbation scale (smoothing factor)
        zeta: Gradient step size
        epsilon: Numerical stability constant
        central_diff: Use central differences (True) or forward differences (False)
    """

    num_queries: int = 64
    mu: float = 0.01
    zeta: float = 1e-2
    epsilon: float = 1e-3
    central_diff: bool = True

    def __post_init__(self):
        if self.num_queries <= 0:
            raise ValueError("num_queries must be strictly positive.")
        if self.mu <= 0:
            raise ValueError("mu must be strictly positive.")
        if self.zeta <= 0:
            raise ValueError("zeta must be strictly positive.")

    def _estimate_gradient_central(
        self,
        rng_key: PRNGKeyArray,
        x0_hat: Array,
        measurement_state: MeasurementState,
    ) -> tuple[Array, Array]:
        y_meas = measurement_state.y
        perturbations = jax.random.normal(rng_key, (self.num_queries, *x0_hat.shape))

        def compute_loss(x: Array) -> Array:
            residual = y_meas - self.forward_model.apply(x, measurement_state)
            return jnp.sum(residual**2)

        base_loss = compute_loss(x0_hat)

        def single_query_gradient(u: Array) -> Array:
            loss_plus = compute_loss(x0_hat + self.mu * u)
            loss_minus = compute_loss(x0_hat - self.mu * u)
            return u * (loss_plus - loss_minus) / (2.0 * self.mu)

        gradients = jax.vmap(single_query_gradient)(perturbations)
        return jnp.mean(gradients, axis=0), base_loss

    def _estimate_gradient_forward(
        self,
        rng_key: PRNGKeyArray,
        x0_hat: Array,
        measurement_state: MeasurementState,
    ) -> tuple[Array, Array]:
        y_meas = measurement_state.y
        perturbations = jax.random.normal(rng_key, (self.num_queries, *x0_hat.shape))

        def compute_loss(x: Array) -> Array:
            residual = y_meas - self.forward_model.apply(x, measurement_state)
            return jnp.sum(residual**2)

        base_loss = compute_loss(x0_hat)

        def single_query_gradient(u: Array) -> Array:
            loss_perturbed = compute_loss(x0_hat + self.mu * u)
            return u * (loss_perturbed - base_loss) / self.mu

        gradients = jax.vmap(single_query_gradient)(perturbations)
        return jnp.mean(gradients, axis=0), base_loss

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        key_gsg, _ = jax.random.split(rng_key)

        position_current = state.integrator_state.position
        t_current = self.integrator.timer(state.integrator_state.step)

        # estimate gradient via GSG
        x0_hat = self.model.tweedie(SDEState(position_current, t_current), self.predictor.score).position

        if self.central_diff:
            gradient, loss_val = self._estimate_gradient_central(key_gsg, x0_hat, measurement_state)
        else:
            gradient, loss_val = self._estimate_gradient_forward(key_gsg, x0_hat, measurement_state)

        # scale gradient from x0 to xt space
        alpha_t = self.model.signal_level(t_current)
        alpha_t_clipped = jnp.maximum(alpha_t, 0.1)
        gradient_xt = gradient / alpha_t_clipped

        zeta = self.zeta / (jnp.sqrt(loss_val) + self.epsilon)

        # apply correction
        integrator_state_uncond = self.integrator(state.integrator_state, self.predictor)
        position_corrected = integrator_state_uncond.position - zeta * gradient_xt

        integrator_state_next = integrator_state_uncond._replace(position=position_corrected)
        return state._replace(integrator_state=integrator_state_next)


__all__ = ["DPSGSGDenoiser"]
