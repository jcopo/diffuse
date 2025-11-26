# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Ensemble Kalman Guidance (EnKG) for derivative-free diffusion inverse problems."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from einops import einsum
from jaxtyping import Array, PRNGKeyArray

from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.base_forward_model import MeasurementState


@dataclass
class EnKGDenoiser(CondDenoiser):
    """Ensemble Kalman Guidance (EnKG) conditional denoiser.

    Derivative-free method using ensemble Kalman updates. Does NOT require
    gradients of the forward model.

    Args:
        guidance_scale: Base learning rate (γ) for Kalman updates
        lr_min_ratio: Min LR ratio (r) for decay: γ(1-r)(N-i)/N + r
        denoising_steps: Euler ODE steps for x̂₀ estimation (1 = Tweedie)

    Reference: https://github.com/devzhk/enkg-pytorch
    """

    guidance_scale: float = 0.5
    lr_min_ratio: float = 0.0
    denoising_steps: int = 15

    def __post_init__(self):
        self.resample = False
        if self.guidance_scale <= 0:
            raise ValueError("guidance_scale must be strictly positive.")

    def _denoise_to_x0(self, x_t: Array, t_start: float) -> Array:
        eps = 1e-3
        n = self.denoising_steps

        def euler_step(x, i):
            t = t_start + i / n * (eps - t_start)
            t_next = t_start + (i + 1) / n * (eps - t_start)
            return x + self.predictor.velocity(x, t) * (t_next - t), None

        x_final, _ = jax.lax.scan(euler_step, x_t, jnp.arange(n))
        return x_final

    def _ensemble_kalman_update(
        self,
        particles: Array,
        measurements: Array,
        observation: Array,
        lr: float,
    ) -> Array:
        N = particles.shape[0]

        # deviations from ensemble mean
        x_diff = particles - jnp.mean(particles, axis=0, keepdims=True)
        y_diff = measurements - jnp.mean(measurements, axis=0, keepdims=True)

        # measurement error: ∇ (0.5 ||ŷ - y||²)
        y_err = measurements - observation

        # coef[i,j] = <y_err[i], y_diff[j]> / N
        coef = einsum(y_err, y_diff, "i ..., j ... -> i j") / N

        # dx[i] = sum_j coef[i,j] * x_diff[j]
        dx = einsum(coef, x_diff, "i j, j ... -> i ...")

        lr_norm = lr / (jnp.linalg.norm(coef) + 1e-8)
        return particles - lr_norm * dx

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        integrator_state_next = self.integrator(state.integrator_state, self.predictor)
        return state._replace(integrator_state=integrator_state_next)

    def generate(
        self,
        rng_key: PRNGKeyArray,
        measurement_state: MeasurementState,
        n_steps: int,
        n_particles: int,
        keep_history: bool = False,
    ):
        rng_key, rng_key_start = jax.random.split(rng_key)
        rndm_start = jax.random.normal(rng_key_start, (n_particles, *self.x0_shape))

        keys = jax.random.split(rng_key, n_particles)
        state = jax.vmap(self.init, in_axes=(0, 0, None))(rndm_start, keys, n_particles)

        def body_fun(state: CondDenoiserState, key: PRNGKeyArray):
            # kalman guidance → diffusion step
            state_guided = self._apply_kalman_guidance(state, measurement_state, n_steps)
            keys = jax.random.split(key, state_guided.integrator_state.position.shape[0])
            state_next = jax.vmap(self.step, in_axes=(0, 0, None))(keys, state_guided, measurement_state)
            return (
                state_next,
                state_next.integrator_state.position if keep_history else None,
            )

        keys = jax.random.split(rng_key, n_steps)
        return jax.lax.scan(body_fun, state, keys)

    def _apply_kalman_guidance(
        self,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
        n_steps: int,
    ) -> CondDenoiserState:
        particles = state.integrator_state.position
        step_idx = state.integrator_state.step[0]
        t_current = self.integrator.timer(step_idx)

        # denoise → measure → kalman update
        x0_estimates = jax.vmap(lambda x: self._denoise_to_x0(x, t_current))(particles)
        measurements = jax.vmap(lambda x: self.forward_model.apply(x, measurement_state))(x0_estimates)

        # decaying LR schedule
        r = self.lr_min_ratio
        lr = self.guidance_scale * (1 - r) * (n_steps - step_idx) / n_steps + r
        particles_updated = self._ensemble_kalman_update(particles, measurements, measurement_state.y, lr)

        integrator_state_updated = state.integrator_state._replace(position=particles_updated)
        return CondDenoiserState(integrator_state_updated, state.log_weights)


__all__ = ["EnKGDenoiser"]
