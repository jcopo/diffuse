# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Integrators that accept externally provided noise sequences.

These helpers mirror the stochastic integrators defined in
``diffuse.integrator.stochastic`` but allow callers to supply the exact noise
increments used at each step. This is required for differentiable sampling
routines (e.g., stochastic MALA) where the full latent noise sequence forms the
state of the Markov chain.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from diffuse.diffusion.sde import DiffusionModel
from diffuse.predictor import Predictor
from diffuse.timer.base import Timer

__all__ = ["EulerMaruyamaExternalNoiseIntegrator"]


@dataclass
class EulerMaruyamaExternalNoiseIntegrator:
    r"""Euler-Maruyama integrator that consumes external noise increments.

    Args:
        model: Diffusion model defining :math:`f(t), g(t)` coefficients
        timer: Timer controlling the time discretization
    """

    model: DiffusionModel
    timer: Timer

    def step(self, position, step: int, predictor: Predictor, noise: jnp.ndarray) -> jnp.ndarray:
        r"""Compute a single Euler-Maruyama update using a provided noise tensor."""

        t = self.timer(step)
        t_next = self.timer(step + 1)
        dt = t - t_next
        _, g_t = self.model.sde_coefficients(t)
        drift = g_t * g_t * (0.5 * position + predictor.score(position, t))
        diffusion = g_t
        return position + drift * dt + diffusion * noise * jnp.sqrt(dt)

    def integrate(self, latent_sequence, predictor: Predictor) -> jnp.ndarray:
        r"""Integrate a full trajectory using the supplied latent sequence.

        The first element of ``latent_sequence`` is treated as the initial noise
        :math:`x_T`, while the remaining elements provide the diffusion
        increments for each timestep.
        """

        initial_state = latent_sequence[0]
        noises = latent_sequence[1:]

        def body_fn(carry, inputs):
            position, step = carry
            next_position = self.step(position, step, predictor, inputs)
            return (next_position, step + 1), next_position

        (final_position, _), history = jax.lax.scan(body_fn, (initial_state, 0), noises)
        return jnp.where(noises.shape[0] > 0, history[-1], initial_state)
