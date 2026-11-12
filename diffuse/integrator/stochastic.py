# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array

from diffuse.integrator.base import IntegratorState, Integrator
from diffuse.diffusion.sde import DiffusionModel
from diffuse.predictor import Predictor

__all__ = ["EulerMaruyamaIntegrator", "euler_maruyama_transition"]


def euler_maruyama_transition(
    model: DiffusionModel, position: Array, t: Array, t_next: Array, scale: Array, predictor: Predictor
) -> tuple[Array, Array]:
    r"""One reverse-time Euler-Maruyama step as a Gaussian, :math:`x_{t'} \mid x_t \sim \mathcal{N}(\mu, s^2 I)`, for the
    forward SDE :math:`dX = f(t) X dt + g(t) dW` and the reverse diffusion :math:`a\,g(t)`:

    .. math::
        \mu = x_t + \left(\tfrac{1 + a^2}{2} g(t)^2 \nabla_x \log p_t(x_t) - f(t) x_t\right)(t - t'), \qquad s = a\,g(t)\sqrt{t - t'}

    Every :math:`a` keeps the probability-flow marginals: :math:`a = 1` is the reverse SDE, :math:`a = 0` an Euler step
    of the probability-flow ODE.

    Args:
        model: Diffusion model defining the diffusion process
        position: The state :math:`x_t`
        t: Current time
        t_next: Next time :math:`t' < t`
        scale: The noise scale :math:`a`
        predictor: Predictor providing the score :math:`\nabla_x \log p_t`

    Returns:
        The mean :math:`\mu` and the standard deviation :math:`s` of the next state
    """
    dt = t - t_next
    f_t, g_t = model.sde_coefficients(t)
    # reverse drift u - (a g)^2 / 2 score, with u = f x - g^2 / 2 score the probability-flow velocity
    mean = position + (0.5 * (1.0 + scale**2) * g_t * g_t * predictor.score(position, t) - f_t * position) * dt
    return mean, scale * g_t * jnp.sqrt(dt)


@dataclass
class EulerMaruyamaIntegrator(Integrator):
    r"""Euler-Maruyama stochastic integrator for the reverse-time SDE of any diffusion model.

    Each step draws from :func:`euler_maruyama_transition`'s Gaussian. With the forward SDE
    :math:`dX(t) = f(t) X(t) dt + g(t) dW(t)` and ``scale`` :math:`a`, it advances by

    .. math::
        X(t - dt) = X(t) + \left(\tfrac{1 + a^2}{2} g(t)^2 \nabla_x \log p_t(X) - f(t) X\right) dt + a\,g(t) \sqrt{dt} \cdot \mathcal{N}(0, I)

    This is the simplest stochastic integration scheme with strong order 0.5
    convergence for general SDEs.

    Args:
        model: Diffusion model defining the diffusion process
        timer: Timer object managing the discretization of the time interval
        scale: Noise scale :math:`a`, one value or one per step (default: 1.0, the reverse SDE; 0: an Euler step of the
            probability-flow ODE). Flow-GRPO's :math:`a\sqrt{t / (1 - t)}` on a Flow model is :math:`a / \sqrt{2}`.
    """

    model: DiffusionModel
    scale: float | Array = 1.0

    def __call__(self, integrator_state: IntegratorState, predictor: Predictor) -> IntegratorState:
        r"""Perform one Euler-Maruyama integration step.

        Args:
            integrator_state: Current state containing (position, rng_key, step)
            predictor: Predictor providing score function :math:`\nabla_x \log p(x|t)`

        Returns:
            Updated IntegratorState with the next position
        """
        position, rng_key, step = integrator_state
        scale = jnp.broadcast_to(jnp.asarray(self.scale), (self.timer.n_steps,))[step]
        mean, std = euler_maruyama_transition(self.model, position, self.timer(step), self.timer(step + 1), scale, predictor)
        _, rng_key_next = jax.random.split(rng_key)
        return IntegratorState(mean + std * jax.random.normal(rng_key, position.shape), rng_key_next, step + 1)
