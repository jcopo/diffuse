# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from diffuse.integrator.base import IntegratorState, Integrator
from diffuse.diffusion.sde import DiffusionModel
from diffuse.predictor import Predictor

__all__ = ["EulerMaruyamaIntegrator"]


@dataclass
class EulerMaruyamaIntegrator(Integrator):
    r"""Euler-Maruyama stochastic integrator for Stochastic Differential Equations (SDEs).

    Implements the Euler-Maruyama method for numerical integration of SDEs of the form:

    .. math::
        dX(t) = \mu(X,t)dt + \sigma(X,t)dW(t)

    where:

    - :math:`\mu(X,t)` is the drift term: :math:`\beta(t) \cdot (0.5 X + \nabla_x \log p(x|t))`
    - :math:`\sigma(X,t)` is the diffusion term: :math:`\sqrt{\beta(t)}`
    - :math:`dW(t)` is the Wiener process increment
    - :math:`\beta(t)` is the noise schedule

    The method advances the solution using the discrete approximation:

    .. math::
        X(t + dt) = X(t) + \mu(X,t)dt + \sigma(X,t)\sqrt{dt} \cdot \mathcal{N}(0,1)

    This is the simplest stochastic integration scheme with strong order 0.5
    convergence for general SDEs.

    Args:
        model: Diffusion model defining the diffusion process
        timer: Timer object managing the discretization of the time interval
    """

    model: DiffusionModel

    def __call__(self, integrator_state: IntegratorState, predictor: Predictor) -> IntegratorState:
        r"""Perform one Euler-Maruyama integration step.

        Args:
            integrator_state: Current state containing (position, rng_key, step)
            predictor: Predictor providing score function :math:`\nabla_x \log p(x|t)`

        Returns:
            Updated IntegratorState with the next position

        Notes:
            The integration step implements:

            .. math::
                dx = \text{drift} \cdot dt + \text{diffusion} \cdot \sqrt{dt} \cdot \varepsilon

            where:

            - :math:`\text{drift} = \beta(t) \cdot (0.5 \cdot x + \nabla_x \log p(x|t))`
            - :math:`\text{diffusion} = \sqrt{\beta(t)}`
            - :math:`\varepsilon \sim \mathcal{N}(0,1)`
        """
        position, rng_key, step = integrator_state
        t, t_next = self.timer(step), self.timer(step + 1)
        dt = t - t_next
        f_t, g_t = self.model.sde_coefficients(t)
        # For reverse-time: drift = f(t)*x - g(t)^2*score, but rearranged as: g(t)^2 * (0.5*x + score)
        # Since f(t) = -0.5*beta(t) and g(t) = sqrt(beta(t)), we have beta(t) = g(t)^2
        drift = g_t * g_t * (0.5 * position + predictor.score(position, t))
        diffusion = g_t
        noise = jax.random.normal(rng_key, position.shape) * jnp.sqrt(dt)

        dx = drift * dt + diffusion * noise
        _, rng_key_next = jax.random.split(rng_key)
        return IntegratorState(position + dx, rng_key_next, step + 1)
