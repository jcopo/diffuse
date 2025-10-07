from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from diffuse.integrator.base import IntegratorState, Integrator
from diffuse.diffusion.sde import DiffusionModel
from diffuse.predictor import Predictor

__all__ = ["EulerMaruyamaIntegrator", "DDCMIntegrator"]


@dataclass
class EulerMaruyamaIntegrator(Integrator):
    """Euler-Maruyama stochastic integrator for Stochastic Differential Equations (SDEs).

    Implements the Euler-Maruyama method for numerical integration of SDEs of the form:
    dX(t) = μ(X,t)dt + σ(X,t)dW(t)

    where:
    - μ(X,t) is the drift term: β(t) * (0.5 * X + score(X,t))
    - σ(X,t) is the diffusion term: sqrt(β(t))
    - dW(t) is the Wiener process increment
    - β(t) is the noise schedule

    The method advances the solution using the discrete approximation:
    X(t + dt) = X(t) + μ(X,t)dt + σ(X,t)√dt * N(0,1)

    This is the simplest stochastic integration scheme with strong order 0.5
    convergence for general SDEs.
    """

    model: DiffusionModel

    def __call__(self, integrator_state: IntegratorState, predictor: Predictor) -> IntegratorState:
        """Perform one Euler-Maruyama integration step.

        Args:
            integrator_state: Current state containing:
                - position: Current position X(t)
                - rng_key: JAX random number generator key
                - step: Current integration step
            score: Score function that approximates ∇ₓ log p(x|t)

        Returns:
            Updated IntegratorState containing:
                - New position X(t + dt)
                - Updated RNG key
                - Incremented step count

        Notes:
            The integration step implements:
            dx = drift*dt + diffusion*√dt*ε
            where:
            - drift = β(t) * (0.5 * position + score(position, t))
            - diffusion = √β(t)
            - ε ~ N(0,1)
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


@dataclass
class DDCMIntegrator(Integrator):
    """Discrete Diffusion with Codebook Matching (DDCM) integrator.

    Implements a variant of Euler-Maruyama where stochastic noise is sampled from
    a discrete codebook rather than a continuous Gaussian distribution. This enables
    learning structured noise patterns that may better capture the data distribution.
    https://arxiv.org/pdf/2502.01189

    The integrator solves SDEs of the form:
    dX(t) = μ(X,t)dt + σ(X,t)dW_codebook(t)

    where:
    - μ(X,t) is the drift term: g(t)² * (0.5 * X + score(X,t))
    - σ(X,t) is the diffusion term: g(t)
    - dW_codebook(t) is sampled uniformly from a learned codebook
    - g(t) = sqrt(β(t)) where β(t) is the noise schedule

    Discretization:
    X(t + dt) = X(t) + μ(X,t)dt + σ(X,t)√dt * codebook[i], i ~ Uniform(0, |codebook|)

    Attributes:
        model: Diffusion model providing SDE coefficients
        codebook: Array of shape (size_codebook, *x0_shape) containing learned noise vectors

    Initialization:
        codebook = jax.random.normal(rng_key, (size_codebook, *x0_shape))
        integrator = DDCMIntegrator(model=model, timer=timer, codebook=codebook)
    """

    model: DiffusionModel
    codebook: Array  # Shape: (size_codebook, *x0_shape)

    def __call__(self, integrator_state: IntegratorState, predictor: Predictor) -> IntegratorState:
        """Perform one DDCM integration step.

        Args:
            integrator_state: Current state containing:
                - position: Current position X(t) with shape (*x0_shape)
                - rng_key: JAX random number generator key
                - step: Current integration step index
            predictor: Predictor providing the score function ∇ₓ log p(x|t)

        Returns:
            Updated IntegratorState containing:
                - New position X(t + dt)
                - Updated RNG key (split for next iteration)
                - Incremented step count

        Notes:
            The integration step implements:
            dx = drift*dt + diffusion*√dt*codebook[i]
            where:
            - drift = g(t)² * (0.5 * position + score(position, t))
            - diffusion = g(t) = sqrt(β(t))
            - i ~ Uniform(0, codebook_size) sampled independently per step
        """

        position, rng_key, step = integrator_state
        t, t_next = self.timer(step), self.timer(step + 1)
        dt = t - t_next
        f_t, g_t = self.model.sde_coefficients(t)
        # For reverse-time: drift = f(t)*x - g(t)^2*score, but rearranged as: g(t)^2 * (0.5*x + score)
        # Since f(t) = -0.5*beta(t) and g(t) = sqrt(beta(t)), we have beta(t) = g(t)^2
        drift = g_t * g_t * (0.5 * position + predictor.score(position, t))
        diffusion = g_t

        rng_key, rng_noise = jax.random.split(rng_key)
        # Sample a single random index from the codebook
        rdx_index = jax.random.randint(rng_noise, shape=(), minval=0, maxval=self.codebook.shape[0])
        # Index the codebook to get noise with the same shape as position
        noise = self.codebook[rdx_index] * jnp.sqrt(dt)

        dx = drift * dt + diffusion * noise
        _, rng_key_next = jax.random.split(rng_key)
        return IntegratorState(position + dx, rng_key_next, step + 1)
