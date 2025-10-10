# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Network adapter providing all prediction types (score, noise, velocity, x0) from any trained network.

This module implements general conversions between different diffusion model parameterizations:

1. **Score parameterization**: Predicts the score function ∇log p_t(x)
2. **Noise parameterization**: Predicts the noise ε added during the forward process
3. **Velocity parameterization**: Predicts the velocity field u_t(x) for probability flow ODEs
4. **x0 parameterization**: Predicts the denoised data x̂_0

The conversions use the general SDE formulation dx_t = f(t) x_t dt + g(t) dW_t
where f(t) and g(t) are model-specific coefficients:

- **SDE**: f(t) = -β(t)/2, g(t) = √β(t)
- **Flow**: f(t) = -1/(1-t), g(t) = √(2t/(1-t))
- **EDM**: f(t) = 0, g(t) = 1

Velocity conversion uses the probability flow ODE:
u_t(x) = f(t) x - g(t)²/2 ∇log p_t(x)
"""

from typing import Callable, Dict
from dataclasses import dataclass

from jaxtyping import Array

from diffuse.diffusion.sde import DiffusionModel


# Conversion functions from score to other types
def score_to_noise(score_fn: Callable, model: DiffusionModel) -> Callable:
    def noise_fn(x: Array, t: Array) -> Array:
        sigma_t = model.noise_level(t)
        score = score_fn(x, t)
        return -sigma_t * score

    return noise_fn


def score_to_velocity(score_fn: Callable, model: DiffusionModel) -> Callable:
    """Convert score function to velocity field using general SDE coefficients.

    Uses the probability flow ODE formula:
    u_t(x) = f(t) x - g(t)²/2 ∇log p_t(x)

    This replaces the previous rectified flow-specific implementation.
    """

    def velocity_fn(x: Array, t: Array) -> Array:
        score = score_fn(x, t)
        f_t, g_t = model.sde_coefficients(t)

        # General velocity formula from probability flow ODE
        return f_t * x - (g_t * g_t / 2) * score

    return velocity_fn


def score_to_x0(score_fn: Callable, model: DiffusionModel) -> Callable:
    def x0_fn(x: Array, t: Array) -> Array:
        alpha_t = model.signal_level(t)
        sigma_t = model.noise_level(t)
        score = score_fn(x, t)
        return (x + sigma_t * sigma_t * score) / (alpha_t + 1e-8)

    return x0_fn


# Conversion functions from noise to other types
def noise_to_score(noise_fn: Callable, model: DiffusionModel) -> Callable:
    def score_fn(x: Array, t: Array) -> Array:
        sigma_t = model.noise_level(t)
        noise = noise_fn(x, t)
        return -noise / (sigma_t + 1e-8)

    return score_fn


def noise_to_velocity(noise_fn: Callable, model: DiffusionModel) -> Callable:
    def velocity_fn(x: Array, t: Array) -> Array:
        # Convert noise -> score -> velocity
        score_fn = noise_to_score(noise_fn, model)
        return score_to_velocity(score_fn, model)(x, t)

    return velocity_fn


def noise_to_x0(noise_fn: Callable, model: DiffusionModel) -> Callable:
    def x0_fn(x: Array, t: Array) -> Array:
        alpha_t = model.signal_level(t)
        sigma_t = model.noise_level(t)
        noise = noise_fn(x, t)
        return (x - sigma_t * noise) / (alpha_t + 1e-8)

    return x0_fn


# Conversion functions from velocity to other types
def velocity_to_score(velocity_fn: Callable, model: DiffusionModel) -> Callable:
    """Convert velocity field to score function using general SDE coefficients.

    Inverts the probability flow ODE formula:
    ∇log p_t(x) = 2(f(t) x - u_t(x)) / g(t)²

    This replaces the previous rectified flow-specific implementation.
    """

    def score_fn(x: Array, t: Array) -> Array:
        v = velocity_fn(x, t)
        f_t, g_t = model.sde_coefficients(t)

        # Invert the velocity formula: score = 2(f(t) x - u_t(x)) / g(t)²
        return 2 * (f_t * x - v) / (g_t * g_t + 1e-8)

    return score_fn


def velocity_to_noise(velocity_fn: Callable, model: DiffusionModel) -> Callable:
    def noise_fn(x: Array, t: Array) -> Array:
        # Convert velocity -> score -> noise
        score_fn = velocity_to_score(velocity_fn, model)
        return score_to_noise(score_fn, model)(x, t)

    return noise_fn


def velocity_to_x0(velocity_fn: Callable, model: DiffusionModel) -> Callable:
    def x0_fn(x: Array, t: Array) -> Array:
        # Convert velocity -> score -> x0
        score_fn = velocity_to_score(velocity_fn, model)
        return score_to_x0(score_fn, model)(x, t)

    return x0_fn


# Conversion functions from x0 to other types
def x0_to_score(x0_fn: Callable, model: DiffusionModel) -> Callable:
    def score_fn(x: Array, t: Array) -> Array:
        x0_pred = x0_fn(x, t)
        alpha_t = model.signal_level(t)
        sigma_t = model.noise_level(t)
        return (alpha_t * x0_pred - x) / (sigma_t * sigma_t + 1e-8)

    return score_fn


def x0_to_noise(x0_fn: Callable, model: DiffusionModel) -> Callable:
    def noise_fn(x: Array, t: Array) -> Array:
        x0_pred = x0_fn(x, t)
        alpha_t = model.signal_level(t)
        sigma_t = model.noise_level(t)
        return (x - alpha_t * x0_pred) / (sigma_t + 1e-8)

    return noise_fn


def x0_to_velocity(x0_fn: Callable, model: DiffusionModel) -> Callable:
    def velocity_fn(x: Array, t: Array) -> Array:
        # Convert x0 -> score -> velocity
        score_fn = x0_to_score(x0_fn, model)
        return score_to_velocity(score_fn, model)(x, t)

    return velocity_fn


# Identity functions
def identity(fn: Callable, model: DiffusionModel) -> Callable:
    return fn


# Registry of all conversion functions
CONVERSIONS: Dict[str, Dict[str, Callable]] = {
    "score": {
        "score": identity,
        "noise": score_to_noise,
        "velocity": score_to_velocity,
        "x0": score_to_x0,
    },
    "noise": {
        "score": noise_to_score,
        "noise": identity,
        "velocity": noise_to_velocity,
        "x0": noise_to_x0,
    },
    "velocity": {
        "score": velocity_to_score,
        "noise": velocity_to_noise,
        "velocity": identity,
        "x0": velocity_to_x0,
    },
    "x0": {
        "score": x0_to_score,
        "noise": x0_to_noise,
        "velocity": x0_to_velocity,
        "x0": identity,
    },
}


@dataclass
class Predictor:
    """Network adapter providing all prediction types (score, noise, velocity, x0).

    Automatically converts between different diffusion model parameterizations:

    - **Score**: Predicts :math:`\\nabla \\log p_t(x)`
    - **Noise**: Predicts :math:`\\varepsilon` added during forward process
    - **Velocity**: Predicts velocity field :math:`u_t(x)` for probability flow ODEs
    - **x0**: Predicts denoised data :math:`\\hat{x}_0`

    Args:
        model: Diffusion model defining the diffusion process
        network: The trained neural network (e.g., UNet)
        prediction_type: Type of prediction the network outputs ("score", "noise", "velocity", or "x0")
    """

    model: DiffusionModel
    network: Callable
    prediction_type: str

    def __post_init__(self):
        if self.prediction_type not in CONVERSIONS:
            available = ", ".join(CONVERSIONS.keys())
            raise ValueError(f"Unknown prediction type '{self.prediction_type}'. Available: {available}")

        # Cache converted functions
        self._score_fn = CONVERSIONS[self.prediction_type]["score"](self.network, self.model)
        self._noise_fn = CONVERSIONS[self.prediction_type]["noise"](self.network, self.model)
        self._velocity_fn = CONVERSIONS[self.prediction_type]["velocity"](self.network, self.model)
        self._x0_fn = CONVERSIONS[self.prediction_type]["x0"](self.network, self.model)

    def score(self, x: Array, t: Array) -> Array:
        r"""Get score function :math:`\nabla \log p_t(x)`.

        Args:
            x: Current state
            t: Current time

        Returns:
            Score prediction
        """
        return self._score_fn(x, t)

    def noise(self, x: Array, t: Array) -> Array:
        r"""Get noise prediction :math:`\varepsilon_\theta(x,t)`.

        Args:
            x: Current state
            t: Current time

        Returns:
            Noise prediction
        """
        return self._noise_fn(x, t)

    def velocity(self, x: Array, t: Array) -> Array:
        r"""Get velocity field :math:`u_t(x)`.

        Args:
            x: Current state
            t: Current time

        Returns:
            Velocity prediction
        """
        return self._velocity_fn(x, t)

    def x0(self, x: Array, t: Array) -> Array:
        r"""Get denoised prediction :math:`\hat{x}_0(x,t)`.

        Args:
            x: Current state
            t: Current time

        Returns:
            Denoised data prediction
        """
        return self._x0_fn(x, t)
