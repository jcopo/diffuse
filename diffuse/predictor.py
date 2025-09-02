"""Network adapter providing all prediction types (score, noise, velocity, x0) from any trained network."""

from typing import Callable, Dict
from dataclasses import dataclass

import jax.numpy as jnp
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
    def velocity_fn(x: Array, t: Array) -> Array:
        t_safe = jnp.clip(t, 1e-6, 1 - 1e-6)
        score = score_fn(x, t_safe)

        # For rectified flow: u_t(x) = -x/(1-t) - (t + t²/(1-t)) * ∇log p_t(x)
        denominator = t_safe + (t_safe * t_safe) / (1 - t_safe)
        return -x / (1 - t_safe) - denominator * score

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
    def score_fn(x: Array, t: Array) -> Array:
        t_safe = jnp.clip(t, 1e-6, 1 - 1e-6)
        v = velocity_fn(x, t_safe)

        # For rectified flow: ∇log p_t(x) = (-x/(1-t) - u_t(x)) / (t + t²/(1-t))
        numerator = -x / (1 - t_safe) - v
        denominator = t_safe + (t_safe * t_safe) / (1 - t_safe)
        return numerator / (denominator + 1e-8)

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
    """Learned network that provides all prediction types (score, noise, velocity, x0)."""

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
        """Get score function ∇log p_t(x)."""
        return self._score_fn(x, t)

    def noise(self, x: Array, t: Array) -> Array:
        """Get noise prediction function ε_θ(x,t)."""
        return self._noise_fn(x, t)

    def velocity(self, x: Array, t: Array) -> Array:
        """Get velocity field function u_t(x)."""
        return self._velocity_fn(x, t)

    def x0(self, x: Array, t: Array) -> Array:
        """Get denoised prediction function x̂_0(x,t)."""
        return self._x0_fn(x, t)
