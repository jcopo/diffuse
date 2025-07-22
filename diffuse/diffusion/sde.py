from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, NamedTuple, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class SDEState(NamedTuple):
    position: Array
    t: float


class Schedule(ABC):
    @abstractmethod
    def __call__(self, t: float) -> float:
        pass

    @abstractmethod
    def integrate(self, t: float, s: float) -> float:
        pass


@dataclass
class LinearSchedule:
    """
    A class representing a linear schedule.

    Attributes:
        b_min (float): The minimum value.
        b_max (float): The maximum value.
        t0 (float): The starting time.
        T (float): The ending time.
    """

    b_min: float
    b_max: float
    t0: float
    T: float

    def __call__(self, t):
        """
        Calculates the value of the linear schedule at a given time.

        Args:
            t (float): The time at which to evaluate the schedule.

        Returns:
            float: The value of the linear schedule at time t.
        """
        b_min, b_max, t0, T = self.b_min, self.b_max, self.t0, self.T
        return (b_max - b_min) / (T - t0) * t + (b_min * T - b_max * t0) / (T - t0)

    def integrate(self, t, s):
        b_min, b_max, t0, T = self.b_min, self.b_max, self.t0, self.T
        slope = (b_max - b_min) / (T - t0)
        intercept = (b_min * T - b_max * t0) / (T - t0)
        return 0.5 * (t - s) * (slope * (t + s) + 2 * intercept)


@dataclass
class CosineSchedule(Schedule):
    """
    A class representing a cosine schedule as described in
    'Improved Denoising Diffusion Probabilistic Models'

    Attributes:
        b_min (float): The minimum beta value
        b_max (float): The maximum beta value
        t0 (float): The starting time
        T (float): The ending time
        s (float): Offset parameter (default: 0.008)
    """

    b_min: float
    b_max: float
    t0: float
    T: float
    s: float = 0.008

    def __call__(self, t):
        """
        Calculates the value of the cosine schedule at a given time.
        """
        t_normalized = (t - self.t0) / (self.T - self.t0)

        beta_t = jnp.pi * jnp.tan(0.5 * jnp.pi * (t_normalized + self.s) / (1 + self.s)) / (self.T * (1 + self.s))
        beta_t = jnp.clip(beta_t, self.b_min, self.b_max)

        return beta_t

    def integrate(self, t, s):
        time_scale = self.T - self.t0
        offset_scale = 1 + self.s

        t_norm = (t - self.t0) / time_scale
        s_norm = (s - self.t0) / time_scale

        f0 = jnp.cos(self.s / offset_scale * jnp.pi * 0.5) ** 2
        ft = jnp.cos((t_norm + self.s) / offset_scale * jnp.pi * 0.5) ** 2
        fs = jnp.cos((s_norm + self.s) / offset_scale * jnp.pi * 0.5) ** 2

        alpha_t = jnp.clip(ft / f0, 0.001, 0.9999)
        alpha_s = jnp.clip(fs / f0, 0.001, 0.9999)

        return jnp.log(alpha_s / alpha_t)


class DiffusionModel(ABC):
    @abstractmethod
    def noise_level(self, t: float) -> float:
        pass

    def snr(self, t: float) -> float:
        """
        Compute Signal-to-Noise Ratio (SNR) at timestep t.

        SNR(t) = α²(t) / (1 - α²(t)) = α²(t) / noise_level(t)
        """
        noise_level = self.noise_level(t)
        alpha_squared = 1 - noise_level
        return alpha_squared / (noise_level + 1e-8)

    def score(self, state: SDEState, state_0: SDEState) -> Array:
        """
        Closed-form expression for the score function ∇ₓ log p(xₜ | xₜ₀) of the Gaussian transition kernel
        """
        x, t = state.position, state.t
        x0, t0 = state_0.position, state_0.t
        noise_level = self.noise_level(t)
        alpha_t = 1 - noise_level

        return -(x - jnp.sqrt(alpha_t) * x0) / noise_level

    def tweedie(self, state: SDEState, score_fn: Callable) -> SDEState:
        """
        Tweedie's formula to compute E[x_t | x_0]
        """
        x, t = state.position, state.t
        noise_level = self.noise_level(t)
        alpha_t = 1 - noise_level
        return SDEState((x + noise_level * score_fn(x, t)) / jnp.sqrt(alpha_t), 0.0)

    def path(
        self, key: PRNGKeyArray, state: SDEState, ts: Array, return_noise: bool = False
    ) -> Union[SDEState, tuple[SDEState, Array]]:
        """
        Samples x_t | x_0 ~ N(sqrt(alpha_t) * x_0, (1 - alpha_t) * I)
        """
        x = state.position
        noise_level = self.noise_level(ts)
        alpha_t = 1 - noise_level

        noise = jax.random.normal(key, x.shape)
        res = jnp.sqrt(alpha_t) * x + jnp.sqrt(noise_level) * noise
        return (SDEState(res, ts), noise) if return_noise else SDEState(res, ts)

    def score_to_noise(self, score_fn: Callable) -> Callable:
        def noise_fn(x: Array, t: Array) -> Array:
            noise_level = self.noise_level(t)
            score = score_fn(x, t)
            return -jnp.sqrt(noise_level) * score

        return noise_fn

    def noise_to_score(self, noise_fn: Callable) -> Callable:
        def score_fn(x: Array, t: Array) -> Array:
            noise_level = self.noise_level(t)
            noise = noise_fn(x, t)
            return -noise / (jnp.sqrt(noise_level) + 1e-6)

        return score_fn


@dataclass
class SDE(DiffusionModel):
    r"""
    dX(t) = -0.5 \beta(t) X(t) dt + \sqrt{\beta(t)} dW(t)
    """

    beta: Schedule
    def __post_init__(self):
        self.tf = self.beta.T  # type: ignore

    def noise_level(self, t: float) -> float:
        """Compute noise level for diffusion process.

        For a diffusion process dX(t) = -0.5 β(t)X(t)dt + √β(t)dW(t):
        - α(t) = exp(-∫β(s)ds) is the signal preservation ratio
        - noise_level = 1 - α(t) is the noise variance

        Solution: X(t) = √α(t) * X₀ + √(1-α(t)) * ε, where ε ~ N(0,I)

        Returns:
            Noise level (1 - α(t)) clipped for numerical stability
        """
        alpha = jnp.exp(-self.beta.integrate(t, 0.0))
        alpha = jnp.clip(alpha, 0.001, 0.9999)
        return 1 - alpha


def check_snr(model: DiffusionModel, t: float, tolerance: float = 1e-3) -> bool:
    """
    Check if SNR at timestep t is effectively zero.

    Args:
        model: DiffusionModel instance
        t: Timestep to check
        tolerance: Tolerance for considering SNR as zero

    Returns:
        True if SNR is effectively zero
    """
    return model.snr(t) < tolerance
