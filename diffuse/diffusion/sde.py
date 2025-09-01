from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, NamedTuple, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class SDEState(NamedTuple):
    position: Array
    t: Array


class Schedule(ABC):
    T: float

    @abstractmethod
    def __call__(self, t: Array) -> Array:
        pass

    @abstractmethod
    def integrate(self, t: Array, s: Array) -> Array:
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

    def __call__(self, t: Array) -> Array:
        """
        Calculates the value of the linear schedule at a given time.

        Args:
            t (Array): The time at which to evaluate the schedule.

        Returns:
            Array: The value of the linear schedule at time t.
        """
        b_min, b_max, t0, T = self.b_min, self.b_max, self.t0, self.T
        return (b_max - b_min) / (T - t0) * t + (b_min * T - b_max * t0) / (T - t0)

    def integrate(self, t: Array, s: Array) -> Array:
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

    def __call__(self, t: Array) -> Array:
        """
        Calculates the value of the cosine schedule at a given time.
        """
        t_normalized = (t - self.t0) / (self.T - self.t0)

        beta_t = jnp.pi * jnp.tan(0.5 * jnp.pi * (t_normalized + self.s) / (1 + self.s)) / (self.T * (1 + self.s))
        beta_t = jnp.clip(beta_t, self.b_min, self.b_max)

        return beta_t

    def integrate(self, t: Array, s: Array) -> Array:
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
    def noise_level(self, t: Array) -> Array:
        pass

    @abstractmethod
    def signal_level(self, t: Array) -> Array:
        pass

    def snr(self, t: Array) -> Array:
        """
        Compute Signal-to-Noise Ratio (SNR) at timestep t.

        For general interpolation x_t = α_t x_0 + σ_t ε:
        SNR(t) = α_t² / σ_t²
        """
        noise_level = self.noise_level(t)
        signal_level = self.signal_level(t)
        return (signal_level * signal_level) / (noise_level * noise_level + 1e-8)

    def score(self, state: SDEState, state_0: SDEState) -> Array:
        """
        Closed-form expression for the score function ∇ₓ log p(xₜ | x₀) of the Gaussian transition kernel.

        From docs: ∇log p_t(x_t|x_0) = -1/σ_t² (x_t - α_t x_0)
        """
        x, t = state.position, state.t
        x0, t0 = state_0.position, state_0.t
        sigma_t = self.noise_level(t)
        signal_level_t = self.signal_level(t)

        return -(x - signal_level_t * x0) / (sigma_t * sigma_t)

    def tweedie(self, state: SDEState, score_fn: Callable) -> SDEState:
        """
        Tweedie's formula to compute E[x_0 | x_t].

        From docs: x̂_0 = 1/α_t (x_t + σ_t² ∇log p_t(x_t))
        """
        x, t = state.position, state.t
        sigma_t = self.noise_level(t)
        signal_level_t = self.signal_level(t)
        return SDEState((x + sigma_t * sigma_t * score_fn(x, t)) / signal_level_t, jnp.zeros_like(t))

    def path(
        self, key: PRNGKeyArray, state: SDEState, ts: Array, return_noise: bool = False
    ) -> Union[SDEState, tuple[SDEState, Array]]:
        """
        Samples from the general interpolation: x_t = α_t x_0 + σ_t ε
        """
        x = state.position
        sigma_t = self.noise_level(ts)
        signal_level_t = self.signal_level(ts)

        noise = jax.random.normal(key, x.shape)
        res = signal_level_t * x + sigma_t * noise
        return (SDEState(res, ts), noise) if return_noise else SDEState(res, ts)

    def score_to_noise(self, score_fn: Callable) -> Callable:
        def noise_fn(x: Array, t: Array) -> Array:
            sigma_t = self.noise_level(t)
            score = score_fn(x, t)
            return -sigma_t * score

        return noise_fn

    def noise_to_score(self, noise_fn: Callable) -> Callable:
        def score_fn(x: Array, t: Array) -> Array:
            sigma_t = self.noise_level(t)
            noise = noise_fn(x, t)
            return -noise / (sigma_t + 1e-6)

        return score_fn


@dataclass
class SDE(DiffusionModel):
    r"""
    dX(t) = -0.5 \beta(t) X(t) dt + \sqrt{\beta(t)} dW(t)
    dx_t = f(t)x_t dt + g(t) dW(t)
    """

    beta: Schedule

    def __post_init__(self):
        self.tf = self.beta.T

    def noise_level(self, t: Array) -> Array:
        """Compute noise level for diffusion process.

        For a diffusion process dX(t) = -0.5 β(t)X(t)dt + √β(t)dW(t):
        - α(t) = exp(-∫β(s)ds) is the signal preservation ratio
        - noise_level = 1 - α(t) is the noise variance

        Solution: X(t) = √α(t) * X₀ + √(1-α(t)) * ε, where ε ~ N(0,I)

        Returns:
            Noise level (1 - α(t)) clipped for numerical stability
        """
        alpha = jnp.exp(-self.beta.integrate(t, jnp.zeros_like(t)))
        sigma = jnp.sqrt(1 - alpha)
        sigma = jnp.clip(sigma, 0.001, 0.9999)
        return sigma

    def signal_level(self, t: Array) -> Array:
        alpha = jnp.sqrt(jnp.exp(-self.beta.integrate(t, jnp.zeros_like(t))))
        alpha = jnp.clip(alpha, 0.001, 0.9999)
        return alpha


@dataclass
class Flow(DiffusionModel):
    """Rectified Flow diffusion model with straight-line interpolation paths.

    Implements the rectified flow formulation from Liu et al. (2022) using:
    - α(t) = 1 - t (signal level decreases linearly)
    - σ(t) = t (noise level increases linearly)

    This creates straight-line paths in the interpolation x_t = (1-t)x_0 + t*ε,
    which are more amenable to ODE-based sampling with fewer discretization steps.

    References:
        Liu, X., Gong, C., & Liu, Q. (2022). Flow straight and fast: Learning to
        generate and transfer data with rectified flow. arXiv:2209.03003
    """

    tf: float = 1.0

    def noise_level(self, t: Array) -> Array:
        """Compute noise level σ(t) = t."""
        return jnp.clip(t / self.tf, 0.001, 0.999)

    def signal_level(self, t: Array) -> Array:
        """Compute signal level α(t) = 1 - t."""
        return jnp.clip(1 - t / self.tf, 0.001, 0.999)


@dataclass
class EDM(DiffusionModel):
    """Efficient Diffusion Model (EDM) from Karras et al. (2022).

    Implements the EDM formulation using:
    - α(t) = 1 (signal level remains constant)
    - σ(t) = t (noise level increases linearly)

    This creates the interpolation x_t = x_0 + t*ε, which simplifies the
    probability-flow ODE and is solved using Heun's method.

    References:
        Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the
        design space of diffusion-based generative models. NeurIPS 35, 26565-26577.
    """

    tf: float = 1.0

    def noise_level(self, t: Array) -> Array:
        """Compute noise level σ(t) = t."""
        return jnp.clip(t, 0.001, 0.999)

    def signal_level(self, t: Array) -> Array:
        """Compute signal level α(t) = 1."""
        return jnp.ones_like(t)


def check_snr(model: DiffusionModel, t: Array, tolerance: float = 1e-3) -> Array:
    """
    Check if SNR at timestep t is effectively zero.

    Args:
        model: DiffusionModel instance
        t: Timestep to check
        tolerance: Tolerance for considering SNR as zero

    Returns:
        True if SNR is effectively zero
    """
    return jnp.all(model.snr(t) < tolerance)
