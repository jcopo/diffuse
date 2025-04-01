from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Tuple, Union

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
        return (
            0.5
            * (t - s)
            * (
                (b_max - b_min) / (T - t0) * (t + s)
                + 2 * (b_min * T - b_max * t0) / (T - t0)
            )
        )


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

        beta_t = jnp.pi * jnp.tan(0.5 * jnp.pi * (t_normalized + self.s) / (1 + self.s)) / (1 + self.s)
        beta_t = jnp.clip(beta_t, self.b_min, self.b_max)

        return beta_t

    def integrate(self, t, s):
        t_normalized = (t - self.t0) / (self.T - self.t0)
        ft = jnp.cos((t_normalized + self.s) / (1 + self.s) * jnp.pi * 0.5) ** 2
        f0 = jnp.cos(self.s / (1 + self.s) * jnp.pi * 0.5) ** 2
        alpha_t = jnp.clip(ft / f0, 0.001, 0.9999)

        s_normalized = (s - self.t0) / (self.T - self.t0)
        fs = jnp.cos((s_normalized + self.s) / (1 + self.s) * jnp.pi * 0.5) ** 2
        f0 = jnp.cos(self.s / (1 + self.s) * jnp.pi * 0.5) ** 2
        alpha_s = jnp.clip(fs / f0, 0.001, 0.9999)

        return jnp.log(alpha_s / alpha_t)


class DiffusionModel(ABC):
    @abstractmethod
    def alpha_beta(self, t: float) -> Tuple[float, float]:
        pass

    def score(self, state: SDEState, state_0: SDEState) -> Array:
        """
        Close form for the Gaussian thingy \nabla \log p(x_t | x_{t_0})
        """
        x, t = state.position, state.t
        x0, t0 = state_0.position, state_0.t
        alpha_t, _ = self.alpha_beta(t)

        return -(x - jnp.sqrt(alpha_t) * x0) / (1 - alpha_t)

    def tweedie(self, state: SDEState, score_fn: Callable) -> SDEState:
        """
        Tweedie's formula to compute E[x_t | x_0]
        """
        x, t = state.position, state.t
        alpha_t, _ = self.alpha_beta(t)
        return SDEState(
            (x + (1 - alpha_t) * score_fn(x, t)) / jnp.sqrt(alpha_t), 0.
        )

    def path(self, key: PRNGKeyArray, state: SDEState, ts: Array, return_noise: bool = False
    ) -> Union[SDEState, tuple[SDEState, Array]]:
        """
        Samples x_t | x_0 ~ N(sqrt(alpha_t) * x_0, (1 - alpha_t) * I)
        """
        x = state.position
        alpha_t, _ = self.alpha_beta(ts)

        noise = jax.random.normal(key, x.shape)
        res = jnp.sqrt(alpha_t) * x + jnp.sqrt(1 - alpha_t) * noise
        return (SDEState(res, ts), noise) if return_noise else SDEState(res, ts)

    def score_to_noise(self, score_fn: Callable) -> Callable:
        def noise_fn(x: Array, t: Array) -> Array:
            alpha_t, _ = self.alpha_beta(t)
            score = score_fn(x, t)
            return -jnp.sqrt(1 - alpha_t) * score

        return noise_fn

    def noise_to_score(self, noise_fn: Callable) -> Callable:
        def score_fn(x: Array, t: Array) -> Array:
            alpha_t, _ = self.alpha_beta(t)
            noise = noise_fn(x, t)
            return -noise / (jnp.sqrt(1 - alpha_t) + 1e-6)

        return score_fn


@dataclass
class SDE(DiffusionModel):
    r"""
    dX(t) = -0.5 \beta(t) X(t) dt + \sqrt{\beta(t)} dW(t)
    """

    beta: Schedule
    tf: float

    def alpha_beta(self, t: float) -> Tuple[float, float]:
        alpha = jnp.exp(-self.beta.integrate(t, 0.))
        beta = self.beta(t)
        return alpha, beta
