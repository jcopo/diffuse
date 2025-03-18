from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Union

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array

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
class CosineSchedule:
    t0: float
    T: float
    dt: float
    s: float = 0.008

    def __call__(self, t):
        return jax.lax.cond(t < 1e-3, lambda: 0., lambda: jnp.clip(1 - self._f(t) / self._f(t - self.dt), max=0.999))

    def _f(self, t):
        return jnp.cos((t + self.s) / (1 + self.s) * jnp.pi / 2) ** 2

    def integrate(self, t, s):
        return self._f(t) / self._f(s)



@dataclass
class SDE:
    r"""
    dX(t) = -0.5 \beta(t) X(t) dt + \sqrt{\beta(t)} dW(t)
    """

    beta: Schedule
    tf: float

    def score(self, state: SDEState, state_0: SDEState) -> Array:
        """
        Close form for the Gaussian thingy \nabla \log p(x_t | x_{t_0})
        """
        x, t = state.position, state.t
        x0, t0 = state_0.position, state_0.t
        int_b = self.beta.integrate(t, t0).squeeze()
        alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)

        return -(x - alpha * x0) / beta

    def tweedie(self, state: SDEState, score: Callable) -> SDEState:
        x, t = state.position, state.t
        int_b = self.beta.integrate(t, self.beta.t0).squeeze()
        alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)
        return SDEState((x + beta * score(x, t)) / alpha, self.beta.t0)

    def path(
        self,
        key: PRNGKeyArray,
        state: SDEState,
        ts: float,
        return_noise: bool = False
    ) -> Union[SDEState, tuple[SDEState, Array]]:
        """
        Generate x_ts | x_t ~ N(.| exp(-0.5 \int_ts^t \beta(s) ds) x_0, 1 - exp(-\int_ts^t \beta(s) ds))

        Args:
            key: Random number generator key
            state: Current state
            ts: Target time
            return_noise: If True, also return the noise used to generate the sample

        Returns:
            If return_noise is False, returns just the new state
            If return_noise is True, returns tuple of (new state, noise used)
        """
        x, t = state.position, state.t

        int_b = self.beta.integrate(ts, t)
        alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)

        noise = jax.random.normal(key, x.shape)
        res = alpha * x + jnp.sqrt(beta) * noise

        return (SDEState(res, ts), noise) if return_noise else SDEState(res, ts)

    def drift(self, state: SDEState) -> Array:
        x, t = state.position, state.t
        return -0.5 * self.beta(t) * x

    def diffusion(self, state: SDEState) -> Array:
        t = state.t
        return jnp.sqrt(self.beta(t))

    def reverse_drift(self, state: SDEState, score: Callable) -> Array:
        x, t = state.position, state.t
        beta_t = self.beta(self.tf - t)
        s = score(x, self.tf - t)
        return 0.5 * beta_t * x + beta_t * s

    def reverse_drift_ode(self, state: SDEState, score: Callable) -> Array:
        x, t = state.position, state.t
        beta_t = self.beta(self.tf - t)
        s = score(x, self.tf - t)
        return 0.5 * (beta_t * x + beta_t * s)

    def reverse_diffusion(self, state: SDEState) -> Array:
        t = state.t
        return jnp.sqrt(self.beta(self.tf - t))

    def score_to_noise(self, score_fn: Callable) -> Callable:
        """
        Convert a score function to a noise prediction function.

        Args:
            score_fn: Function that predicts score, with signature (x, t) -> score

        Returns:
            Function that predicts noise, with signature (x, t) -> noise
        """
        def noise_fn(x: Array, t: Array) -> Array:
            int_b = self.beta.integrate(t, self.beta.t0).squeeze()
            beta = 1 - jnp.exp(-int_b)  # β_t = 1 - α_t
            score = score_fn(x, t)
            return -jnp.sqrt(beta) * score  # ε = -√β * score

        return noise_fn

    def noise_to_score(self, noise_fn: Callable) -> Callable:
        """
        Convert a noise prediction function to a score function.

        Args:
            noise_fn: Function that predicts noise, with signature (x, t) -> noise

        Returns:
            Function that predicts score, with signature (x, t) -> score
        """
        def score_fn(x: Array, t: Array) -> Array:
            int_b = self.beta.integrate(t, self.beta.t0).squeeze()
            beta = 1 - jnp.exp(-int_b)  # β_t = 1 - α_t
            noise = noise_fn(x, t)
            return -noise / (jnp.sqrt(beta) + 1e-6)  # Correct equivalence: score = -ε / √β
        return score_fn