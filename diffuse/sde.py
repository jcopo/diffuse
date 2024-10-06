import pdb
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTreeDef, Array


class SDEState(NamedTuple):
    position: PyTreeDef
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
class SDE:
    r"""
    dX(t) = -0.5 \beta(t) X(t) dt + \sqrt{\beta(t)} dW(t)
    """

    beta: Schedule

    def score(self, state: SDEState, state_0: SDEState) -> PyTreeDef:
        """
        Close form for the Gaussian thingy \nabla \log p(x_t | x_{t_0})
        """
        x, t = state
        x0, t0 = state_0
        int_b = self.beta.integrate(t, t0).squeeze()
        alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)

        return -(x - alpha * x0) / beta

    def path(self, key: PRNGKeyArray, state: SDEState, ts: float) -> SDEState:
        """
        Generate x_ts | x_t ~ N(.| exp(-0.5 \int_ts^t \beta(s) ds) x_0, 1 - exp(-\int_ts^t \beta(s) ds))
        """
        x, t = state

        int_b = self.beta.integrate(ts, t)
        alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)

        rndm = jax.random.normal(key, x.shape)
        res = alpha * x + jnp.sqrt(beta) * rndm
        return SDEState(res, ts)

    def drift(self, state: SDEState) -> PyTreeDef:
        x, t = state
        return -0.5 * self.beta(t) * x

    def diffusion(self, state: SDEState) -> PyTreeDef:
        x, t = state
        return jnp.sqrt(self.beta(t))

    def reverso(
        self, key: PRNGKeyArray, state_tf: SDEState, score: Callable, dts: float
    ) -> SDEState:
        x_tf, tf = state_tf
        state_tf_0 = SDEState(x_tf, jnp.array([0.0]))

        def reverse_drift(state):
            x, t = state
            beta_t = self.beta(tf - t)
            s = score(x, tf - t)
            return 0.5 * beta_t * x + beta_t * s

        def reverse_diffusion(state):
            x, t = state
            return jnp.sqrt(self.beta(tf - t))

        step = partial(
            euler_maryama_step, drift=reverse_drift, diffusion=reverse_diffusion
        )

        def body_fun(state, tup):
            dt, key = tup
            next_state = step(state, dt, key)
            return next_state, next_state

        n_dt = dts.shape[0]
        keys = jax.random.split(key, n_dt)
        state_f, history = jax.lax.scan(body_fun, state_tf_0, (dts, keys))
        # stack initial state into history
        history = jax.tree_map(
            lambda arr, x: jnp.concatenate([arr[None], x]), state_tf_0, history
        )
        return state_f, history


def euler_maryama_step(
    state: SDEState, dt: float, key: PRNGKeyArray, drift: Callable, diffusion: Callable
) -> SDEState:
    dx = drift(state) * dt + diffusion(state) * jax.random.normal(
        key, state.position.shape
    ) * jnp.sqrt(dt)
    return SDEState(state.position + dx, state.t + dt)


def euler_maryama_step_array(
    state: SDEState, dt: float, key: PRNGKeyArray, drift: Array, diffusion: Array
) -> SDEState:
    dx = drift * dt + diffusion * jax.random.normal(
        key, state.position.shape
    ) * jnp.sqrt(dt)
    return SDEState(state.position + dx, state.t + dt)
