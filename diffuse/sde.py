import jax
from jaxtyping import PyTreeDef, PRNGKeyArray
from typing import NamedTuple, Callable
from functools import partial

from dataclasses import dataclass
from abc import ABC, abstractmethod
import jax.numpy as jnp


class SDEState(NamedTuple):
    position: PyTreeDef
    t: float

class Schedule(ABC):
    @abstractmethod
    def __call__(self, t:float)->float:
        pass
    
    @abstractmethod
    def integrate(self, t:float, s:float)->float:
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
        return 0.5 * (t - s) * ((b_max - b_min) / (T - t0) * (t + s)
                                + 2 * (b_min * T - b_max * t0) / (T - t0))


@dataclass
class SDE:
    r"""
    dX(t) = -0.5 \beta(t) X(t) dt + \sqrt{\beta(t)} dW(t)
    """
    beta: Schedule

    def score(self, state:SDEState, state_0:SDEState)->PyTreeDef:
        """
        Close form for the Gaussian thingy \nabla \log p(x_t | x_{t_0})
        """
        x, t  = state
        x0, t0 = state_0
        int_b = self.beta.integrate(t, t0)
        alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)
        return -(x - alpha * x0) / beta

    def path(self, key:PRNGKeyArray, state:SDEState, dt:float)->SDEState:
        """
        Generate a path
        """
        x, t = state
        key1, key2 = jax.random.split(key)
        int_b = self.beta.integrate(t + dt, t)
        alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)
        return SDEState(x * alpha + jnp.sqrt(beta) * jax.random.normal(key1, x.shape), t + dt)

    def reverso(self, key:PRNGKeyArray, state_tf:SDEState, score:Callable, dts:float)->SDEState:
        x_tf, tf = state_tf
        def reverse_drift(state):
            x, t = state
            beta_t = self.beta(tf - t)
            return -0.5 *  beta_t * x - beta_t * score(SDEState(x, tf - t))
        
        def reverse_diffusion(state):
            x, t = state
            return jnp.sqrt(self.beta(tf - t)) * x

        step = partial(euler_maryama_step, drift=reverse_drift, diffusion=reverse_diffusion)
        def body_fun(state, tup):
            dt, key = tup
            next_state = step(state, dt, key)
            return next_state, next_state

        n_dt = dts.shape[0]
        keys = jax.random.split(key, n_dt)
        return jax.lax.scan(body_fun, state_tf, (dts, keys))


def euler_maryama_step(state:SDEState, dt:float, key:PRNGKeyArray, drift:Callable, diffusion:Callable)->SDEState:
    dx = drift(state) * dt + diffusion(state) * jax.random.normal(key, state.position.shape) * jnp.sqrt(dt)
    return SDEState(state.position + dx, state.t + dt)