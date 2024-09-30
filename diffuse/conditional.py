from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Tuple
import pdb

import einops
import jax
import jax.numpy as jnp
from blackjax.smc.resampling import stratified
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, PRNGKeyArray, PyTreeDef

from diffuse.sde import SDE, SDEState, euler_maryama_step
from diffuse.images import SquareMask


@register_pytree_node_class
class CondState(NamedTuple):
    x: jnp.ndarray  # Current Markov Chain State x_t
    y: jnp.ndarray  # Current Observation Path y_t
    xi: jnp.ndarray  # Measured position of y_t | x_t, xi_t
    t: float

    def tree_flatten(self):
        children = (self.x, self.y, self.xi, self.t)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@dataclass
class CondSDE(SDE):
    mask: SquareMask
    tf: float
    score: Callable[[Array, float], Array]

    def reverse_drift(self, state: SDEState) -> Array:
        x, t = state
        beta_t = self.beta(self.tf - t)
        s = self.score(x, self.tf - t)
        return 0.5 * beta_t * x + beta_t * s

    def reverse_diffusion(self, state: SDEState) -> Array:
        x, t = state
        return jnp.sqrt(self.beta(self.tf - t))

    def logpdf(self, obs: Array, state_p: CondState, dt: float):
        """
        y_{k-1} | y_{k}, x_k ~ N(.| y_k + rev_drift*dt, sqrt(dt)*rev_diff)
        Args:
            obs (Array): The observation y_{k-1}.
            state_p (CondState): The previous state containing:
                - x_p (Array): The particle state x_k.
                - y_p (Array): The observation y_k.
                - xi_p (Array): The measurement position.
                - t_p (float): The time step.
            dt (float): The time step size.

        Returns:
            float: The log probability density of the observation.
        """
        x_p, y_p, xi, t_p = state_p
        # mean = y_p + cond_reverse_drift(state_p, self) * dt
        mean = y_p + self.mask.measure(xi, cond_reverse_drift(state_p, self)) * dt
        std = jnp.sqrt(dt) * cond_reverse_diffusion(state_p, self)

        return jax.scipy.stats.norm.logpdf(obs, mean, std).sum()

    def cond_reverse_step(
        self, state: CondState, dt: float, key: PRNGKeyArray
    ) -> CondState:
        """
        x_{k-1} | x_k, y_k ~ N(.| x_k + rev_drift*dt, sqrt(dt)*rev_diff)
        """
        x, y, xi, t = state

        def revese_drift(state):
            x, t = state
            return cond_reverse_drift(CondState(x, y, xi, t), self)

        def reverse_diffusion(state):
            x, t = state
            return cond_reverse_diffusion(CondState(x, y, xi, t), self)

        # jax.debug.print("meas{}\n", measure(xi, img, self.mask))
        # jax.debug.print("y{}\n", y.shape)
        # jax.debug.print("diff{}\n", measure(xi, img, self.mask) - y )

        x, _ = euler_maryama_step(
            SDEState(x, t), dt, key, revese_drift, reverse_diffusion
        )
        y = self.mask.measure(xi, x)
        return CondState(x, y, xi, t - dt)


def cond_reverse_drift(state: CondState, cond_sde: CondSDE) -> Array:
    # stack together x and y and apply reverse drift
    x, y, xi, t = state
    # img = restore(xi, x, cond_sde.mask, y)
    # return cond_sde.reverse_drift(SDEState(img, t))
    drift_x = cond_sde.reverse_drift(SDEState(x, t))
    beta_t = cond_sde.beta(cond_sde.tf - t)
    meas_x = cond_sde.mask.measure(xi, x)
    alpha_t = jnp.exp(cond_sde.beta.integrate(0.0, t))
    # here if needed we average over y
    drift_y = (
        beta_t * cond_sde.mask.restore(xi, jnp.zeros_like(x), y - meas_x) / alpha_t
    )
    # f = lambda y: beta_t * (y - meas_x) / alpha_t
    # drifts = jax.vmap(f)(y)
    # drift_y = drifts.mean(axis=0)
    return drift_x + drift_y


def cond_reverse_diffusion(state: CondState, cond_sde: CondSDE) -> Array:
    # stack together x and y and apply reverse diffusion
    x, y, xi, t = state
    img = cond_sde.mask.restore(xi, x, y)
    return cond_sde.reverse_diffusion(SDEState(img, t))
