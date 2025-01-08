from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.integrator.base import Integrator, IntegratorState
from diffuse.diffusion.sde import SDE


class DenoiserState(NamedTuple):
    integrator_state: IntegratorState


@dataclass
class Denoiser:
    """Denoiser"""

    integrator: Integrator
    sde: SDE
    score: Callable[[Array, float], Array]  # x -> t -> score(x, t)
    n_steps: int
    x0_shape: Tuple[int, ...]
    deterministic: bool = False

    def init(self, position: Array, rng_key: PRNGKeyArray, dt: float) -> DenoiserState:
        if self.deterministic:
            integrator_state = self.integrator.init(position, 0.0, dt)
        else:
            integrator_state = self.integrator.init(position, rng_key, 0.0, dt)
        return DenoiserState(integrator_state)

    def step(
        self,
        state: DenoiserState,
    ) -> DenoiserState:
        r"""
        sample p(\theta_t-1 | \theta_t)
        """
        integrator_state = state.integrator_state
        integrator_state_next = self.integrator(integrator_state, self.score)

        return DenoiserState(integrator_state_next)

    def generate(self, rng_key: PRNGKeyArray) -> Tuple[Array, Array]:
        r"""Generate denoised samples \theta_0"""
        rng_key, rng_key_start = jax.random.split(rng_key)
        dt = self.sde.tf / (self.n_steps)

        rndm_start = jax.random.normal(rng_key_start, shape=self.x0_shape)
        state = self.init(rndm_start, rng_key, dt)

        def body_fun(state, _):
            state_next = self.step(state)
            return state_next, state_next.integrator_state.position

        return jax.lax.scan(body_fun, state, jnp.arange(self.n_steps))
