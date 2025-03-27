from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.integrator.base import Integrator
from diffuse.diffusion.sde import SDE

from diffuse.denoisers.base import DenoiserState, BaseDenoiser


@dataclass
class Denoiser(BaseDenoiser):
    """Denoiser"""

    integrator: Integrator
    sde: SDE
    score: Callable[[Array, float], Array]  # x -> t -> score(x, t)
    x0_shape: Tuple[int, ...]

    def init(self, position: Array, rng_key: PRNGKeyArray, dt: float) -> DenoiserState:
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

    def generate(self,
        rng_key: PRNGKeyArray,
        n_steps: int,
        n_particles: int,
    ) -> Tuple[Array, Array]:
        r"""Generate denoised samples \theta_0"""
        rng_key, rng_key_start = jax.random.split(rng_key)

        dt = (self.sde.tf - self.sde.beta.t0) / n_steps

        rndm_start = jax.random.normal(rng_key_start, shape=(n_particles, *self.x0_shape))
        # sample with
        # ppf = jax.scipy.stats.norm.ppf(
        #     jnp.arange(0, n_particles) / n_particles + 1 / (2 * n_particles)
        #     )[:, None]
        keys = jax.random.split(rng_key, n_particles)
        state = jax.vmap(self.init, in_axes=(0, 0, None))(rndm_start, keys, dt)

        def body_fun(state, _):
            state_next = jax.vmap(self.step)(state)
            return state_next, state_next.integrator_state.position

        return jax.lax.scan(body_fun, state, jnp.arange(n_steps))
