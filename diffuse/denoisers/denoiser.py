
from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from blackjax.smc.resampling import stratified

from diffuse.integrator.base import Integrator, IntegratorState
from diffuse.diffusion.sde import SDE, SDEState


class DenoiserState(NamedTuple):
    integrator_state: IntegratorState


@dataclass
class Denoiser:
    """Denoiser"""
    integrator: Integrator
    logpdf: Callable[[SDEState, Array], Array] # x -> t -> logpdf(x, t)
    sde: SDE
    score: Callable[[Array, float], Array] # x -> t -> score(x, t)

    def init(self, position: Array, rng_key: PRNGKeyArray, dt: float) -> DenoiserState:
        integrator_state = self.integrator.init(position, rng_key, 0., dt)
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

    def generate(self, rng_key: PRNGKeyArray, dt: float, tf: float, shape: Tuple[int, ...]) -> Array:
        """Generate samples from the denoiser"""
        rng_key, rng_key_start = jax.random.split(rng_key)

        rndm_start = jax.random.normal(rng_key_start, shape=shape)
        state = self.init(rndm_start, rng_key, dt)
        n_steps = int(tf / dt)

        def body_fun(state, _):
            return self.step(state), None

        return jax.lax.scan(body_fun, state, jnp.arange(n_steps))
