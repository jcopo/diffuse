from dataclasses import dataclass
from typing import Callable, Tuple, Union, Optional, Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.integrator.base import Integrator
from diffuse.diffusion.sde import SDE, DiffusionModel
from diffuse.predictor import Predictor

from diffuse.denoisers.base import DenoiserState, BaseDenoiser


@dataclass
class Denoiser(BaseDenoiser):
    """Denoiser"""

    integrator: Integrator
    model: DiffusionModel
    predictor: Predictor
    x0_shape: Tuple[int, ...]

    def init(self, position: Array, rng_key: PRNGKeyArray) -> DenoiserState:
        integrator_state = self.integrator.init(position, rng_key)
        return DenoiserState(integrator_state)

    def step(
        self,
        state: DenoiserState,
    ) -> DenoiserState:
        r"""
        sample p(\theta_t-1 | \theta_t)
        """
        integrator_state = state.integrator_state
        integrator_state_next = self.integrator(integrator_state, self.predictor)

        return DenoiserState(integrator_state_next)

    def generate(
        self,
        rng_key: PRNGKeyArray,
        n_steps: int,
        n_particles: int,
        keep_history: bool = False,
        data_sharding: Optional[Any] = None,
    ) -> Tuple[DenoiserState, Union[Array, None]]:
        r"""Generate denoised samples \theta_0"""
        rng_key, rng_key_start = jax.random.split(rng_key)

        rndm_start = jax.random.normal(rng_key_start, shape=(n_particles, *self.x0_shape))

        # Shard the initial noise if sharding is provided
        if data_sharding is not None:
            rndm_start = jax.device_put(rndm_start, data_sharding)

        keys = jax.random.split(rng_key, n_particles)

        # Also shard the keys
        if data_sharding is not None:
            keys = jax.device_put(keys, data_sharding)

        state = jax.vmap(self.init, in_axes=(0, 0))(rndm_start, keys)

        def body_fun(state, _):
            state_next = jax.vmap(self.step)(state)
            return state_next, state_next.integrator_state.position if keep_history else None

        return jax.lax.scan(body_fun, state, jnp.arange(n_steps))
