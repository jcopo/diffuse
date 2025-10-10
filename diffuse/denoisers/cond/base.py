# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from abc import abstractmethod
from typing import Optional, NamedTuple
from jaxtyping import Array, PRNGKeyArray
from diffuse.integrator.base import IntegratorState
import jax.numpy as jnp
from dataclasses import dataclass
import jax
from diffuse.diffusion.sde import DiffusionModel
from diffuse.integrator.base import Integrator
from diffuse.base_forward_model import ForwardModel, MeasurementState
from diffuse.utils.mapping import pmapper
from typing import Tuple
from diffuse.denoisers.base import BaseDenoiser
from diffuse.predictor import Predictor


class CondDenoiserState(NamedTuple):
    """Conditional denoiser state"""

    integrator_state: IntegratorState
    log_weights: float = 0.0


@dataclass
class CondDenoiser(BaseDenoiser):
    integrator: Integrator
    model: DiffusionModel
    predictor: Predictor
    forward_model: ForwardModel
    x0_shape: Tuple[int, ...]
    resample: Optional[bool] = False
    ess_low: Optional[float] = 0.2
    ess_high: Optional[float] = 0.5

    def init(self, position: Array, rng_key: PRNGKeyArray, n_particles: int) -> CondDenoiserState:
        """
        Initialize the conditional denoiser state.

        Args:
            position: Initial position array.
            rng_key: Random number generator key.
            n_particles: Number of particles to initialize.

        Returns:
            CondDenoiserState: Initialized state with integrator state and log weights.
        """
        log_weights = -jnp.log(n_particles)
        integrator_state = self.integrator.init(position, rng_key)

        return CondDenoiserState(integrator_state, log_weights)

    def generate(
        self,
        rng_key: PRNGKeyArray,
        measurement_state: MeasurementState,
        n_steps: int,
        n_particles: int,
        keep_history: bool = False,
    ):
        rng_key, rng_key_start = jax.random.split(rng_key)
        rndm_start = jax.random.normal(rng_key_start, (n_particles, *self.x0_shape))

        keys = jax.random.split(rng_key, n_particles)
        state = jax.vmap(self.init, in_axes=(0, 0, None))(rndm_start, keys, n_particles)

        def body_fun(state: CondDenoiserState, key: PRNGKeyArray):
            # state_next = self.batch_step(key, state, posterior, measurement_state)
            keys = jax.random.split(key, state.integrator_state.position.shape[0])
            state_next = jax.vmap(self.step, in_axes=(0, 0, None))(keys, state, measurement_state)
            if self.resample:
                state_next = self.resampler(state_next, measurement_state, key)
            return state_next, state_next.integrator_state.position if keep_history else None

        keys = jax.random.split(rng_key, n_steps)
        return jax.lax.scan(body_fun, state, keys)

    @abstractmethod
    def step(self, rng_key: PRNGKeyArray, state: CondDenoiserState, measurement_state: MeasurementState) -> CondDenoiserState:
        """
        Abstract method to perform a single step of conditional denoising.

        This method should be implemented by subclasses to define how to update the state
        based on the current measurement and random key.

        Args:
            rng_key: Random number generator key for stochastic operations
            state: Current state of the denoiser containing position and weights
            measurement_state: Current measurement state containing observations

        Returns:
            CondDenoiserState: Updated state after performing the denoising step
        """
        pass

    def batch_step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        r"""
        Batching for memory efficiency.
        """

        state_next = pmapper(self.step, state, measurement_state)

        # if self.resample:
        if True:
            state_next = self.resampler(state_next, measurement_state, rng_key)

        return state_next
