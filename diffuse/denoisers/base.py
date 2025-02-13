from abc import ABC, abstractmethod
from typing import Callable, NamedTuple
from jaxtyping import Array, PRNGKeyArray
from diffuse.integrator.base import IntegratorState


class DenoiserState(NamedTuple):
    """Base state for all denoisers"""

    integrator_state: IntegratorState


class BaseDenoiser(ABC):
    @abstractmethod
    def init(self, position: Array, rng_key: PRNGKeyArray, dt: float) -> DenoiserState:
        """Initialize denoiser state"""
        pass

    @abstractmethod
    def step(
        self, state: DenoiserState, score: Callable[[Array, float], Array]
    ) -> DenoiserState:
        """Perform single denoising step"""
        pass

    @abstractmethod
    def generate(
        self, rng_key: PRNGKeyArray, measurement_state, n_steps: int, n_particles: int
    ):
        """Generate samples"""
        pass
