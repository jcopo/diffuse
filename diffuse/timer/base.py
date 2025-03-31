from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable
import jax.numpy as jnp
from jax.random import PRNGKeyArray


class Timer(ABC):
    """Abstract base class for timers"""

    @abstractmethod
    def __call__(self, step: int) -> float:
        pass


class VpTimer(Timer):
    n_steps: int
    eps: float

    def __call__(self, step: int) -> float:
        return 1 + step / self.n_steps * (self.eps - 1)


