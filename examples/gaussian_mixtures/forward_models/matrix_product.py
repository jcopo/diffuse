from dataclasses import dataclass
from typing import NamedTuple

import jax
from jaxtyping import Array, PRNGKeyArray


class MeasurementState(NamedTuple):
    y: Array
    mask_history: Array

@dataclass
class MatrixProduct:
    A: Array
    std: float

    def measure(self, key: PRNGKeyArray, design: Array, x: Array) -> Array:
        return self.A @ x + jax.random.normal(key, shape=x.shape) * self.std

    def measure_from_mask(self, key: PRNGKeyArray, mask: Array, x: Array) -> Array:
        return self.A @ x + jax.random.normal(key, shape=x.shape) * self.std

    def restore_from_mask(self, mask: Array, x: Array, measured: Array) -> Array:
        return self.A.T @ measured

    def restore(self, measured: Array) -> Array:
        return self.A.T @ measured