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

    def restore(self, measured: Array) -> Array:
        return self.A.T @ measured