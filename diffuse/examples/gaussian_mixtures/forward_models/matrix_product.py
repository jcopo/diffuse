# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
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

    def measure(self, key: PRNGKeyArray, x: Array, *args) -> Array:
        """
        Measure the output of the forward model:
        y = A x + \epsilon

        Args:
            key: Random key.
            x: Input to the forward model.

        Returns:
            Array: Result of measuring the output of the forward model.
        """
        _y = self.A @ x
        return _y + jax.random.normal(key, shape=_y.shape) * self.std

    def apply(self, x: Array, *args) -> Array:
        """
        Apply the forward model to the input:
        y = A x

        Args:
            x: Input to the forward model.
            *args: Additional arguments (not used in this implementation).

        Returns:
            Array: Result of applying the forward model to the input.
        """
        return self.A @ x

    def restore(self, measured: Array, *args) -> Array:
        """
        Restore the input from the measured output x = A^T y

        Args:
            measured: Measured output.
            *args: Additional arguments (not used in this implementation).

        Returns:
            Array: Result of restoring the input from the measured output.
        """
        return self.A.T @ measured
