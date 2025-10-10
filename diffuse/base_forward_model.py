# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from typing import Protocol, NamedTuple
from jaxtyping import Array


class MeasurementState(NamedTuple):
    """Container for measurement information during conditional sampling.

    Attributes:
        y: The measured/observed data
        mask_history: History of measurement masks (for partial observations)
    """

    y: Array
    mask_history: Array


class ForwardModel(Protocol):
    """Protocol defining the interface for forward models in inverse problems.

    Forward models implement measurement operators and their adjoint operators
    for conditional generation tasks (e.g., inpainting, super-resolution, denoising).

    Attributes:
        std: Standard deviation of measurement noise
    """

    std: float

    def apply(self, img: Array, measurement_state: MeasurementState) -> Array:
        """Apply the forward measurement operator.

        Args:
            img: Input image/data
            measurement_state: Current measurement state

        Returns:
            Measured/degraded output
        """
        ...

    def restore(self, img: Array, measurement_state: MeasurementState) -> Array:
        """Apply the adjoint operator.

        Args:
            img: Data to apply adjoint to
            measurement_state: Current measurement state

        Returns:
            Output of adjoint operator
        """
        ...
