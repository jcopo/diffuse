from chex import dataclass

from jax import Array


@dataclass
class CondUNet2DOutput:
    """Output of the CondUNet2D model."""

    output: Array


@dataclass
class SDVaeOutput:
    """Output of the SDVae model."""

    output: Array
    mean: Array
    logvar: Array
