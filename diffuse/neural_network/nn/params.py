# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from chex import dataclass

from jax import Array


@dataclass
class CondUNet2DOutput:
    """Output of the CondUNet2D model.

    Attributes:
        output: The processed output tensor from the U-Net
    """

    output: Array


@dataclass
class SDVaeOutput:
    """Output of the Stable Diffusion VAE model.

    Attributes:
        output: The reconstructed/decoded output tensor
        mean: Mean of the latent distribution
        logvar: Log variance of the latent distribution
    """

    output: Array
    mean: Array
    logvar: Array
