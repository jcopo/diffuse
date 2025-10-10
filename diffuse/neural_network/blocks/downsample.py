# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
import jax.numpy as jnp

from flax import nnx
from jax import Array
from jax.typing import ArrayLike, DTypeLike


class Downsample(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=(0, 0),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike) -> Array:
        pad_width = ((0, 0), (0, 1), (0, 1), (0, 0))
        x = jnp.pad(array=x, pad_width=pad_width, mode="constant", constant_values=0)
        return self.conv(x)
