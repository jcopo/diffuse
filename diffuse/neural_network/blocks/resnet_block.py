from typing import Callable, Optional

import jax.numpy as jnp

from einops import rearrange
from flax import nnx
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .timestep import TimestepBlock


class ResnetBlock(TimestepBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        activation: Callable = nnx.swish,
        embedding_dim: int = None,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        dropout: bool = True,
        rngs: nnx.Rngs = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.embedding_dim = embedding_dim

        actual_out_channels = self.in_channels if self.out_channels is None else self.out_channels

        self.norm1 = nnx.GroupNorm(
            num_features=in_channels,
            num_groups=32,
            epsilon=1e-6,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=actual_out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.norm2 = nnx.GroupNorm(
            num_features=actual_out_channels,
            num_groups=32,
            epsilon=1e-6,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=actual_out_channels,
            out_features=actual_out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        if self.in_channels != actual_out_channels:
            self.nin_shortcut = nnx.Conv(
                in_features=in_channels,
                out_features=actual_out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=(0, 0),
                param_dtype=param_dtype,
                dtype=dtype,
                rngs=rngs,
            )

        if dropout:
            self.dropout = nnx.Dropout(rate=0.2, rngs=rngs)
        else:
            self.dropout = lambda x: x

        if self.embedding_dim:
            self.time_mlp = nnx.Linear(
                in_features=self.embedding_dim,
                out_features=actual_out_channels * 2,
                param_dtype=param_dtype,
                dtype=dtype,
                rngs=rngs,
            )

    def __call__(self, x: ArrayLike, time_emb: Optional[ArrayLike] = None) -> Array:
        h = x
        h = self.norm1(h)
        h = self.activation(h)
        h = self.conv1(h)

        if self.embedding_dim and time_emb is not None:
            time_emb = self.time_mlp(self.activation(time_emb))
            time_emb = rearrange(time_emb, "b c -> b 1 1 c")
            scale, shift = jnp.split(time_emb, 2, axis=-1)
            h = h * (scale + 1) + shift

        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        actual_out_channels = self.in_channels if self.out_channels is None else self.out_channels
        if self.in_channels != actual_out_channels:
            x = self.nin_shortcut(x)

        h = h + x

        return h
