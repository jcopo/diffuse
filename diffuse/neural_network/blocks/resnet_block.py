# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""ResNet block with optional timestep conditioning for diffusion models.

This module implements a ResNet block that can optionally receive timestep embeddings
for conditioning in diffusion models. Supports FiLM (Feature-wise Linear Modulation)
conditioning and configurable dropout.
"""

from collections.abc import Callable

import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .timestep import TimestepBlock


class ResnetBlock(TimestepBlock):
    """ResNet block with optional timestep conditioning.

    A residual block consisting of two convolutional layers with GroupNorm and
    activation functions. Optionally accepts timestep embeddings for FiLM conditioning
    in diffusion models. Includes skip connections for residual learning.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels. If None, uses in_channels.
        activation: Activation function to use. Default is swish.
        embedding_dim: Dimension of timestep embeddings for conditioning. If None,
            no timestep conditioning is applied.
        param_dtype: Data type for parameters.
        dtype: Data type for computation.
        dropout: Whether to apply dropout.
        dropout_rate: Dropout rate when dropout is enabled.
        rngs: Random number generators for parameter initialization.

    Example:
        >>> rngs = nnx.Rngs(42)
        >>> block = ResnetBlock(in_channels=128, out_channels=256, rngs=rngs)
        >>> x = jnp.ones((2, 32, 32, 128))
        >>> output = block(x)  # Shape: (2, 32, 32, 256)
        >>>
        >>> # With timestep conditioning
        >>> block_cond = ResnetBlock(in_channels=128, embedding_dim=512, rngs=rngs)
        >>> time_emb = jnp.ones((2, 512))
        >>> output = block_cond(x, time_emb)
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int | None = None,
        activation: Callable = nnx.swish,
        embedding_dim: int | None = None,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        dropout: bool = True,
        dropout_rate: float = 0.1,
        rngs: nnx.Rngs = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.embedding_dim = embedding_dim

        self.out_channels = self.in_channels if self.out_channels is None else self.out_channels

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
            out_features=self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.norm2 = nnx.GroupNorm(
            num_features=self.out_channels,
            num_groups=32,
            epsilon=1e-6,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=self.out_channels,
            out_features=self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nnx.Conv(
                in_features=in_channels,
                out_features=self.out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=(0, 0),
                param_dtype=param_dtype,
                dtype=dtype,
                rngs=rngs,
            )

        if dropout:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = lambda x: x

        if self.embedding_dim:
            self.time_mlp = nnx.Linear(
                in_features=self.embedding_dim,
                out_features=self.out_channels * 2,
                param_dtype=param_dtype,
                dtype=dtype,
                rngs=rngs,
            )

    def __call__(self, x: ArrayLike, time_emb: ArrayLike | None = None) -> Array:
        """Forward pass with optional timestep conditioning.

        Args:
            x: Input feature map of shape (batch, height, width, in_channels).
            time_emb: Optional timestep embedding of shape (batch, embedding_dim).
                Only used if embedding_dim was provided during initialization.

        Returns:
            Output feature map with residual connection applied.
        """
        h = x
        h = self.norm1(h)
        h = self.activation(h)
        h = self.dropout(h)
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

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return h + x
