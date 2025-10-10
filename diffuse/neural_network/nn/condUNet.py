# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike, DTypeLike

from ..blocks import (
    AttnBlock,
    Downsample,
    ResnetBlock,
    TimestepEmbedding,
    TimestepEmbedSequential,
    Timesteps,
    Upsample,
)
from .params import CondUNet2DOutput


class CondUNet2D(nnx.Module):
    """Conditional U-Net for diffusion models with timestep conditioning.

    A U-Net architecture for conditional image generation, featuring hierarchical
    downsampling/upsampling paths with skip connections, ResNet blocks, attention
    mechanisms, and sinusoidal timestep embeddings.

    Args:
        in_channels: Number of input channels.
        ch: Base number of channels.
        ch_mult: Channel multipliers for each resolution level.
        num_res_blocks: Number of ResNet blocks at each resolution.
        attention_resolutions: Resolution levels where attention is applied.
        activation: Activation function to use throughout the network.
        dropout: Whether to enable dropout in ResNet blocks.
        num_heads: Number of attention heads for multi-head attention.
        param_dtype: Data type for parameters.
        dtype: Data type for computation.
        rngs: Random number generators for parameter initialization.

    Example:
        >>> rngs = nnx.Rngs(42)
        >>> unet = CondUNet2D(ch=128, attention_resolutions=(8, 16), rngs=rngs)
        >>> x = jnp.ones((2, 64, 64, 3))
        >>> t = jnp.array([100, 200])
        >>> output = unet(x, t)
    """

    def __init__(
        self,
        in_channels: int = 3,
        ch: int = 64,
        ch_mult: tuple[int, ...] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (1, 2, 4, 8),
        activation: Callable = nnx.swish,
        dropout: bool = True,
        dropout_rate: float = 0.1,
        num_heads: int = 8,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.param_dtype = param_dtype
        self.dtype = dtype

        time_embed_dim = ch * 4

        self.time_proj = Timesteps(embedding_dim=time_embed_dim, max_period=10_000, dtype=dtype)
        self.time_embedding = TimestepEmbedding(
            embedding_dim=time_embed_dim,
            activation=activation,
            param_dtype=self.param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        current_ch = int(ch_mult[0] * ch)
        conv_in = nnx.Conv(
            in_features=in_channels,
            out_features=current_ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        blocks_down = [TimestepEmbedSequential(conv_in)]

        input_block_channels = [current_ch]
        ds = 1
        for level, mult in enumerate(ch_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock(
                        in_channels=current_ch,
                        out_channels=int(mult * ch),
                        activation=activation,
                        embedding_dim=time_embed_dim,
                        param_dtype=self.param_dtype,
                        dtype=dtype,
                        dropout=dropout,
                        dropout_rate=dropout_rate,
                        rngs=rngs,
                    )
                ]
                current_ch = int(mult * ch)
                if ds in attention_resolutions:
                    layers.append(
                        AttnBlock(
                            in_channels=current_ch,
                            num_heads=num_heads,
                            param_dtype=param_dtype,
                            dtype=dtype,
                            rngs=rngs,
                        )
                    )

                blocks_down.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(current_ch)

            if level != len(ch_mult) - 1:
                blocks_down.append(
                    TimestepEmbedSequential(
                        Downsample(
                            in_channels=current_ch,
                            param_dtype=self.param_dtype,
                            dtype=dtype,
                            rngs=rngs,
                        )
                    )
                )
                input_block_channels.append(current_ch)
                ds *= 2

        self.blocks_down = blocks_down

        # mid
        self.block_mid = TimestepEmbedSequential(
            ResnetBlock(
                in_channels=current_ch,
                out_channels=current_ch,
                activation=activation,
                embedding_dim=time_embed_dim,
                param_dtype=self.param_dtype,
                dtype=dtype,
                dropout=dropout,
                dropout_rate=dropout_rate,
                rngs=rngs,
            ),
            AttnBlock(
                in_channels=current_ch,
                num_heads=num_heads,
                param_dtype=param_dtype,
                dtype=dtype,
                rngs=rngs,
            ),
            ResnetBlock(
                in_channels=current_ch,
                out_channels=current_ch,
                activation=activation,
                embedding_dim=time_embed_dim,
                param_dtype=self.param_dtype,
                dtype=dtype,
                dropout=dropout,
                dropout_rate=dropout_rate,
                rngs=rngs,
            ),
        )

        # up
        blocks_up = []
        for level, mult in reversed(list(enumerate(ch_mult))):
            for i in range(num_res_blocks + 1):
                ich = input_block_channels.pop()
                layers = [
                    ResnetBlock(
                        in_channels=current_ch + ich,
                        out_channels=int(mult * ch),
                        activation=activation,
                        embedding_dim=time_embed_dim,
                        param_dtype=self.param_dtype,
                        dtype=dtype,
                        dropout=dropout,
                        dropout_rate=dropout_rate,
                        rngs=rngs,
                    )
                ]

                current_ch = int(mult * ch)
                if ds in attention_resolutions:
                    layers.append(
                        AttnBlock(
                            in_channels=current_ch,
                            num_heads=num_heads,
                            param_dtype=self.param_dtype,
                            dtype=dtype,
                            rngs=rngs,
                        )
                    )

                if level and i == num_res_blocks:
                    layers.append(
                        Upsample(
                            in_channels=current_ch,
                            method="pixel_shuffle",
                            scale_factor=2,
                            param_dtype=param_dtype,
                            dtype=dtype,
                            rngs=rngs,
                        )
                    )
                    ds //= 2

                blocks_up.append(TimestepEmbedSequential(*layers))
        self.blocks_up = blocks_up

        norm_out = nnx.GroupNorm(
            num_features=current_ch,
            num_groups=32,
            epsilon=1e-6,
            param_dtype=self.param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        activation_out = activation

        conv_out = nnx.Conv(
            in_features=current_ch,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=self.param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        self.out = nnx.Sequential(norm_out, activation_out, conv_out)

    def __call__(self, x: ArrayLike, t: ArrayLike | None = None) -> CondUNet2DOutput:
        """Forward pass through the U-Net with timestep conditioning.

        Args:
            x: Input tensor of shape (batch, channels, height, width).
            t: Timestep values for conditioning. If None, defaults to zeros.

        Returns:
            CondUNet2DOutput containing the processed tensor.
        """
        squeeze = False
        if x.ndim < 4:
            x = jnp.expand_dims(x, 0)
            squeeze = True

        h = x

        if t is None:
            t = jnp.zeros((x.shape[0], 1)).astype(self.dtype)

        t_emb = self.time_proj(t)
        t_emb = self.time_embedding(t_emb)

        hs = []
        for block in self.blocks_down:
            h = block(h, t_emb)
            hs.append(h)

        h = self.block_mid(h, t_emb)

        for block in self.blocks_up:
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = block(h, t_emb)

        h = self.out(h)

        if squeeze:
            h = h.squeeze(axis=0)

        return CondUNet2DOutput(output=h)
