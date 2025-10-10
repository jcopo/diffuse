# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from typing import Callable

import jax.numpy as jnp

from flax import nnx
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .attention import AttnBlock
from .downsample import Downsample
from .resnet_block import ResnetBlock


class Encoder(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        activation: Callable = nnx.swish,
        dropout: bool = False,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.in_channels = in_channels
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels
        self.activation = activation
        self.dropout = dropout
        self.param_dtype = param_dtype

        num_resolutions = len(self.ch_mult)

        self.conv_in = nnx.Conv(
            in_features=self.in_channels,
            out_features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=self.param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        in_ch_mult = (1,) + tuple(self.ch_mult)
        blocks_down = []
        block_in = self.ch
        for i_level in range(num_resolutions):
            block = []
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * self.ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        activation=self.activation,
                        param_dtype=self.param_dtype,
                        dtype=dtype,
                        dropout=self.dropout,
                        rngs=rngs,
                    )
                )
                block_in = block_out

            if i_level != num_resolutions - 1:
                block.append(
                    Downsample(
                        in_channels=block_in,
                        param_dtype=self.param_dtype,
                        dtype=dtype,
                        rngs=rngs,
                    )
                )
            blocks_down += block

        self.down = nnx.Sequential(*blocks_down)

        res_block_mid_in = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            activation=self.activation,
            param_dtype=self.param_dtype,
            dtype=dtype,
            dropout=self.dropout,
            rngs=rngs,
        )
        mid_attn = AttnBlock(
            in_channels=block_in,
            param_dtype=self.param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        res_block_mid_out = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            activation=self.activation,
            param_dtype=self.param_dtype,
            dtype=dtype,
            dropout=self.dropout,
            rngs=rngs,
        )

        self.mid = nnx.Sequential(res_block_mid_in, mid_attn, res_block_mid_out)

        self.norm_out = nnx.GroupNorm(
            num_features=block_in,
            num_groups=32,
            epsilon=1e-6,
            param_dtype=self.param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        self.conv_out = nnx.Conv(
            in_features=block_in,
            out_features=2 * self.z_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=self.param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike) -> Array:
        h = self.conv_in(x)
        h = self.down(h)
        h = self.mid(h)
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        return h
