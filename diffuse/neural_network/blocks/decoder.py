# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from typing import Callable

import jax.numpy as jnp

from flax import nnx
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .attention import AttnBlock
from .resnet_block import ResnetBlock
from .upsample import Upsample


class Decoder(nnx.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        activation: Callable = nnx.swish,
        dropout: bool = False,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.activation = activation
        self.dropout = dropout

        num_resolutions = len(ch_mult)
        block_in = ch * ch_mult[num_resolutions - 1]

        conv_z_in = nnx.Conv(
            in_features=z_channels,
            out_features=block_in,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        res_block_mid_in = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            activation=self.activation,
            param_dtype=param_dtype,
            dtype=dtype,
            dropout=self.dropout,
            rngs=rngs,
        )

        mid_attn = AttnBlock(in_channels=block_in, param_dtype=param_dtype, dtype=dtype, rngs=rngs)

        res_block_mid_out = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            activation=self.activation,
            param_dtype=param_dtype,
            dtype=dtype,
            dropout=self.dropout,
            rngs=rngs,
        )

        self.mid = nnx.Sequential(conv_z_in, res_block_mid_in, mid_attn, res_block_mid_out)

        blocks_up = []
        for i_level in reversed(range(num_resolutions)):
            block = []
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        activation=self.activation,
                        param_dtype=param_dtype,
                        dtype=dtype,
                        dropout=self.dropout,
                        rngs=rngs,
                    )
                )
                block_in = block_out

            if i_level != 0:
                block.append(
                    Upsample(
                        in_channels=block_in,
                        method="resize",
                        scale_factor=2,
                        param_dtype=param_dtype,
                        dtype=dtype,
                        rngs=rngs,
                    )
                )
            blocks_up += block

        self.up = nnx.Sequential(*blocks_up)

        self.norm_out = nnx.GroupNorm(
            num_features=block_in,
            num_groups=32,
            epsilon=1e-6,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        self.conv_out = nnx.Conv(
            in_features=block_in,
            out_features=out_ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, z: ArrayLike) -> Array:
        h = self.mid(z)
        h = self.up(h)
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        return h
