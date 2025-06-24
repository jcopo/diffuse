from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
from chex import Array
from jax.typing import DTypeLike

from diffuse.neural_network.block import AttnBlock, ResnetBlock, Upsample


class Decoder(nn.Module):
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    in_channels: int
    z_channels: int
    activation: Callable = nn.swish
    deterministic: bool = True
    param_dtype: DTypeLike = jnp.bfloat16

    def setup(self) -> None:
        self.conv_in = nn.Conv(
            features=self.z_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=self.param_dtype,
        )

        num_resolutions = len(self.ch_mult)

        block_in = self.ch * self.ch_mult[num_resolutions - 1]

        conv_z_in = nn.Conv(
            features=block_in,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=self.param_dtype,
        )

        res_block_mid_in = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            activation=self.activation,
            deterministic=self.deterministic,
            param_dtype=self.param_dtype,
        )

        mid_attn = AttnBlock(
            in_channels=block_in,
            param_dtype=self.param_dtype,
        )

        res_block_mid_out = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            activation=self.activation,
            deterministic=self.deterministic,
            param_dtype=self.param_dtype,
        )

        self.mid = nn.Sequential(
            [conv_z_in, res_block_mid_in, mid_attn, res_block_mid_out]
        )

        blocks_up = []
        for i_level in reversed(range(num_resolutions)):
            block = []
            block_out = self.ch * self.ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        activation=self.activation,
                        deterministic=self.deterministic,
                        param_dtype=self.param_dtype,
                    )
                )
                block_in = block_out

            if i_level != 0:
                block.append(
                    Upsample(
                        in_channels=block_in,
                        method="resize",
                        scale_factor=2,
                        param_dtype=self.param_dtype,
                    )
                )
            blocks_up += block

        self.up = nn.Sequential(blocks_up)

        self.norm_out = nn.GroupNorm(
            num_groups=32,
            epsilon=1e-6,
            param_dtype=self.param_dtype,
        )

        self.conv_out = nn.Conv(
            features=self.out_ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=self.param_dtype,
        )

    def __call__(self, z: Array) -> Array:
        h = self.mid(z)
        h = self.up(h)
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        return h
