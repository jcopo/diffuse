from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
from chex import Array
from jax.typing import DTypeLike

from diffuse.neural_network.block import AttnBlock, Downsample, ResnetBlock


class Encoder(nn.Module):
    in_channels: int
    ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    activation: Callable = nn.swish
    deterministic: bool = True
    param_dtype: DTypeLike = jnp.bfloat16

    def setup(self) -> None:
        num_resolutions = len(self.ch_mult)

        self.conv_in = nn.Conv(
            features=self.ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=self.param_dtype,
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
                        deterministic=self.deterministic,
                        param_dtype=self.param_dtype,
                    )
                )
                block_in = block_out

            if i_level != num_resolutions - 1:
                block.append(
                    Downsample(
                        in_channels=block_in,
                        param_dtype=self.param_dtype,
                    )
                )
            blocks_down += block

        self.down = nn.Sequential(blocks_down)

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

        self.mid = nn.Sequential([res_block_mid_in, mid_attn, res_block_mid_out])

        self.norm_out = nn.GroupNorm(
            num_groups=32,
            epsilon=1e-6,
            param_dtype=self.param_dtype,
        )

        self.conv_out = nn.Conv(
            features=2 * self.z_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=self.param_dtype,
        )

    def __call__(self, x: Array) -> Array:
        h = self.conv_in(x)
        h = self.down(h)
        h = self.mid(h)
        h = self.norm_out(h)
        h = self.activation(h)
        h = self.conv_out(h)
        return h
