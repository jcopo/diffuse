from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import flax.linen as nn
from chex import Array
from einops import rearrange
from jax.typing import DTypeLike

from diffuse.neural_network.block import (
    AttnBlock,
    Timesteps,
    TimestepEmbedding,
    ResnetBlock,
    Downsample,
    Upsample,
    TimestepEmbedSequential,
)


@dataclass
class UNetParams:
    """Configuration parameters for the UNet model.

    This dataclass defines all the hyperparameters needed to configure
    a UNet model for diffusion processes.

    Attributes:
        in_channels: Number of input channels for the model (default: 3)
        ch: Base number of channels in the model (default: 64)
        ch_mult: List of channel multipliers for different resolution levels (default: [1, 2, 3, 4])
        num_res_blocks: Number of residual blocks per resolution level (default: 2)
        num_head_channels: Number of channels per attention head (default: 32)
        attention_resolutions: List of resolution levels where attention is applied (default: [1, 2, 4, 8])
        deterministic: Whether to use deterministic behavior (affects dropout) (default: True)
        activation: Activation function to use throughout the model (default: nn.swish)
        param_dtype: Data type for model parameters (default: jnp.bfloat16)
    """

    in_channels: int = 3
    ch: int = 64
    ch_mult: list[int] = [1, 2, 3, 4]
    num_res_blocks: int = 2
    attention_resolutions: list[int] = [1, 2, 4, 8]
    deterministic: bool = True
    activation: Callable = nn.swish
    param_dtype: DTypeLike = jnp.bfloat16


class UNet(nn.Module):
    params: UNetParams

    def setup(self) -> None:
        time_embed_dim = self.params.ch * 4

        self.time_proj = Timesteps(
            embedding_dim=time_embed_dim,
            max_period=10_000,
        )
        self.time_embedding = TimestepEmbedding(
            embedding_dim=time_embed_dim,
            activation=self.params.activation,
            param_dtype=self.params.param_dtype,
        )

        ch = int(self.params.ch_mult[0] * self.params.ch)
        conv_in = nn.Conv(
            features=ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=self.params.param_dtype,
        )
        blocks_down = [TimestepEmbedSequential([conv_in])]

        input_block_channels = [ch]
        ds = 1
        for level, mult in enumerate(self.params.ch_mult):
            for _ in range(self.params.num_res_blocks):
                layers = [
                    ResnetBlock(
                        in_channels=ch,
                        out_channels=int(mult * self.params.ch),
                        activation=self.params.activation,
                        time_embedding=True,
                        deterministic=self.params.deterministic,
                        param_dtype=self.params.param_dtype,
                    )
                ]
                ch = int(mult * self.params.ch)
                if ds in self.params.attention_resolutions:
                    layers.append(AttnBlock(in_channels=ch, param_dtype=self.params.param_dtype))

                blocks_down.append(TimestepEmbedSequential(layers))
                input_block_channels.append(ch)

            if level != len(self.params.ch_mult) - 1:
                blocks_down.append(
                    TimestepEmbedSequential([Downsample(in_channels=ch, param_dtype=self.params.param_dtype)])
                )
                input_block_channels.append(ch)
                ds *= 2

        self.blocks_down = blocks_down

        # mid
        self.block_mid = TimestepEmbedSequential(
            [
                ResnetBlock(
                    in_channels=ch,
                    out_channels=ch,
                    activation=self.params.activation,
                    time_embedding=True,
                    deterministic=self.params.deterministic,
                    param_dtype=self.params.param_dtype,
                ),
                AttnBlock(in_channels=ch, param_dtype=self.params.param_dtype),
                ResnetBlock(
                    in_channels=ch,
                    out_channels=ch,
                    activation=self.params.activation,
                    time_embedding=True,
                    deterministic=self.params.deterministic,
                    param_dtype=self.params.param_dtype,
                ),
            ]
        )

        # up
        blocks_up = []
        for level, mult in reversed(list(enumerate(self.params.ch_mult))):
            for i in range(self.params.num_res_blocks + 1):
                ich = input_block_channels.pop()
                layers = [
                    ResnetBlock(
                        in_channels=ch + ich,
                        out_channels=int(mult * self.params.ch),
                        activation=self.params.activation,
                        time_embedding=True,
                        deterministic=self.params.deterministic,
                        param_dtype=self.params.param_dtype,
                    )
                ]

                ch = int(mult * self.params.ch)
                if ds in self.params.attention_resolutions:
                    layers.append(AttnBlock(in_channels=ch, param_dtype=self.params.param_dtype))

                if level and i == self.params.num_res_blocks:
                    layers.append(
                        Upsample(
                            in_channels=ch,
                            method="pixel_shuffle",
                            scale_factor=2,
                            param_dtype=self.params.param_dtype,
                        )
                    )
                    ds //= 2

                blocks_up.append(TimestepEmbedSequential(layers))
        self.blocks_up = blocks_up

        norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-6, param_dtype=self.params.param_dtype)

        activation_out = self.params.activation

        conv_out = nn.Conv(
            features=self.params.in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=self.params.param_dtype,
        )

        self.out = nn.Sequential([norm_out, activation_out, conv_out])

    def __call__(self, x: Array, t: Array) -> Array:
        h = rearrange(x, "b c h w -> b h w c")

        t_emb = self.time_proj(t).astype(self.params.param_dtype)
        t_emb = self.time_embedding(t_emb)

        hs = []
        for block in self.blocks_down:
            print(h.shape)
            h = block(h, t_emb)
            hs.append(h)
        print("*********")
        h = self.block_mid(h, t_emb)
        print(h.shape)
        print("*********")
        for block in self.blocks_up:
            print(h.shape)
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = block(h, t_emb)

        h = self.out(h)

        return rearrange(h, "b h w c -> b c h w")
