from typing import Callable, Optional

import jax.numpy as jnp

from einops import rearrange
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
    def __init__(
        self,
        in_channels: int = 3,
        ch: int = 64,
        ch_mult: tuple[int, ...] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (1, 2, 4, 8),
        activation: Callable = nnx.swish,
        dropout: bool = True,
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
                        rngs=rngs,
                    )
                ]
                current_ch = int(mult * ch)
                if ds in attention_resolutions:
                    layers.append(
                        AttnBlock(
                            in_channels=current_ch,
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
                rngs=rngs,
            ),
            AttnBlock(in_channels=current_ch, param_dtype=param_dtype, dtype=dtype, rngs=rngs),
            ResnetBlock(
                in_channels=current_ch,
                out_channels=current_ch,
                activation=activation,
                embedding_dim=time_embed_dim,
                param_dtype=self.param_dtype,
                dtype=dtype,
                dropout=dropout,
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
                        rngs=rngs,
                    )
                ]

                current_ch = int(mult * ch)
                if ds in attention_resolutions:
                    layers.append(
                        AttnBlock(
                            in_channels=current_ch,
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

    def __call__(self, x: ArrayLike, t: Optional[ArrayLike] = None) -> CondUNet2DOutput:
        squeeze = False
        if x.ndim < 4:
            x = jnp.expand_dims(x, 0)
            squeeze = True

        h = rearrange(x, "b c h w -> b h w c")

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
        h = rearrange(h, "b h w c -> b c h w")

        if squeeze:
            h = h.squeeze(axis=0)

        return CondUNet2DOutput(output=h)
