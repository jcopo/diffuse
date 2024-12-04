"""
This script is adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl.py#L37
under the Apache-2.0 license.
"""

import flax.linen as nn
import jax.numpy as jnp
import jax

from typing import Tuple

from diffuse.neural_network.encoder import FlaxResnetBlock2D, FlaxUNetMidBlock2D


class FlaxUpsample2D(nn.Module):
    """
    Flax implementation of 2D Upsample layer

    Args:
        in_channels (`int`):
            Input channels
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    in_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        hidden_states = self.conv(hidden_states)
        return hidden_states


class FlaxUpDecoderBlock2D(nn.Module):
    r"""
    Flax Resnet blocks-based Decoder block for diffusion-based VAE.

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of Resnet layer block
        resnet_groups (:obj:`int`, *optional*, defaults to `32`):
            The number of groups to use for the Resnet block group norm
        add_upsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add upsample layer
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    resnet_groups: int = 32
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets = []
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels
            res_block = FlaxResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout=self.dropout,
                groups=self.resnet_groups,
                dtype=self.dtype,
            )
            resnets.append(res_block)

        self.resnets = resnets

        if self.add_upsample:
            self.upsamplers_0 = FlaxUpsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, deterministic=deterministic)

        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)

        return hidden_states


class Decoder(nn.Module):
    r"""
    Flax Implementation of VAE Decoder.

    This model is a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
    general usage and behavior.

    Finally, this model supports inherent JAX features such as:
    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        in_channels (:obj:`int`, *optional*, defaults to 3):
            Input channels
        out_channels (:obj:`int`, *optional*, defaults to 3):
            Output channels
        up_block_types (:obj:`Tuple[str]`, *optional*, defaults to `(UpDecoderBlock2D)`):
            UpDecoder block type
        block_out_channels (:obj:`Tuple[str]`, *optional*, defaults to `(64,)`):
            Tuple containing the number of output channels for each block
        layers_per_block (:obj:`int`, *optional*, defaults to `2`):
            Number of Resnet layer for each block
        norm_num_groups (:obj:`int`, *optional*, defaults to `32`):
            norm num group
        act_fn (:obj:`str`, *optional*, defaults to `silu`):
            Activation function
        double_z (:obj:`bool`, *optional*, defaults to `False`):
            Whether to double the last output channels
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            parameters `dtype`
    """

    in_channels: int = 3
    out_channels: int = 3
    up_block_types: Tuple[str] = ("UpDecoderBlock2D",)
    block_out_channels: int = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        block_out_channels = self.block_out_channels

        # z to block_in
        self.conv_in = nn.Conv(
            block_out_channels[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # middle
        self.mid_block = FlaxUNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_groups=self.norm_num_groups,
            num_attention_heads=None,
            dtype=self.dtype,
        )

        # upsampling
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        up_blocks = []
        for i, _ in enumerate(self.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = FlaxUpDecoderBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block + 1,
                resnet_groups=self.norm_num_groups,
                add_upsample=not is_final_block,
                dtype=self.dtype,
            )
            up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.up_blocks = up_blocks

        # end
        self.conv_norm_out = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-6)
        self.conv_out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, sample, deterministic: bool = True):
        # z to block_in
        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample, deterministic=deterministic)

        # upsampling
        for block in self.up_blocks:
            sample = block(sample, deterministic=deterministic)

        sample = self.conv_norm_out(sample)
        sample = nn.swish(sample)
        sample = self.conv_out(sample)

        return sample
