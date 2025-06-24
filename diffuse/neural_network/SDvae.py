from dataclasses import dataclass
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
from chex import Array
from einops import rearrange
from jax.typing import DTypeLike

from diffuse.neural_network.decoder import Decoder
from diffuse.neural_network.encoder import Encoder


@dataclass
class AutoEncoderParams:
    """Configuration parameters for the AutoEncoder model.

    This dataclass defines all the hyperparameters needed to configure
    an AutoEncoder with encoder and decoder components.

    Attributes:
        in_channels: Number of input channels for the encoder
        ch: Base number of channels in the model
        out_ch: Number of output channels from the decoder
        ch_mult: List of channel multipliers for different resolution levels
        num_res_blocks: Number of residual blocks per resolution level
        z_channels: Number of channels in the latent space
        scale_factor: Scaling factor applied to the latent representation (default: 0.18215)
        shift_factor: Shift factor applied to the latent representation (default: 0.0)
        activation: Activation function to use throughout the model (default: nn.swish)
        training: Whether the model is in training mode (default: True)
        param_dtype: Data type for model parameters (default: jnp.bfloat16)
    """

    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = [1, 2, 4]
    num_res_blocks: int = 2
    z_channels: int = 8
    scale_factor: float = 0.18215
    shift_factor: float = 0.0
    activation: Callable = nn.swish
    training: bool = True
    param_dtype: DTypeLike = jnp.bfloat16


class DiagonalGaussian(nn.Module):
    sample: bool = True
    chunk_dim: int = -1

    def __call__(self, z: Array) -> Array:
        mean, logvar = jnp.split(z, 2, axis=self.chunk_dim)
        if self.sample:
            std = jnp.exp(0.5 * logvar)
            kl = 0.5 * jnp.sum(mean**2 + jnp.exp(logvar) - logvar - 1, axis=[1, 2, 3])
            return kl, mean + std * jax.random.normal(
                key=self.make_rng("reg"), shape=mean.shape, dtype=z.dtype
            )
        else:
            return mean


class AutoEncoder(nn.Module):
    params: AutoEncoderParams

    def setup(self) -> None:
        self.encoder = Encoder(
            resolution=self.params.resolution,
            in_channels=self.params.in_channels,
            ch=self.params.ch,
            ch_mult=self.params.ch_mult,
            num_res_blocks=self.params.num_res_blocks,
            z_channels=self.params.z_channels,
            activation=self.params.activation,
            deterministic=True,  # Never activate dropout for VAE training
            param_dtype=self.params.param_dtype,
        )

        self.decoder = Decoder(
            ch=self.params.ch,
            out_ch=self.params.out_ch,
            ch_mult=self.params.ch_mult,
            num_res_blocks=self.params.num_res_blocks,
            in_channels=self.params.in_channels,
            resolution=self.params.resolution,
            z_channels=self.params.z_channels,
            activation=self.params.activation,
            deterministic=True,  # Never activate dropout for VAE training
            param_dtype=self.params.param_dtype,
        )

        self.reg = DiagonalGaussian(sample=self.params.training)

        self.scale_factor = self.params.scale_factor
        self.shift_factor = self.params.shift_factor

    def encode(self, x: Array) -> Union[Array, Tuple[Array, Array]]:
        x = rearrange(x, "b c h w -> b h w c")

        z = self.encoder(x)
        if self.params.training:
            kl, z = self.reg(z)
        else:
            z = self.reg(z)

        z = self.scale_factor * (z - self.shift_factor)

        z = rearrange(z, "b h w c -> b c h w")
        if self.params.training:
            return z, kl

        return z

    def decode(self, z: Array) -> Array:
        z = rearrange(z, "b c h w -> b h w c")

        z = z / self.scale_factor + self.shift_factor
        z = self.decoder(z)

        z = rearrange(z, "b h w c -> b c h w")
        return z

    def __call__(self, x: Array) -> Array:
        if self.params.training:
            z, kl = self.encode(x)
            x_recon = self.decode(z)
            return x_recon, kl
        else:
            z = self.encode(x)
            x_recon = self.decode(z)
            return x_recon
