# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from typing import Tuple, Union

import jax
import jax.numpy as jnp

from einops import rearrange
from flax import nnx
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..blocks import Decoder, Encoder
from .params import SDVaeOutput


class DiagonalGaussian(nnx.Module):
    sample: bool = True
    chunk_dim: int = -1

    def __init__(self, sample: bool = True, chunk_dim: int = -1, rngs: nnx.Rngs = None, dtype: DTypeLike = jnp.float32):
        self.sample = sample
        self.chunk_dim = chunk_dim
        self.rngs = rngs
        self.dtype = dtype

    def __call__(self, z: ArrayLike) -> Array:
        mean, logvar = jnp.split(z, 2, axis=self.chunk_dim)
        if self.sample:
            std = jnp.exp(0.5 * logvar)
            return (
                mean,
                logvar,
                mean + std * jax.random.normal(key=self.rngs, shape=mean.shape, dtype=self.dtype),
            )
        else:
            return mean


class SDVae(nnx.Module):
    """Stable Diffusion Variational Autoencoder (VAE).

    A VAE architecture for encoding images into latent representations and decoding
    them back. Uses an encoder-decoder structure with diagonal Gaussian posterior,
    commonly used in latent diffusion models for compression.

    Args:
        in_channels: Number of input image channels
        ch: Base number of channels in the network
        out_ch: Number of output channels
        ch_mult: Channel multipliers for each resolution level
        num_res_blocks: Number of ResNet blocks at each resolution
        z_channels: Number of latent space channels
        scale_factor: Scaling factor applied to latent codes (default: 0.18215 for SD)
        shift_factor: Shift applied to latent codes before scaling
        activation: Activation function used throughout the network
        param_dtype: Data type for parameters
        dtype: Data type for computation
        rngs: Random number generators for parameter initialization
    """

    def __init__(
        self,
        in_channels: int = 3,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        z_channels: int = 8,
        scale_factor: float = 0.18215,
        shift_factor: float = 0.0,
        activation=nnx.swish,
        param_dtype=jnp.float32,
        dtype=jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.param_dtype = param_dtype
        self.encoder = Encoder(
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            activation=activation,
            dropout=False,  # Never activate dropout for VAE
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        self.decoder = Decoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            activation=activation,
            dropout=False,  # Never activate dropout for VAE
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        rng_noise = getattr(rngs, "noise", rngs)
        self.reg = DiagonalGaussian(rngs=rng_noise(), dtype=dtype)

        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

    def encode(self, x: ArrayLike) -> Union[Array, Tuple[Array, Array]]:
        """Encode image into latent representation.

        Args:
            x: Input image tensor of shape (batch, channels, height, width)

        Returns:
            Tuple of (latent_code, mean, logvar) where latent_code is the sampled
            latent representation and mean/logvar define the diagonal Gaussian posterior
        """
        x = rearrange(x, "b c h w -> b h w c")

        z = self.encoder(x)
        mean, logvar, z = self.reg(z)

        z = self.scale_factor * (z - self.shift_factor)

        z = rearrange(z, "b h w c -> b c h w")
        mean = rearrange(mean, "b h w c -> b c h w")
        logvar = rearrange(logvar, "b h w c -> b c h w")
        return z, mean, logvar

    def decode(self, z: ArrayLike) -> Array:
        """Decode latent representation back to image space.

        Args:
            z: Latent code tensor of shape (batch, z_channels, latent_h, latent_w)

        Returns:
            Reconstructed image tensor of shape (batch, out_ch, height, width)
        """
        z = rearrange(z, "b c h w -> b h w c")

        z = z / self.scale_factor + self.shift_factor
        z = self.decoder(z)

        z = rearrange(z, "b h w c -> b c h w")
        return z

    def __call__(self, x: ArrayLike) -> SDVaeOutput:
        """Full forward pass: encode and decode.

        Args:
            x: Input image tensor of shape (batch, channels, height, width)

        Returns:
            SDVaeOutput containing reconstructed image, mean, and log variance
        """
        z, mean, logvar = self.encode(x)
        x_recon = self.decode(z)
        return SDVaeOutput(output=x_recon, mean=mean, logvar=logvar)
