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
        x = rearrange(x, "b c h w -> b h w c")

        z = self.encoder(x)
        mean, logvar, z = self.reg(z)

        z = self.scale_factor * (z - self.shift_factor)

        z = rearrange(z, "b h w c -> b c h w")
        mean = rearrange(mean, "b h w c -> b c h w")
        logvar = rearrange(logvar, "b h w c -> b c h w")
        return z, mean, logvar

    def decode(self, z: ArrayLike) -> Array:
        z = rearrange(z, "b c h w -> b h w c")

        z = z / self.scale_factor + self.shift_factor
        z = self.decoder(z)

        z = rearrange(z, "b h w c -> b c h w")
        return z

    def __call__(self, x: ArrayLike) -> SDVaeOutput:
        z, mean, logvar = self.encode(x)
        x_recon = self.decode(z)
        return SDVaeOutput(output=x_recon, mean=mean, logvar=logvar)
