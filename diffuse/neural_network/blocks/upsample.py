import jax
import jax.numpy as jnp

from einops import rearrange
from flax import nnx
from jax import Array
from jax.typing import ArrayLike, DTypeLike


class PixelShuffle(nnx.Module):
    def __init__(self, scale: int):
        self.scale = scale

    def __call__(self, x: ArrayLike) -> Array:
        return rearrange(x, "b h w (h2 w2 c) -> b (h h2) (w w2) c", h2=self.scale, w2=self.scale)


class Upsample(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        method: str = "resize",
        scale_factor: float = 2.0,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.method = method
        self.scale_factor = scale_factor

        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        self.conv_pixel_shuffle = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels * self.scale_factor**2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.pixel_shuffle = PixelShuffle(scale=self.scale_factor)

    def __call__(self, x: ArrayLike) -> Array:
        if self.method == "resize":
            b, h, w, c = x.shape
            new_height = int(h * self.scale_factor)
            new_width = int(w * self.scale_factor)
            new_shape = (b, new_height, new_width, c)
            x = jax.image.resize(x, new_shape, method="nearest")
        elif self.method == "pixel_shuffle":
            x = self.conv_pixel_shuffle(self.conv(x))
            x = self.pixel_shuffle(x)
        else:
            raise ValueError(f"Invalid method: {self.method}")

        return self.conv(x)
