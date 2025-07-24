import jax.numpy as jnp

from einops import rearrange
from flax import nnx
from jax import Array
from jax.typing import ArrayLike, DTypeLike


class AttnBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.dtype = dtype

        self.norm = nnx.GroupNorm(
            num_features=in_channels,
            num_groups=32,
            epsilon=1e-6,
            param_dtype=param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

        self.q = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=(0, 0),
            param_dtype=param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.k = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=(0, 0),
            param_dtype=param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.v = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=(0, 0),
            param_dtype=param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.proj_out = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            param_dtype=param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

    def attention(self, h_: ArrayLike) -> Array:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, h, w, c = q.shape
        q = rearrange(q, "b h w c-> b (h w) 1 c")
        k = rearrange(k, "b h w c-> b (h w) 1 c")
        v = rearrange(v, "b h w c-> b (h w) 1 c")

        h_ = nnx.dot_product_attention(q, k, v, dtype=self.dtype)

        return rearrange(h_, "b (h w) 1 c -> b h w c", h=h, w=w, c=c, b=b)

    def __call__(self, x: ArrayLike) -> Array:
        return x + self.proj_out(self.attention(x))
