from typing import Callable, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from chex import Array
from einops import rearrange
from jax.typing import DTypeLike


def get_sinusoidal_embedding(t: Array, embedding_dim: int = 64, max_period: int = 10_000) -> Array:
    half = embedding_dim // 2
    fs = jnp.exp(-jnp.log(max_period) * jnp.arange(half) / (half - 1))
    embs = jnp.einsum("b,c->bc", t, fs)
    embs = jnp.concatenate([jnp.sin(embs), jnp.cos(embs)], axis=-1)
    return embs


class Timesteps(nn.Module):
    embedding_dim: int
    max_period: int = 10_000

    @nn.compact
    def __call__(self, t: Array) -> Array:
        return get_sinusoidal_embedding(t, self.embedding_dim, self.max_period)


class TimestepEmbedding(nn.Module):
    embedding_dim: int
    activation: Callable = nn.swish
    param_dtype: DTypeLike = jnp.bfloat16

    @nn.compact
    def __call__(self, temb: Array) -> Array:
        temb = nn.Dense(self.embedding_dim, param_dtype=self.param_dtype)(temb)
        temb = self.activation(temb)
        temb = nn.Dense(self.embedding_dim, param_dtype=self.param_dtype)(temb)
        return temb


class TimestepBlock(nn.Module):
    def __call__(self, x: Array, time_emb: Optional[Array] = None) -> Array:
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def __call__(self, x: Array, time_emb: Optional[Array] = None) -> Array:
        for layer in self.layers:
            if isinstance(layer, TimestepBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        return x


class ResnetBlock(TimestepBlock):
    in_channels: int
    out_channels: Optional[int] = None
    activation: Callable = nn.swish
    time_embedding: bool = False
    deterministic: bool = True
    param_dtype: DTypeLike = jnp.bfloat16

    def setup(self) -> None:
        actual_out_channels = self.in_channels if self.out_channels is None else self.out_channels

        self.norm1 = nn.GroupNorm(
            num_groups=32,
            epsilon=1e-6,
            param_dtype=self.param_dtype,
        )
        self.conv1 = nn.Conv(
            features=actual_out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=self.param_dtype,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32,
            epsilon=1e-6,
            param_dtype=self.param_dtype,
        )
        self.conv2 = nn.Conv(
            features=actual_out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=self.param_dtype,
        )
        if self.in_channels != actual_out_channels:
            self.nin_shortcut = nn.Conv(
                features=actual_out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=(0, 0),
                param_dtype=self.param_dtype,
            )

        self.dropout = nn.Dropout(rate=0.2, deterministic=self.deterministic)
        if self.time_embedding:
            self.time_mlp = nn.Dense(
                features=actual_out_channels * 2,
                param_dtype=self.param_dtype,
            )

    def __call__(self, x: Array, time_emb: Optional[Array] = None) -> Array:
        h = x
        h = self.norm1(h)
        h = self.activation(h)
        h = self.conv1(h)

        if self.time_embedding and time_emb is not None:
            time_emb = self.time_mlp(self.activation(time_emb))
            time_emb = rearrange(time_emb, "b c -> b 1 1 c")
            scale, shift = jnp.split(time_emb, 2, axis=-1)
            h = h * (scale + 1) + shift

        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        actual_out_channels = self.in_channels if self.out_channels is None else self.out_channels
        if self.in_channels != actual_out_channels:
            x = self.nin_shortcut(x)

        h = h + x

        return h


class AttnBlock(nn.Module):
    in_channels: int
    param_dtype: DTypeLike = jnp.bfloat16

    def setup(self) -> None:
        self.norm = nn.GroupNorm(
            num_groups=32,
            epsilon=1e-6,
            param_dtype=self.param_dtype,
        )

        self.q = nn.Conv(
            features=self.in_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=(0, 0),
            param_dtype=self.param_dtype,
        )
        self.k = nn.Conv(
            features=self.in_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=(0, 0),
            param_dtype=self.param_dtype,
        )
        self.v = nn.Conv(
            features=self.in_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=(0, 0),
            param_dtype=self.param_dtype,
        )
        self.proj_out = nn.Conv(
            features=self.in_channels,
            kernel_size=(1, 1),
            param_dtype=self.param_dtype,
        )

    def attention(self, h_: Array) -> Array:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, h, w, c = q.shape
        q = rearrange(q, "b h w c-> b (h w) 1 c")
        k = rearrange(k, "b h w c-> b (h w) 1 c")
        v = rearrange(v, "b h w c-> b (h w) 1 c")

        h_ = nn.dot_product_attention(q, k, v)

        return rearrange(h_, "b (h w) 1 c -> b h w c", h=h, w=w, c=c, b=b)

    def __call__(self, x: Array, *args) -> Array:
        return x + self.proj_out(self.attention(x))


class Downsample(nn.Module):
    in_channels: int
    param_dtype: DTypeLike = jnp.bfloat16

    def setup(self) -> None:
        self.conv = nn.Conv(
            features=self.in_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=(0, 0),
            param_dtype=self.param_dtype,
        )

    def __call__(self, x: Array) -> Array:
        pad_width = ((0, 0), (0, 1), (0, 1), (0, 0))
        x = jnp.pad(array=x, pad_width=pad_width, mode="constant", constant_values=0)
        return self.conv(x)


class PixelShuffle(nn.Module):
    scale: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return rearrange(x, "b h w (h2 w2 c) -> b (h h2) (w w2) c", h2=self.scale, w2=self.scale)


class Upsample(nn.Module):
    in_channels: int
    method: str = "resize"
    scale_factor: float = 2.0
    param_dtype: DTypeLike = jnp.bfloat16

    def setup(self) -> None:
        self.conv = nn.Conv(
            features=self.in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=self.param_dtype,
        )

        self.conv_pixel_shuffle = nn.Conv(
            features=self.in_channels * self.scale_factor**2,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=self.param_dtype,
        )
        self.pixel_shuffle = PixelShuffle(scale=self.scale_factor)

    def __call__(self, x: Array) -> Array:
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
