from typing import Callable, Optional

import jax.numpy as jnp

from flax import nnx
from jax import Array
from jax.typing import ArrayLike, DTypeLike


def get_sinusoidal_embedding(t: ArrayLike, embedding_dim: int = 64, max_period: int = 10_000) -> Array:
    half = embedding_dim // 2
    fs = jnp.exp(-jnp.log(max_period) * jnp.arange(half) / (half - 1))
    t = t.reshape(-1)
    embs = jnp.einsum("b,c->bc", t, fs)
    embs = jnp.concatenate([jnp.sin(embs), jnp.cos(embs)], axis=-1)
    return embs


class Timesteps(nnx.Module):
    def __init__(self, embedding_dim: int, max_period: int = 10_000, dtype: Optional[DTypeLike] = None):
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        self.dtype = dtype

    def __call__(self, t: ArrayLike) -> Array:
        return get_sinusoidal_embedding(t, self.embedding_dim, self.max_period).astype(self.dtype)


class TimestepEmbedding(nnx.Module):
    def __init__(
        self,
        embedding_dim: int,
        activation: Callable = nnx.swish,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.activation = activation
        self.linear1 = nnx.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, temb: ArrayLike) -> Array:
        temb = self.linear1(temb)
        temb = self.activation(temb)
        temb = self.linear2(temb)
        return temb
