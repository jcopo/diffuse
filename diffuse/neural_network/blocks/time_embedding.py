# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Timestep embedding modules for diffusion models.

This module provides sinusoidal positional encodings and MLP-based timestep
embeddings commonly used for conditioning diffusion model predictions on the
current noise level or timestep.
"""

from typing import Callable, Optional

import jax.numpy as jnp

from flax import nnx
from jax import Array
from jax.typing import ArrayLike, DTypeLike


def get_sinusoidal_embedding(t: ArrayLike, embedding_dim: int = 64, max_period: int = 10_000) -> Array:
    """Generate sinusoidal positional embeddings for timesteps.

    Creates embeddings using sine and cosine functions at different frequencies,
    similar to the positional encodings in "Attention is All You Need".

    Args:
        t: Timestep values to encode
        embedding_dim: Dimension of the output embedding
        max_period: Maximum period for the sinusoidal functions

    Returns:
        Sinusoidal embeddings of shape (batch, embedding_dim)
    """
    half = embedding_dim // 2
    fs = jnp.exp(-jnp.log(max_period) * jnp.arange(half) / (half - 1))
    t = t.reshape(-1)
    embs = jnp.einsum("b,c->bc", t, fs)
    embs = jnp.concatenate([jnp.sin(embs), jnp.cos(embs)], axis=-1)
    return embs


class Timesteps(nnx.Module):
    """Sinusoidal timestep embedding layer.

    Converts scalar timestep values into high-dimensional sinusoidal embeddings
    for use as conditioning signals in diffusion models.

    Args:
        embedding_dim: Dimension of the output embedding
        max_period: Maximum period for the sinusoidal functions
        dtype: Data type for the output embeddings
    """

    def __init__(self, embedding_dim: int, max_period: int = 10_000, dtype: Optional[DTypeLike] = None):
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        self.dtype = dtype

    def __call__(self, t: ArrayLike) -> Array:
        """Convert timesteps to sinusoidal embeddings.

        Args:
            t: Timestep values

        Returns:
            Sinusoidal embeddings of shape (batch, embedding_dim)
        """
        return get_sinusoidal_embedding(t, self.embedding_dim, self.max_period).astype(self.dtype)


class TimestepEmbedding(nnx.Module):
    """MLP-based timestep embedding processor.

    Transforms sinusoidal timestep embeddings through a two-layer MLP with
    an activation function. Commonly used to increase expressiveness of the
    timestep conditioning signal.

    Args:
        embedding_dim: Dimension of input and output embeddings
        activation: Activation function to use between layers
        param_dtype: Data type for parameters
        dtype: Data type for computation
        rngs: Random number generators for parameter initialization
    """

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
        """Process timestep embeddings through MLP.

        Args:
            temb: Input timestep embeddings

        Returns:
            Processed timestep embeddings of the same shape
        """
        temb = self.linear1(temb)
        temb = self.activation(temb)
        temb = self.linear2(temb)
        return temb
