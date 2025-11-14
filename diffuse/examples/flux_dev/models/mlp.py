from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.typing import DTypeLike


class Mlp(nnx.Module):
    """MLP block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        *,
        activation: Callable = nnx.gelu,
        norm: Callable = None,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        dropout: bool = True,
        dropout_rate: float = 0.1,
        rngs: nnx.Rngs,
    ):
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nnx.Linear(
            in_features=in_features,
            out_features=hidden_features,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.activation = activation

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs) if dropout else lambda x: x
        self.norm = norm if norm is not None else lambda x: x

        self.fc2 = nnx.Linear(
            in_features=hidden_features,
            out_features=out_features,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
