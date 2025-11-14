"""Minimal T5 encoder implementation backed by Flax NNX."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


def gelu_new(x: jax.Array) -> jax.Array:
    """Approximation used by T5."""
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * (x**3))))


ACTIVATIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "gelu_new": gelu_new,
}


@dataclass
class T5EncoderConfig:
    vocab_size: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    d_kv: int
    layer_norm_epsilon: float = 1e-6
    dropout_rate: float = 0.0
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    is_gated_act: bool = True
    feed_forward_proj: str = "gated-gelu"
    dense_act_fn: str = "gelu_new"
    initializer_factor: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> T5EncoderConfig:
        return cls(
            vocab_size=data["vocab_size"],
            d_model=data["d_model"],
            d_ff=data["d_ff"],
            num_layers=data["num_layers"],
            num_heads=data["num_heads"],
            d_kv=data["d_kv"],
            layer_norm_epsilon=data.get("layer_norm_epsilon", 1e-6),
            dropout_rate=data.get("dropout_rate", 0.0),
            relative_attention_num_buckets=data.get("relative_attention_num_buckets", 32),
            relative_attention_max_distance=data.get("relative_attention_max_distance", 128),
            is_gated_act=data.get("is_gated_act", True),
            feed_forward_proj=data.get("feed_forward_proj", "gated-gelu"),
            dense_act_fn=data.get("dense_act_fn", "gelu_new"),
            initializer_factor=data.get("initializer_factor", 1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_kv": self.d_kv,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "dropout_rate": self.dropout_rate,
            "relative_attention_num_buckets": self.relative_attention_num_buckets,
            "relative_attention_max_distance": self.relative_attention_max_distance,
            "is_gated_act": self.is_gated_act,
            "feed_forward_proj": self.feed_forward_proj,
            "dense_act_fn": self.dense_act_fn,
            "initializer_factor": self.initializer_factor,
        }


class T5LayerNorm(nnx.Module):
    def __init__(self, hidden_size: int, *, eps: float, param_dtype: jnp.dtype, dtype: jnp.dtype, rngs: nnx.Rngs):
        self.eps = eps
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.scale = nnx.Param(jnp.ones((hidden_size,), dtype=param_dtype))

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.eps)
        return hidden_states.astype(self.dtype) * self.scale.value.astype(self.dtype)


class T5DenseReluDense(nnx.Module):
    def __init__(
        self,
        config: T5EncoderConfig,
        *,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        wi_init = jax.nn.initializers.normal(config.initializer_factor * (config.d_model**-0.5))
        wo_init = jax.nn.initializers.normal(config.initializer_factor * (config.d_ff**-0.5))
        self.wi = nnx.Linear(
            in_features=config.d_model,
            out_features=config.d_ff,
            use_bias=False,
            param_dtype=param_dtype,
            dtype=dtype,
            kernel_init=wi_init,
            rngs=rngs,
        )
        self.wo = nnx.Linear(
            in_features=config.d_ff,
            out_features=config.d_model,
            use_bias=False,
            param_dtype=param_dtype,
            dtype=dtype,
            kernel_init=wo_init,
            rngs=rngs,
        )
        self.act = ACTIVATIONS[self.config.dense_act_fn]

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        x = self.wi(hidden_states)
        x = self.act(x)
        x = self.wo(x)
        return x


class T5DenseGatedActDense(nnx.Module):
    def __init__(
        self,
        config: T5EncoderConfig,
        *,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        wi_init = jax.nn.initializers.normal(config.initializer_factor * (config.d_model**-0.5))
        wo_init = jax.nn.initializers.normal(config.initializer_factor * (config.d_ff**-0.5))
        self.wi_0 = nnx.Linear(
            in_features=config.d_model,
            out_features=config.d_ff,
            use_bias=False,
            param_dtype=param_dtype,
            dtype=dtype,
            kernel_init=wi_init,
            rngs=rngs,
        )
        self.wi_1 = nnx.Linear(
            in_features=config.d_model,
            out_features=config.d_ff,
            use_bias=False,
            param_dtype=param_dtype,
            dtype=dtype,
            kernel_init=wi_init,
            rngs=rngs,
        )
        self.wo = nnx.Linear(
            in_features=config.d_ff,
            out_features=config.d_model,
            use_bias=False,
            param_dtype=param_dtype,
            dtype=dtype,
            kernel_init=wo_init,
            rngs=rngs,
        )
        self.act = ACTIVATIONS[self.config.dense_act_fn]

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5FeedForward(nnx.Module):
    def __init__(
        self,
        config: T5EncoderConfig,
        *,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        if config.is_gated_act or config.feed_forward_proj.startswith("gated"):
            self.DenseReluDense = T5DenseGatedActDense(config, param_dtype=param_dtype, dtype=dtype, rngs=rngs)
        else:
            self.DenseReluDense = T5DenseReluDense(config, param_dtype=param_dtype, dtype=dtype, rngs=rngs)

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        return self.DenseReluDense(hidden_states)


class T5Attention(nnx.Module):
    def __init__(
        self,
        config: T5EncoderConfig,
        *,
        has_relative_attention_bias: bool,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_heads = config.num_heads
        self.key_value_proj_dim = config.d_kv
        self.inner_dim = self.num_heads * self.key_value_proj_dim
        self.dtype = dtype
        q_init = jax.nn.initializers.normal(config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5))
        kv_init = jax.nn.initializers.normal(config.initializer_factor * (self.inner_dim**-0.5))
        o_init = jax.nn.initializers.normal(config.initializer_factor * (self.inner_dim**-0.5))

        self.q = nnx.Linear(
            config.d_model,
            self.inner_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=q_init,
            rngs=rngs,
        )
        self.k = nnx.Linear(
            config.d_model,
            self.inner_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kv_init,
            rngs=rngs,
        )
        self.v = nnx.Linear(
            config.d_model,
            self.inner_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kv_init,
            rngs=rngs,
        )
        self.o = nnx.Linear(
            self.inner_dim,
            config.d_model,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=o_init,
            rngs=rngs,
        )

        if has_relative_attention_bias:
            self.relative_attention_bias = nnx.Embed(
                num_embeddings=config.relative_attention_num_buckets,
                features=config.num_heads,
                param_dtype=param_dtype,
                dtype=dtype,
                rngs=rngs,
            )

    @staticmethod
    def _relative_position_bucket(relative_position: jax.Array, bidirectional: bool, num_buckets: int, max_distance: int) -> jax.Array:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            relative_position = -jnp.clip(relative_position, a_max=0)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        )
        relative_position_if_large = jnp.minimum(relative_position_if_large, num_buckets - 1)

        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets.astype("i4")

    def _compute_bias(self, seq_len: int) -> jax.Array:
        context_position = jnp.arange(seq_len, dtype="i4")[:, None]
        memory_position = jnp.arange(seq_len, dtype="i4")[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.config.relative_attention_num_buckets,
            max_distance=self.config.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        values = values.transpose((2, 0, 1))[None, :, :, :]
        return values

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        batch_size, seq_length, _ = hidden_states.shape

        def shape(x):
            x = x.reshape(batch_size, seq_length, self.num_heads, self.key_value_proj_dim)
            return x

        query = shape(self.q(hidden_states))
        key = shape(self.k(hidden_states))
        value = shape(self.v(hidden_states))

        query = query / jnp.sqrt(self.key_value_proj_dim).astype(query.dtype)
        scores = jnp.einsum("bthd,bshd->bhts", query, key)

        if self.has_relative_attention_bias:
            position_bias = self._compute_bias(seq_length).astype(scores.dtype)
            scores = scores + position_bias

        if attention_mask is not None:
            mask = attention_mask[:, None, None, :]
            scores = scores + (1.0 - mask) * jnp.finfo(scores.dtype).min

        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = jnp.einsum("bhts,bshd->bthd", attn_weights, value)
        attn_output = attn_output.reshape(batch_size, seq_length, self.inner_dim)
        attn_output = self.o(attn_output)
        return attn_output


class T5Block(nnx.Module):
    def __init__(
        self,
        config: T5EncoderConfig,
        *,
        has_relative_attention_bias: bool,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.self_attn_layer_norm = T5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.self_attn = T5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.ff_layer_norm = T5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.feed_forward = T5FeedForward(config, param_dtype=param_dtype, dtype=dtype, rngs=rngs)

    def __call__(self, hidden_states: jax.Array, attention_mask: jax.Array | None = None) -> jax.Array:
        normed = self.self_attn_layer_norm(hidden_states)
        attn_output = self.self_attn(normed, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_output

        ff_input = self.ff_layer_norm(hidden_states)
        ff_output = self.feed_forward(ff_input)
        hidden_states = hidden_states + ff_output
        return hidden_states


class T5Encoder(nnx.Module):
    def __init__(
        self,
        config: T5EncoderConfig,
        *,
        param_dtype: jnp.dtype = jnp.float32,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.embed_tokens = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.blocks = nnx.List(
            [
                T5Block(
                    config,
                    has_relative_attention_bias=(i == 0),
                    param_dtype=param_dtype,
                    dtype=dtype,
                    rngs=rngs,
                )
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        hidden_states = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = jnp.ones((hidden_states.shape[0], hidden_states.shape[1]), dtype=hidden_states.dtype)
        attention_mask = attention_mask.astype(hidden_states.dtype)

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.final_layer_norm(hidden_states)
        return {"last_hidden_state": hidden_states}
