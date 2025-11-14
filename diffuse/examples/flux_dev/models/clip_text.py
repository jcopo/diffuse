"""Minimal CLIP text transformer implemented with Flax NNX."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


def quick_gelu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(1.702 * x)


ACT_FNS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "quick_gelu": quick_gelu,
}


@dataclass
class CLIPTextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    attention_dropout: float = 0.0
    dropout: float = 0.0
    hidden_act: str = "quick_gelu"
    pad_token_id: int = 1
    eos_token_id: int = 2
    bos_token_id: int = 0
    projection_dim: int = 0
    model_type: str | None = None
    architectures: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CLIPTextConfig:
        return cls(
            vocab_size=data["vocab_size"],
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            max_position_embeddings=data["max_position_embeddings"],
            layer_norm_eps=data.get("layer_norm_eps", 1e-5),
            initializer_range=data.get("initializer_range", 0.02),
            initializer_factor=data.get("initializer_factor", 1.0),
            attention_dropout=data.get("attention_dropout", 0.0),
            dropout=data.get("dropout", 0.0),
            hidden_act=data.get("hidden_act", "quick_gelu"),
            pad_token_id=data.get("pad_token_id", 1),
            eos_token_id=data.get("eos_token_id", 2),
            bos_token_id=data.get("bos_token_id", 0),
            projection_dim=data.get("projection_dim", data["hidden_size"]),
            model_type=data.get("model_type"),
            architectures=data.get("architectures"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "layer_norm_eps": self.layer_norm_eps,
            "initializer_range": self.initializer_range,
            "initializer_factor": self.initializer_factor,
            "attention_dropout": self.attention_dropout,
            "dropout": self.dropout,
            "hidden_act": self.hidden_act,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "projection_dim": self.projection_dim,
            "model_type": self.model_type,
            "architectures": self.architectures,
        }


class CLIPSelfAttention(nnx.Module):
    def __init__(
        self,
        config: CLIPTextConfig,
        *,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        if self.head_dim * self.num_heads != config.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (hidden_size={config.hidden_size}, heads={config.num_attention_heads})"
            )
        kernel_init = jax.nn.initializers.normal(config.initializer_range)
        self.q_proj = nnx.Linear(
            config.hidden_size,
            config.hidden_size,
            use_bias=True,
            param_dtype=param_dtype,
            dtype=dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            config.hidden_size,
            config.hidden_size,
            use_bias=True,
            param_dtype=param_dtype,
            dtype=dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            config.hidden_size,
            config.hidden_size,
            use_bias=True,
            param_dtype=param_dtype,
            dtype=dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            config.hidden_size,
            config.hidden_size,
            use_bias=True,
            param_dtype=param_dtype,
            dtype=dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(self, hidden_states: jax.Array, attention_mask: jax.Array | None = None) -> jax.Array:
        batch, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        def reshape_heads(x: jax.Array) -> jax.Array:
            return x.reshape(batch, seq_len, self.num_heads, self.head_dim)

        query = reshape_heads(self.q_proj(hidden_states))
        key = reshape_heads(self.k_proj(hidden_states))
        value = reshape_heads(self.v_proj(hidden_states))

        query = query * (self.head_dim**-0.5)

        attn_scores = jnp.einsum("bthd,bshd->bhts", query, key)

        causal_mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=bool), k=1)
        attn_scores = jnp.where(causal_mask[None, None, :, :], jnp.finfo(dtype).min, attn_scores)

        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :] == 0
            attn_scores = jnp.where(pad_mask, jnp.finfo(dtype).min, attn_scores)

        attn_probs = jax.nn.softmax(attn_scores, axis=-1).astype(dtype)
        attn_output = jnp.einsum("bhts,bshd->bthd", attn_probs, value)
        attn_output = attn_output.reshape(batch, seq_len, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class CLIPMLP(nnx.Module):
    def __init__(
        self,
        config: CLIPTextConfig,
        *,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        kernel_init = jax.nn.initializers.normal(config.initializer_range)
        self.fc1 = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=True,
            param_dtype=param_dtype,
            dtype=dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=True,
            param_dtype=param_dtype,
            dtype=dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        if config.hidden_act not in ACT_FNS:
            raise ValueError(f"Unsupported activation: {config.hidden_act}")
        self.activation = ACT_FNS[config.hidden_act]

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nnx.Module):
    def __init__(
        self,
        config: CLIPTextConfig,
        *,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.self_attn = CLIPSelfAttention(
            config,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.ln1 = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.mlp = CLIPMLP(
            config,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, hidden_states: jax.Array, attention_mask: jax.Array | None) -> jax.Array:
        attn_input = self.ln1(hidden_states)
        attn_output = self.self_attn(attn_input, attention_mask=attention_mask)
        hidden_states = hidden_states + attn_output

        ff_input = self.ln2(hidden_states)
        ff_output = self.mlp(ff_input)
        hidden_states = hidden_states + ff_output
        return hidden_states


class CLIPTextEncoder(nnx.Module):
    def __init__(
        self,
        config: CLIPTextConfig,
        *,
        param_dtype: jnp.dtype = jnp.float32,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.config = config
        embed_init = jax.nn.initializers.normal(config.initializer_range)
        self.token_embedding = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            embedding_init=embed_init,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.position_embedding = nnx.Embed(
            num_embeddings=config.max_position_embeddings,
            features=config.hidden_size,
            embedding_init=embed_init,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.layers = nnx.List(
            [
                CLIPEncoderLayer(
                    config,
                    param_dtype=param_dtype,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = nnx.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        input_ids = input_ids.astype(jnp.int32)
        batch, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        hidden_states = token_embeddings + position_embeddings

        if attention_mask is None:
            attention_mask = jnp.ones((batch, seq_len), dtype=jnp.int32)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.final_layer_norm(hidden_states)

        eos_token_id = self.config.eos_token_id
        eos_mask = jnp.equal(input_ids, eos_token_id)
        eos_indices = jnp.argmax(eos_mask, axis=-1)
        pooled_output = hidden_states[jnp.arange(batch), eos_indices]

        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooled_output,
        }
