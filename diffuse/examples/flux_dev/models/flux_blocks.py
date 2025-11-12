"""FLUX Transformer blocks for text-to-image generation.

Based on FLUX.1-dev architecture from black-forest-labs.
"""

import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .mlp import Mlp
from .utils import modulate


def _apply_rotary(x: Array, rotary: Array | tuple[Array, Array]) -> Array:
    """Apply rotary embedding to queries/keys.

    Supports either (cos, sin) tuples (for classic RoPE) or full 2x2 rotation
    matrices of shape (seq, dim, 2, 2) as produced by FluxPosEmbed.
    """
    if isinstance(rotary, tuple):
        cos, sin = rotary
        dtype = x.dtype
        cos = cos.astype(dtype)
        sin = sin.astype(dtype)

        if cos.shape[-1] != x.shape[-1]:
            if x.shape[-1] % cos.shape[-1] != 0:
                raise ValueError("Rotary embedding dimension mismatch.")
            repeat = x.shape[-1] // cos.shape[-1]
            cos = jnp.repeat(cos, repeat, axis=-1)
            sin = jnp.repeat(sin, repeat, axis=-1)

        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]
        x_rotated = jnp.stack([-x_imag, x_real], axis=-1).reshape(x.shape)
        return x * cos + x_rotated * sin

    # Matrix-based rotary embedding (seq, dim_half, 2, 2)
    matrix = rotary.astype(x.dtype)
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)  # (..., dim_half, 2)
    xr = x_reshaped[..., 0]
    xi = x_reshaped[..., 1]

    mat = matrix[None, :, None, :, :, :]  # (1, seq, 1, dim_half, 2, 2)
    out_real = mat[..., 0, 0] * xr + mat[..., 0, 1] * xi
    out_imag = mat[..., 1, 0] * xr + mat[..., 1, 1] * xi
    rotated = jnp.stack([out_real, out_imag], axis=-1)
    return rotated.reshape(x.shape)


########################################################
################ FLUX Attention with QK Norm ###########
########################################################


class FluxAttention(nnx.Module):
    """FLUX attention with QK normalization supporting dual-stream inputs.

    This module mirrors the reference FLUX implementation: image tokens (hidden states)
    and text tokens (encoder states) have separate QKV projections and output heads,
    but share the attention kernel. This lets us perform joint attention across the
    concatenated stream while preserving modality-specific parameters.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if in_channels % num_heads != 0:
            raise ValueError("in_channels must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        # Image stream projections
        self.img_qkv = nnx.Linear(
            in_features=in_channels,
            out_features=3 * in_channels,
            use_bias=qkv_bias,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.img_proj = nnx.Linear(
            in_features=in_channels,
            out_features=in_channels,
            use_bias=proj_bias,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.img_query_norm = nnx.RMSNorm(
            num_features=self.head_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.img_key_norm = nnx.RMSNorm(
            num_features=self.head_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        # Text/encoder stream projections
        self.txt_qkv = nnx.Linear(
            in_features=in_channels,
            out_features=3 * in_channels,
            use_bias=qkv_bias,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.txt_proj = nnx.Linear(
            in_features=in_channels,
            out_features=in_channels,
            use_bias=proj_bias,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.txt_query_norm = nnx.RMSNorm(
            num_features=self.head_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.txt_key_norm = nnx.RMSNorm(
            num_features=self.head_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def _project(
        self,
        linear: nnx.Linear,
        query_norm: nnx.RMSNorm,
        key_norm: nnx.RMSNorm,
        states: ArrayLike,
        rotary_emb: Array | tuple[Array, Array] | None,
    ) -> tuple[Array, Array, Array]:
        qkv = linear(states)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.num_heads)

        q = query_norm(q)
        k = key_norm(k)

        if rotary_emb is not None:
            q = _apply_rotary(q, rotary_emb)
            k = _apply_rotary(k, rotary_emb)

        return q, k, v

    def __call__(
        self,
        hidden_states: ArrayLike,
        rotary_emb: Array | tuple[Array, Array] | None = None,
        *,
        encoder_hidden_states: ArrayLike | None = None,
        encoder_rotary_emb: Array | tuple[Array, Array] | None = None,
    ) -> Array | tuple[Array, Array]:
        if encoder_hidden_states is None:
            # Standard self-attention on image tokens only.
            q, k, v = self._project(self.img_qkv, self.img_query_norm, self.img_key_norm, hidden_states, rotary_emb)
            attn = nnx.dot_product_attention(q, k, v)
            attn = rearrange(attn, "b s h d -> b s (h d)")
            return self.img_proj(attn)

        # Joint attention across image (hidden) and text (encoder) tokens.
        img_q, img_k, img_v = self._project(
            self.img_qkv,
            self.img_query_norm,
            self.img_key_norm,
            hidden_states,
            rotary_emb,
        )
        txt_q, txt_k, txt_v = self._project(
            self.txt_qkv,
            self.txt_query_norm,
            self.txt_key_norm,
            encoder_hidden_states,
            encoder_rotary_emb if encoder_rotary_emb is not None else rotary_emb,
        )

        q = jnp.concatenate([txt_q, img_q], axis=1)
        k = jnp.concatenate([txt_k, img_k], axis=1)
        v = jnp.concatenate([txt_v, img_v], axis=1)

        attn = nnx.dot_product_attention(q, k, v)
        attn = rearrange(attn, "b s h d -> b s (h d)")

        txt_len = encoder_hidden_states.shape[1]
        txt_out, img_out = jnp.split(attn, [txt_len], axis=1)
        img_out = self.img_proj(img_out)
        txt_out = self.txt_proj(txt_out)
        return img_out, txt_out


########################################################
################ FLUX Timestep Embedder ################
########################################################


class FluxTimestepEmbedder(nnx.Module):
    """MLP embedder for timestep, guidance, and pooled text embeddings.

    Projects a scalar or vector input through a 2-layer MLP with SiLU activation.
    Used for timestep, guidance scale, and pooled text conditioning in FLUX.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        *,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.hidden_size = hidden_size

        self.in_layer = nnx.Linear(
            in_features=in_features,
            out_features=hidden_size,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.out_layer = nnx.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike) -> Array:
        """Forward pass through the embedder.

        Args:
            x: Input tensor of shape (batch, in_features)

        Returns:
            Embedded tensor of shape (batch, hidden_size)
        """
        emb = self.in_layer(x)
        emb = nnx.silu(emb)
        emb = self.out_layer(emb)
        return emb


########################################################
################ FLUX Double Block #####################
########################################################


class FluxDoubleBlock(nnx.Module):
    """Dual-stream transformer block for FLUX.

    Processes image and text tokens with a shared joint attention layer while
    keeping per-modality MLPs and modulation. This mirrors the MaxDiffusion
    reference implementation where the dual-stream stage performs cross-modal
    attention before the streams are merged for the single blocks.

    Architecture:
        - Joint attention over concatenated [text, image] tokens
        - Separate MLP for image and text tokens
        - Shared timestep/guidance conditioning via modulation
        - Uses RMSNorm with separate query/key normalization
    """

    def __init__(
        self,
        hidden_dim: int = 3072,
        num_heads: int = 24,
        *,
        mlp_ratio: float = 4.0,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Image stream components
        self.img_norm1 = nnx.RMSNorm(
            num_features=hidden_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.img_norm2 = nnx.RMSNorm(
            num_features=hidden_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.attn = FluxAttention(
            in_channels=hidden_dim,
            num_heads=num_heads,
            qkv_bias=True,
            proj_bias=True,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.img_mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            out_features=hidden_dim,
            activation=lambda x: nnx.gelu(x, approximate=True).astype(dtype),
            dropout=False,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        # Text stream components
        self.txt_norm1 = nnx.RMSNorm(
            num_features=hidden_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.txt_norm2 = nnx.RMSNorm(
            num_features=hidden_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.txt_mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            out_features=hidden_dim,
            activation=lambda x: nnx.gelu(x, approximate=True).astype(dtype),
            dropout=False,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        # Modulation layers
        # Image modulation: 6 params (scale/shift for norm1, norm2, gate for attn, mlp)
        self.img_mod = nnx.Linear(
            in_features=hidden_dim,
            out_features=6 * hidden_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        # Text modulation: 6 params (same structure as image)
        self.txt_mod = nnx.Linear(
            in_features=hidden_dim,
            out_features=6 * hidden_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        img: Array,
        txt: Array,
        vec: Array,
        rotary_emb: Array | tuple[Array, Array],
    ) -> tuple[Array, Array]:
        """Forward pass through dual-stream block.

        Args:
            img: Image tokens of shape (batch, num_img_tokens, hidden_dim)
            txt: Text tokens of shape (batch, num_txt_tokens, hidden_dim)
            vec: Conditioning vector of shape (batch, hidden_dim) from timestep/guidance

        Returns:
            Tuple of (processed_img, processed_txt) with same shapes as inputs
        """
        # Compute modulation parameters
        img_mod_params = self.img_mod(nnx.silu(vec))
        img_shift_msa, img_scale_msa, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = jnp.split(img_mod_params, 6, axis=-1)

        txt_mod_params = self.txt_mod(nnx.silu(vec))
        txt_shift_msa, txt_scale_msa, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = jnp.split(txt_mod_params, 6, axis=-1)

        # Joint attention across text and image tokens
        img_norm = modulate(self.img_norm1(img), img_shift_msa, img_scale_msa)
        txt_norm = modulate(self.txt_norm1(txt), txt_shift_msa, txt_scale_msa)

        if isinstance(rotary_emb, tuple):
            cos, sin = rotary_emb
            txt_len = txt_norm.shape[1]
            img_len = img_norm.shape[1]
            txt_rotary = (cos[:txt_len], sin[:txt_len])
            img_rotary = (cos[txt_len : txt_len + img_len], sin[txt_len : txt_len + img_len])
        else:
            txt_len = txt_norm.shape[1]
            img_len = img_norm.shape[1]
            txt_rotary = rotary_emb[:txt_len] if rotary_emb is not None else None
            img_rotary = rotary_emb[txt_len : txt_len + img_len] if rotary_emb is not None else None

        img_attn, txt_attn = self.attn(
            img_norm,
            img_rotary,
            encoder_hidden_states=txt_norm,
            encoder_rotary_emb=txt_rotary,
        )

        img = img + img_gate_msa[:, None, :] * img_attn
        txt = txt + txt_gate_msa[:, None, :] * txt_attn

        # Image stream: MLP + residual
        img_norm = modulate(self.img_norm2(img), img_shift_mlp, img_scale_mlp)
        img = img + img_gate_mlp[:, None, :] * self.img_mlp(img_norm)

        # Text stream: MLP + residual
        txt_norm = modulate(self.txt_norm2(txt), txt_shift_mlp, txt_scale_mlp)
        txt = txt + txt_gate_mlp[:, None, :] * self.txt_mlp(txt_norm)

        return img, txt


########################################################
################ FLUX Single Block #####################
########################################################


class FluxSingleBlock(nnx.Module):
    """Unified-stream transformer block for FLUX.

    After the dual-stream blocks, FLUX concatenates image and text tokens and
    processes them together in single-stream blocks. These blocks use fused
    projections for efficiency (combined QKV and MLP projections).

    Architecture:
        - Self-attention over concatenated [img, txt] sequence
        - Fused linear projections for efficiency
        - Modulated with timestep/guidance conditioning
        - Uses RMSNorm
    """

    def __init__(
        self,
        hidden_dim: int = 3072,
        num_heads: int = 24,
        *,
        mlp_ratio: float = 4.0,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Normalization
        self.norm = nnx.RMSNorm(
            num_features=hidden_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        # QK normalization for attention
        self.query_norm = nnx.RMSNorm(
            num_features=self.head_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.key_norm = nnx.RMSNorm(
            num_features=self.head_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        # Fused projection: combines QKV (3*hidden_dim) + MLP expansion (mlp_ratio*hidden_dim)
        # For FLUX: 3*3072 + 4*3072 = 21504
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        fused_dim = 3 * hidden_dim + mlp_hidden_dim

        self.linear1 = nnx.Linear(
            in_features=hidden_dim,
            out_features=fused_dim,
            use_bias=True,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        # Output projection: combines attention output + MLP output
        # For FLUX: hidden_dim + mlp_hidden_dim = 3072 + 12288 = 15360 → 3072
        self.linear2 = nnx.Linear(
            in_features=hidden_dim + mlp_hidden_dim,
            out_features=hidden_dim,
            use_bias=True,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        # Modulation: 3 params (scale, shift, gate)
        # For FLUX: 3*3072 = 9216
        self.modulation = nnx.Linear(
            in_features=hidden_dim,
            out_features=3 * hidden_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Array,
        vec: Array,
        rotary_emb: Array | tuple[Array, Array] | None,
    ) -> Array:
        """Forward pass through single-stream block.

        Args:
            hidden_states: Concatenated [txt, img] tokens of shape (batch, seq_len, hidden_dim)
            vec: Conditioning vector of shape (batch, hidden_dim)
            rotary_emb: Rotary embedding matrix or (cos, sin) tuple for the concatenated sequence.

        Returns:
            Processed hidden states with the same shape as the input.
        """
        x = hidden_states

        # Compute modulation parameters
        mod_params = self.modulation(nnx.silu(vec))
        shift, scale, gate = jnp.split(mod_params, 3, axis=-1)

        # Normalize and modulate
        x_norm = modulate(self.norm(x), shift, scale)

        # Fused forward projection
        fused_proj = self.linear1(x_norm)

        # Split into QKV and MLP components
        qkv_dim = 3 * self.hidden_dim
        qkv, mlp_hidden = jnp.split(fused_proj, [qkv_dim], axis=-1)

        # Self-attention with QK normalization
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.num_heads)

        # Apply QK normalization and rotary embeddings
        q = self.query_norm(q)
        k = self.key_norm(k)
        if rotary_emb is not None:
            q = _apply_rotary(q, rotary_emb)
            k = _apply_rotary(k, rotary_emb)

        attn_out = nnx.dot_product_attention(q, k, v)
        attn_out = rearrange(attn_out, "b s h d -> b s (h d)")

        # MLP with GELU activation
        mlp_out = nnx.gelu(mlp_hidden, approximate=True)

        # Concatenate attention and MLP outputs
        combined = jnp.concatenate([attn_out, mlp_out], axis=-1)

        # Output projection with gated residual
        out = self.linear2(combined)
        x = x + gate[:, None, :] * out

        return x


########################################################
################ FLUX Final Layer ######################
########################################################


class FluxFinalLayer(nnx.Module):
    """Final output layer for FLUX transformer.

    Projects the transformer hidden states back to the latent space dimension
    with adaptive layer normalization conditioning.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        *,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.norm = nnx.RMSNorm(
            num_features=hidden_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        # AdaLN modulation: 2 params (scale, shift)
        self.adaLN_modulation = nnx.Linear(
            in_features=hidden_dim,
            out_features=2 * hidden_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        # Final projection to latent dimension
        self.linear = nnx.Linear(
            in_features=hidden_dim,
            out_features=latent_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: Array, vec: Array) -> Array:
        """Forward pass through final layer.

        Args:
            x: Hidden states of shape (batch, num_tokens, hidden_dim)
            vec: Conditioning vector of shape (batch, hidden_dim)

        Returns:
            Output tensor of shape (batch, num_tokens, latent_dim)
        """
        # Compute modulation parameters
        shift, scale = jnp.split(self.adaLN_modulation(nnx.silu(vec)), 2, axis=-1)

        # Modulate normalized hidden states
        x = modulate(self.norm(x), shift, scale)

        # Project to latent dimension
        x = self.linear(x)

        return x
