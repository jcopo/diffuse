"""FLUX Transformer for text-to-image generation.

Based on FLUX.1-dev architecture from black-forest-labs.
Implements a dual-stream transformer that processes image and text tokens separately
before merging them for unified processing.
"""

import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from chex import dataclass

from .flux_blocks import FluxDoubleBlock, FluxFinalLayer, FluxSingleBlock, FluxTimestepEmbedder


@dataclass
class FluxTransformerOutput:
    """Output of the FLUX Transformer model."""
    output: Array


def timestep_embedding(
    timesteps: ArrayLike,
    dim: int,
    max_period: int = 10000,
    time_factor: float = 1000.0,
) -> Array:
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: Timestep values of shape (batch,)
        dim: Embedding dimension
        max_period: Maximum period for sinusoidal encoding
        time_factor: Scaling factor applied to timesteps before computing frequencies.

    Returns:
        Timestep embeddings of shape (batch, dim)
    """
    timesteps = jnp.asarray(timesteps, dtype=jnp.float32) * time_factor
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = timesteps[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


def _get_1d_rotary_pos_embed(dim: int, positions: Array, theta: float = 10000.0) -> tuple[Array, Array]:
    positions = jnp.asarray(positions, dtype=jnp.float32)
    half_dim = dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(half_dim, dtype=jnp.float32) / half_dim))
    freqs = positions[:, None] * inv_freq[None, :]
    cos = jnp.repeat(jnp.cos(freqs), 2, axis=-1)
    sin = jnp.repeat(jnp.sin(freqs), 2, axis=-1)
    return cos, sin


class FluxPosEmbed(nnx.Module):
    def __init__(self, theta: float = 10000.0, axes_dim: tuple[int, ...] = (16, 56, 56)):
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: Array) -> Array:
        pos = jnp.asarray(ids, dtype=jnp.float32)
        if len(self.axes_dim) != pos.shape[-1]:
            raise ValueError(f"Expected positional ids with {len(self.axes_dim)} axes, but received {pos.shape[-1]} axes.")
        rope_segments = []
        for axis, dim in enumerate(self.axes_dim):
            if dim % 2 != 0:
                raise ValueError("RoPE axis dimensions must be even.")
            half = dim // 2
            scale = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
            omega = 1.0 / (self.theta**scale)
            angles = pos[:, axis][:, None] * omega[None, :]
            cos = jnp.cos(angles)
            sin = jnp.sin(angles)
            rope = jnp.stack([cos, -sin, sin, cos], axis=-1)  # (tokens, half, 4)
            rope = rope.reshape(pos.shape[0], half, 2, 2)
            rope_segments.append(rope)
        return jnp.concatenate(rope_segments, axis=1)


class FluxTransformer2D(nnx.Module):
    """FLUX Transformer for text-to-image generation.

    A dual-stream transformer that processes image latents and text embeddings
    through separate pathways before merging them for joint processing.

    Architecture:
        1. Input projections: img_in, txt_in, time_in, vector_in, guidance_in
        2. 19 double blocks: separate processing of image and text streams
        3. 38 single blocks: unified processing of concatenated streams
        4. Final layer: project back to latent space

    Args:
        in_channels: Number of input latent channels (default: 64 for FLUX VAE)
        hidden_dim: Hidden dimension of transformer (default: 3072)
        num_heads: Number of attention heads (default: 24)
        num_double_layers: Number of dual-stream blocks (default: 19)
        num_single_layers: Number of single-stream blocks (default: 38)
        joint_attention_dim: Dimension of text embeddings (default: 4096 for T5-XXL)
        pooled_projection_dim: Dimension of pooled text embedding (default: 768 for CLIP)
        mlp_ratio: MLP expansion ratio (default: 4.0)
        guidance_embeds: Whether to use guidance scale conditioning (default: True)
        param_dtype: Parameter dtype (default: float32)
        dtype: Computation dtype (default: float32)
        rngs: Random number generator state
    """

    def __init__(
        self,
        *,
        in_channels: int = 64,
        hidden_dim: int = 3072,
        num_heads: int = 24,
        num_double_layers: int = 19,
        num_single_layers: int = 38,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        mlp_ratio: float = 4.0,
        guidance_embeds: bool = True,
        axes_dims_rope: tuple[int, ...] = (16, 56, 56),
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_double_layers = num_double_layers
        self.num_single_layers = num_single_layers
        self.joint_attention_dim = joint_attention_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.guidance_embeds = guidance_embeds
        self.axes_dims_rope = axes_dims_rope

        # Input projections
        self.img_in = nnx.Linear(
            in_features=in_channels,
            out_features=hidden_dim,
            use_bias=True,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        self.txt_in = nnx.Linear(
            in_features=joint_attention_dim,
            out_features=hidden_dim,
            use_bias=True,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        # Timestep embedding (256 → hidden_dim)
        self.time_in = FluxTimestepEmbedder(
            in_features=256,
            hidden_size=hidden_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        # Pooled text projection (CLIP pooled embedding)
        self.vector_in = FluxTimestepEmbedder(
            in_features=pooled_projection_dim,
            hidden_size=hidden_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        # Guidance scale embedding (if enabled)
        if guidance_embeds:
            self.guidance_in = FluxTimestepEmbedder(
                in_features=256,
                hidden_size=hidden_dim,
                param_dtype=param_dtype,
                dtype=dtype,
                rngs=rngs,
            )

        self.pos_embed = FluxPosEmbed(theta=10000.0, axes_dim=axes_dims_rope)

        # Double blocks (dual-stream processing)
        self.double_blocks = nnx.List(
            [
                FluxDoubleBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    param_dtype=param_dtype,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(num_double_layers)
            ]
        )

        # Single blocks (unified processing)
        self.single_blocks = nnx.List(
            [
                FluxSingleBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    param_dtype=param_dtype,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(num_single_layers)
            ]
        )

        # Final output layer
        self.final_layer = FluxFinalLayer(
            hidden_dim=hidden_dim,
            latent_dim=in_channels,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        img_latents: ArrayLike,
        txt_embeddings: ArrayLike,
        timesteps: ArrayLike,
        pooled_txt_emb: ArrayLike,
        guidance_scale: ArrayLike | None = None,
        img_ids: ArrayLike | None = None,
        txt_ids: ArrayLike | None = None,
    ) -> FluxTransformerOutput:
        """Forward pass through FLUX transformer.

        Args:
            img_latents: Image latents of shape (batch, height, width, in_channels)
                         or (batch, num_patches, in_channels) if pre-flattened
            txt_embeddings: Text embeddings of shape (batch, seq_len, joint_attention_dim)
                            Pre-encoded from T5-XXL (4096-dim)
            timesteps: Diffusion timesteps of shape (batch,) or scalar
            pooled_txt_emb: Pooled text embedding of shape (batch, pooled_projection_dim)
                            From CLIP text encoder (768-dim)
            guidance_scale: Guidance scale values of shape (batch,) or scalar (optional)
            img_ids: Image latent positional ids of shape (num_img_tokens, 3)
            txt_ids: Text positional ids of shape (num_txt_tokens, 3)

        Returns:
            FluxTransformerOutput containing predicted noise/velocity of same shape as img_latents
        """
        if img_ids is None or txt_ids is None:
            raise ValueError("FluxTransformer2D requires `img_ids` and `txt_ids`.")

        # Handle batch dimension
        squeeze = False
        if img_latents.ndim == 3:
            img_latents = jnp.expand_dims(img_latents, 0)
            txt_embeddings = jnp.expand_dims(txt_embeddings, 0)
            pooled_txt_emb = jnp.expand_dims(pooled_txt_emb, 0)
            squeeze = True

        batch = img_latents.shape[0]

        # Flatten spatial dimensions if needed
        if img_latents.ndim == 4:
            # (batch, h, w, c) → (batch, h*w, c)
            h, w = img_latents.shape[1:3]
            img_latents = img_latents.reshape(batch, h * w, -1)
            spatial_shape = (h, w)
        else:
            spatial_shape = None
            h = w = None

        img_ids_arr = jnp.asarray(img_ids, dtype=jnp.float32)
        if img_ids_arr.ndim == 3:
            img_ids_arr = img_ids_arr[0]
        txt_ids_arr = jnp.asarray(txt_ids, dtype=jnp.float32)
        if txt_ids_arr.ndim == 3:
            txt_ids_arr = txt_ids_arr[0]

        # Handle timesteps
        if timesteps is None:
            timesteps = jnp.zeros((batch,), dtype=jnp.float32)
        if jnp.ndim(timesteps) == 0:
            timesteps = jnp.array([timesteps])
        if timesteps.ndim == 2:
            timesteps = timesteps.squeeze(axis=-1)

        # Handle pooled text embedding
        if pooled_txt_emb.ndim == 1:
            pooled_txt_emb = jnp.expand_dims(pooled_txt_emb, 0)

        # Project inputs
        img = self.img_in(img_latents)  # (batch, num_img_tokens, hidden_dim)
        txt = self.txt_in(txt_embeddings)  # (batch, num_txt_tokens, hidden_dim)

        if img_ids_arr.shape[0] != img.shape[1]:
            raise ValueError(f"Image id count {img_ids_arr.shape[0]} does not match latent tokens {img.shape[1]}")
        if txt_ids_arr.shape[0] != txt.shape[1]:
            raise ValueError(f"Text id count {txt_ids_arr.shape[0]} does not match text tokens {txt.shape[1]}")

        # Create conditioning vector by summing all embeddings
        # Timestep embedding
        t_emb = timestep_embedding(timesteps, 256)
        t_vec = self.time_in(t_emb)

        # Pooled text embedding
        pooled_vec = self.vector_in(pooled_txt_emb)

        # Sum timestep and pooled text embeddings
        vec = t_vec + pooled_vec

        # Add guidance embedding if enabled
        if self.guidance_embeds and guidance_scale is not None:
            if jnp.ndim(guidance_scale) == 0:
                guidance_scale = jnp.array([guidance_scale])
            if guidance_scale.ndim == 2:
                guidance_scale = guidance_scale.squeeze(axis=-1)
            g_emb = timestep_embedding(guidance_scale, 256)
            g_vec = self.guidance_in(g_emb)
            vec = vec + g_vec

        ids = jnp.concatenate([txt_ids_arr, img_ids_arr], axis=0)
        rotary_emb = self.pos_embed(ids).astype(img.dtype)

        # Process through double blocks (dual-stream)
        for block in self.double_blocks:
            img, txt = block(img, txt, vec, rotary_emb)

        # Merge streams for single-block processing
        txt_len = txt.shape[1]
        hidden_states = jnp.concatenate([txt, img], axis=1)

        # Process through single blocks (unified stream)
        for block in self.single_blocks:
            hidden_states = block(hidden_states, vec, rotary_emb)

        # Extract image tokens (drop text tokens)
        img = hidden_states[:, txt_len:, :]

        # Final projection back to latent space
        output = self.final_layer(img, vec)

        # Reshape back to spatial dimensions if needed
        if spatial_shape is not None:
            h, w = spatial_shape
            output = output.reshape(batch, h, w, self.in_channels)

        if squeeze:
            output = output.squeeze(axis=0)

        return FluxTransformerOutput(output=output)
