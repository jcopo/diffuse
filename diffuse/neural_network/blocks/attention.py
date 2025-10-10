# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Multi-head self-attention block for VAE-GAN and diffusion models.

This module implements spatial self-attention over feature maps, allowing each
spatial position to attend to all other positions. Commonly used in the bottleneck
of VAE encoders/decoders and UNet architectures for diffusion models.
"""

import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from jax import Array
from jax.typing import ArrayLike, DTypeLike


class AttnBlock(nnx.Module):
    """Multi-head self-attention block with spatial attention mechanism.

    Performs self-attention over spatial dimensions of feature maps, where each
    pixel can attend to all other pixels. Uses multi-head attention to learn
    diverse attention patterns. Automatically leverages JAX's optimized attention
    implementations including cuDNN flash attention when available.

    Args:
        in_channels: Number of input channels. Must be divisible by num_heads.
        num_heads: Number of attention heads. Default is 8.
        param_dtype: Data type for parameters (weights and biases).
        dtype: Data type for computation.
        rngs: Random number generators for parameter initialization.

    Raises:
        AssertionError: If in_channels is not divisible by num_heads.

    Example:
        >>> rngs = nnx.Rngs(42)
        >>> attn = AttnBlock(in_channels=256, num_heads=8, rngs=rngs)
        >>> x = jnp.ones((2, 16, 16, 256))  # (batch, height, width, channels)
        >>> output = attn(x)  # Same shape as input
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        assert in_channels % num_heads == 0, f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"

        self.dtype = dtype
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

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
            param_dtype=param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.k = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
            param_dtype=param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.v = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1, 1),
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
        """Compute multi-head self-attention over spatial dimensions.

        Args:
            h_: Input feature map of shape (batch, height, width, channels).

        Returns:
            Attention output of shape (batch, height, width, channels).
        """
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, h, w, c = q.shape

        # Reshape to multi-head format: [batch, seq_len, num_heads, head_dim]
        q = rearrange(q, "b h w (nh hd) -> b (h w) nh hd", nh=self.num_heads, hd=self.head_dim)
        k = rearrange(k, "b h w (nh hd) -> b (h w) nh hd", nh=self.num_heads, hd=self.head_dim)
        v = rearrange(v, "b h w (nh hd) -> b (h w) nh hd", nh=self.num_heads, hd=self.head_dim)

        # Apply multi-head attention (automatically uses best backend including cuDNN flash attention)
        h_ = nnx.dot_product_attention(q, k, v, dtype=self.dtype)

        # Reshape back to spatial format
        h_ = rearrange(h_, "b (h w) nh hd -> b h w (nh hd)", h=h, w=w, nh=self.num_heads, hd=self.head_dim)

        return h_

    def __call__(self, x: ArrayLike) -> Array:
        """Forward pass with residual connection.

        Args:
            x: Input feature map of shape (batch, height, width, channels).

        Returns:
            Output with residual connection: x + proj_out(attention(x)).
        """
        return x + self.proj_out(self.attention(x))
