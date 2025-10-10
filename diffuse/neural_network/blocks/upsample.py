# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Upsampling blocks for neural networks.

This module provides various upsampling methods including nearest neighbor resize
and pixel shuffle (sub-pixel convolution) for increasing spatial resolution in
generative models and decoders.
"""

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
from jax import Array
from jax.typing import ArrayLike, DTypeLike


class PixelShuffle(nnx.Module):
    """Pixel shuffle operation for sub-pixel convolution upsampling.

    Rearranges elements in a tensor from channel dimension to spatial dimensions.
    Commonly used in super-resolution and generative models for learnable upsampling.

    Args:
        scale: Upsampling scale factor (both height and width).

    Example:
        >>> shuffle = PixelShuffle(scale=2)
        >>> x = jnp.ones((1, 4, 4, 16))  # 4 channels per output pixel after 2x upsample
        >>> y = shuffle(x)  # Shape: (1, 8, 8, 4)
    """

    def __init__(self, scale: int):
        self.scale = scale

    def __call__(self, x: ArrayLike) -> Array:
        """Apply pixel shuffle upsampling.

        Args:
            x: Input tensor of shape (batch, height, width, channels).

        Returns:
            Upsampled tensor with spatial dimensions increased by scale factor.
        """
        return rearrange(x, "b h w (h2 w2 c) -> b (h h2) (w w2) c", h2=self.scale, w2=self.scale)


class Upsample(nnx.Module):
    """Flexible upsampling block with multiple methods.

    Supports different upsampling methods: nearest neighbor resize and pixel shuffle.
    Includes a convolutional layer after upsampling for feature refinement.

    Args:
        in_channels: Number of input channels.
        method: Upsampling method, either "resize" or "pixel_shuffle".
        scale_factor: Upsampling scale factor (integer).
        param_dtype: Data type for parameters.
        dtype: Data type for computation.
        rngs: Random number generators for parameter initialization.

    Example:
        >>> rngs = nnx.Rngs(42)
        >>> upsample = Upsample(in_channels=128, method="resize", rngs=rngs)
        >>> x = jnp.ones((2, 16, 16, 128))
        >>> y = upsample(x)  # Shape: (2, 32, 32, 128)
    """

    def __init__(
        self,
        in_channels: int,
        method: str = "resize",
        scale_factor: int = 2,
        param_dtype: DTypeLike = jnp.float32,
        dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        self.method = method
        self.scale_factor = scale_factor

        self.conv_resize = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

        self.conv_pixel_shuffle = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels * (self.scale_factor**2),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=(1, 1),
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.pixel_shuffle = PixelShuffle(scale=self.scale_factor)

    def __call__(self, x: ArrayLike) -> Array:
        """Apply upsampling with the specified method.

        Args:
            x: Input tensor of shape (batch, height, width, in_channels).

        Returns:
            Upsampled tensor with refined features after convolution.

        Raises:
            ValueError: If method is not "resize" or "pixel_shuffle".
        """
        if self.method == "resize":
            b, h, w, c = x.shape
            new_height = int(h * self.scale_factor)
            new_width = int(w * self.scale_factor)
            new_shape = (b, new_height, new_width, c)
            x = jax.image.resize(x, new_shape, method="nearest")
            return self.conv_resize(x)
        elif self.method == "pixel_shuffle":
            x = self.conv_pixel_shuffle(x)
            x = self.pixel_shuffle(x)
            return x
        else:
            raise ValueError(f"Invalid method: {self.method}")
