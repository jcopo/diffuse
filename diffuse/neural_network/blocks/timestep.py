# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Base classes for modules that accept timestep embeddings."""

from typing import Optional

from flax import nnx
from jax import Array
from jax.typing import ArrayLike


class TimestepBlock(nnx.Module):
    """Base class for modules that can accept optional timestep embeddings.

    This abstract class defines the interface for neural network modules that
    support timestep conditioning, commonly used in diffusion models.
    """

    def __call__(self, x: ArrayLike, time_emb: Optional[ArrayLike] = None) -> Array:
        """Forward pass with optional timestep embedding.

        Args:
            x: Input tensor
            time_emb: Optional timestep embedding for conditioning

        Returns:
            Processed output tensor
        """
        pass


class TimestepEmbedSequential(nnx.Sequential, TimestepBlock):
    """Sequential container that passes timestep embeddings to compatible layers.

    Extends nnx.Sequential to support modules that accept timestep embeddings.
    Automatically passes time_emb to layers that inherit from TimestepBlock,
    while calling other layers without the timestep argument.
    """

    def __call__(self, x: ArrayLike, time_emb: Optional[ArrayLike] = None) -> Array:
        """Forward pass through sequential layers with optional timestep embedding.

        Args:
            x: Input tensor
            time_emb: Optional timestep embedding passed to TimestepBlock layers

        Returns:
            Output tensor after processing through all layers
        """
        for layer in self.layers:
            if isinstance(layer, TimestepBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        return x
