# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jaxtyping import Array
from diffuse.base_forward_model import MeasurementState, ForwardModel


@dataclass
class MaskState:
    y: Array
    mask_history: Array
    xi: Array


@dataclass
class SquareMask(ForwardModel):
    size: int
    img_shape: tuple
    std: float = 1.0

    def make(self, xi: Array) -> Array:
        """Create a differentiable square mask."""
        # assert xi is a 2D array
        assert xi.shape[0] == 2
        height, width, *_ = self.img_shape
        y, x = jnp.mgrid[:height, :width]

        # Calculate distances from the center
        y_dist = jnp.abs(y - xi[1])
        x_dist = jnp.abs(x - xi[0])

        # Create a soft mask using sigmoid function
        mask_half_size = self.size // 2
        softness = 0.1  # Adjust this value to control the softness of the edges

        mask = jax.nn.sigmoid((-jnp.maximum(y_dist, x_dist) + mask_half_size) / softness)
        # return jnp.where(mask > 0.5, 1.0, 0.0)[..., None]
        return mask[..., None]

    def apply(self, img: Array, measurement_state: MeasurementState):
        hist_mask = measurement_state.mask_history
        return img * hist_mask

    def restore(self, img: Array, measurement_state: MeasurementState):
        mask = measurement_state.mask_history
        inv_mask = 1 - mask
        return img * inv_mask
