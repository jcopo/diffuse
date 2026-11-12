# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import MeasurementState
from .base import MRIMaskForwardModel, RADIAL_LINE_RANGE, RADIAL_SIGMA


@partial(jax.vmap, in_axes=(0, 0, None, None, None))
def _generate_line_soft(angle: Array, size: Array, shape: tuple[int, int], width: float, sigma: float) -> Array:
    """Smooth approximation of a radial acquisition line using regular FFT."""
    height, width_px = shape
    xs = jnp.linspace(-width_px / 2, width_px / 2, width_px)
    ys = jnp.linspace(-height / 2, height / 2, height)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")

    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    x_rot = grid_x * cos_a + grid_y * sin_a
    y_rot = -grid_x * sin_a + grid_y * cos_a

    soft = jax.nn.sigmoid((size / 2 - jnp.abs(x_rot)) / sigma)
    thickness = jax.nn.sigmoid((width / 2 - jnp.abs(y_rot)) / sigma)
    return soft * thickness


@partial(jax.vmap, in_axes=(0, 0, None))
def _generate_line_hard(angle: Array, size: Array, shape: tuple[int, int]) -> Array:
    """Hard binary line indicator (used for straight-through gradient)."""
    height, width_px = shape
    xs = jnp.linspace(-width_px / 2, width_px / 2, width_px)
    ys = jnp.linspace(-height / 2, height / 2, height)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")

    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    x_rot = grid_x * cos_a + grid_y * sin_a
    y_rot = -grid_x * sin_a + grid_y * cos_a

    # Use a small threshold instead of exact equality
    epsilon = 0.5  # Half pixel width
    mask = (jnp.abs(x_rot) <= size / 2) & (jnp.abs(y_rot) <= epsilon)
    return mask.astype(jnp.float32)


@dataclass
class RadialMask(MRIMaskForwardModel):
    """Radial k-space sampling pattern using regular FFT (not NUFFT)."""

    num_lines: int = 10
    line_width: float = 1.0
    sigma_soft: Optional[float] = None
    samples_per_line: int = 256

    def sample_design(self, rng_key: PRNGKeyArray) -> Array:
        """Sample random angles and line extents."""
        bounds = RADIAL_LINE_RANGE[self.data_model]
        angles_key, size_key = jax.random.split(rng_key)
        angles = jax.random.uniform(angles_key, shape=(self.num_lines,), minval=0.0, maxval=2 * jnp.pi)
        sizes = jax.random.uniform(size_key, shape=(self.num_lines,), **bounds)
        return jnp.stack([angles, sizes], axis=-1)

    def make(self, xi: Array, rng_key: Optional[PRNGKeyArray] = None) -> Array:
        """Generate radial mask from design parameters using regular FFT."""
        del rng_key
        xi = jnp.asarray(xi)
        if xi.shape != (self.num_lines, 2):
            raise ValueError(f"Expected design parameters of shape ({self.num_lines}, 2), got {xi.shape}")

        angle = xi[:, 0]
        size = xi[:, 1]
        shape = self.img_shape[:-1]

        # Generate soft/hard mask
        sigma = self.sigma_soft or RADIAL_SIGMA[self.data_model]
        soft_lines = _generate_line_soft(angle, size, shape, self.line_width, sigma)
        hard_lines = _generate_line_hard(angle, size, shape)

        # Straight-through estimator
        lines = soft_lines + jax.lax.stop_gradient(hard_lines - soft_lines)

        # Accumulate lines
        def accumulate(mask, line):
            inv_mask = 1.0 - mask
            return inv_mask * line + mask, None

        mask0 = jnp.zeros(shape)
        mask_final, _ = jax.lax.scan(accumulate, mask0, lines)

        return mask_final


__all__ = ["RadialMask"]
