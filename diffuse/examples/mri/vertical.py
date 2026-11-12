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
import optax

from .base import MRIMaskForwardModel, VERTICAL_SIGMA


def _generate_vertical_line_soft(x_pos: Array, shape: tuple[int, int], sigma: float) -> Array:
    height, width = shape
    x_pos = x_pos * width - width / 2.0

    xs = jnp.linspace(-width / 2.0, width / 2.0, width)
    ys = jnp.linspace(-height / 2.0, height / 2.0, height)
    _, grid_x = jnp.meshgrid(ys, xs, indexing="ij")
    return jax.nn.sigmoid(-jnp.abs(grid_x - x_pos) / sigma)


def _generate_vertical_line_hard(x_pos: Array, shape: tuple[int, int]) -> Array:
    height, width = shape
    pixel_x = jnp.floor(x_pos * (width - 1)).astype(jnp.int32)
    pixel_x = jnp.clip(pixel_x, 0, width - 1)
    mask = jnp.zeros((height, width), dtype=jnp.float32)
    return mask.at[:, pixel_x].set(1.0)


@partial(jax.vmap, in_axes=(0, None, None))
def _generate_vertical_line(x_pos: Array, shape: tuple[int, int], sigma: float) -> Array:
    soft = _generate_vertical_line_soft(x_pos, shape, sigma)
    hard = _generate_vertical_line_hard(x_pos, shape)
    return soft + jax.lax.stop_gradient(hard - soft)


@dataclass
class VerticalMask(MRIMaskForwardModel):
    """Soft vertical line sampling with optional thickness tuning."""

    num_lines: int = 1
    sigma: Optional[float] = None

    def sample_design(self, rng_key: PRNGKeyArray) -> Array:
        return jax.random.uniform(rng_key, shape=(self.num_lines,), minval=0.0, maxval=1.0)

    def projection_design(self, xi: Array) -> Array:
        return optax.projections.projection_box(xi, lower=0.0, upper=1.0)

    def make(self, xi: Array, rng_key: Optional[PRNGKeyArray] = None) -> Array:
        del rng_key
        xi = self.projection_design(jnp.asarray(xi))
        if xi.shape != (self.num_lines,):
            raise ValueError(f"Expected design parameters of shape ({self.num_lines},), got {xi.shape}")

        sigma = self.sigma or VERTICAL_SIGMA[self.data_model]
        lines = _generate_vertical_line(xi, self.img_shape[:-1], sigma)

        def accumulate(mask, line):
            inv_mask = 1.0 - mask
            return inv_mask * line + mask, None

        mask0 = jnp.zeros(self.img_shape[:-1])
        mask_final, _ = jax.lax.scan(accumulate, mask0, lines)
        return mask_final


__all__ = ["VerticalMask"]
