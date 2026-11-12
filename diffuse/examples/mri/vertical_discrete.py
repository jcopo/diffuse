# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from .base import MRIMaskForwardModel


def _vertical_line_discrete(index: Array, shape: tuple[int, int]) -> Array:
    height, width = shape
    idx = jnp.clip(index.astype(jnp.int32), 0, width - 1)
    mask = jnp.zeros((height, width), dtype=jnp.float32)
    return mask.at[:, idx].set(1.0)


@dataclass
class VerticalDiscreteMask(MRIMaskForwardModel):
    """Discrete vertical line sampling (ADS variant)."""

    num_lines: int = 1

    def sample_design(self, rng_key: PRNGKeyArray) -> Array:
        width = self.img_shape[1]
        return jax.random.randint(rng_key, shape=(self.num_lines,), minval=0, maxval=width)

    def make(self, xi: Array, rng_key: PRNGKeyArray | None = None) -> Array:
        del rng_key
        xi = jnp.asarray(xi, dtype=jnp.int32)
        if xi.shape != (self.num_lines,):
            raise ValueError(f"Expected integer design parameters of shape ({self.num_lines},), got {xi.shape}")

        lines = jax.vmap(_vertical_line_discrete, in_axes=(0, None))(xi, self.img_shape[:-1])

        def accumulate(mask, line):
            inv_mask = 1.0 - mask
            return inv_mask * line + mask, None

        mask0 = jnp.zeros(self.img_shape[:-1])
        mask_final, _ = jax.lax.scan(accumulate, mask0, lines)
        return mask_final


__all__ = ["VerticalDiscreteMask"]
