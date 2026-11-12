# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from .base import MRIMaskForwardModel


@dataclass
class FullMask(MRIMaskForwardModel):
    """Full k-space sampling (no masking)."""

    def sample_design(self, rng_key: PRNGKeyArray) -> Array:
        """No design parameters needed for full sampling."""
        return jnp.array([1.0])

    def make(self, xi: Array, rng_key: Optional[PRNGKeyArray] = None) -> Array:
        """Return a full mask (all ones)."""
        return jnp.ones(self.img_shape[:-1])


__all__ = ["FullMask"]
