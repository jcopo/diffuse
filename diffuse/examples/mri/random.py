# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from .base import MRIMaskForwardModel, RANDOM_SIGMA, RANDOM_SPARSITY


def _rescale(prob: Array, sparsity: float, sigma: float) -> Array:
    mean_val = prob.mean()
    ratio = sparsity / jnp.clip(mean_val, a_min=1e-6)
    beta = (1.0 - sparsity) / jnp.clip(1.0 - mean_val, a_min=1e-6)

    le_hard = ratio < 1.0
    hard_rescale = jnp.where(le_hard, prob * ratio, 1.0 - (1.0 - prob) * beta)

    le_soft = jax.nn.sigmoid((1.0 - ratio) / sigma)
    soft_rescale = le_soft * prob + (1.0 - le_soft) * (1.0 - (1.0 - prob) * beta)
    return soft_rescale + jax.lax.stop_gradient(hard_rescale - soft_rescale)


def _sampler(rng_key: PRNGKeyArray, prob: Array, sparsity: float, sigma: float) -> Array:
    """Differentiable Bernoulli sampler using straight-through estimator."""
    prob = prob / jnp.maximum(prob.max(), 1e-6)
    prob = jax.nn.softplus(prob)
    prob = _rescale(prob, sparsity, sigma)

    uniform = jax.random.uniform(rng_key, prob.shape)
    mask_hard = (uniform < prob).astype(jnp.float32)
    mask_soft = jax.nn.sigmoid((prob - uniform) / sigma)
    return mask_soft + jax.lax.stop_gradient(mask_hard - mask_soft)


@dataclass
class RandomMask(MRIMaskForwardModel):
    """IID random masks governed by per-pixel sampling probabilities."""

    def sample_design(self, rng_key: PRNGKeyArray) -> Array:
        height, width, _ = self.img_shape
        return jax.random.uniform(rng_key, shape=(height, width))

    def projection_design(self, xi: Array) -> Array:
        return jnp.clip(xi, 0.0, 1.0)

    def make(self, xi: Array, rng_key: Optional[PRNGKeyArray] = None) -> Array:
        if rng_key is None:
            raise ValueError("RandomMask.make requires an rng_key for sampling.")

        xi = self.projection_design(xi)
        return _sampler(
            rng_key,
            xi,
            sparsity=RANDOM_SPARSITY[self.data_model],
            sigma=RANDOM_SIGMA[self.data_model],
        )


__all__ = ["RandomMask"]
