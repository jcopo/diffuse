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
from .base import MRIMaskForwardModel, SPIRAL_RANGE


def _generate_spiral_2D(
    num_samples: int = 1000,
    k_max: float = 1.0,
    FOV: float = 1.0,
    angle_offset: float = 0.0,
    max_angle: Optional[float] = None,
    N: int = 1,
) -> tuple[Array, Array]:
    """Generate 2D spiral k-space trajectory."""
    theta_max = (2 * jnp.pi / N) * k_max * FOV
    if max_angle is not None:
        theta_max = min(theta_max, max_angle)
    theta = jnp.linspace(0, theta_max, num_samples)

    r = (N * theta) / (2 * jnp.pi * FOV)

    kx = jnp.zeros(num_samples * N)
    ky = jnp.zeros(num_samples * N)

    theta_shifted = theta[:, None] + (2 * jnp.pi * jnp.arange(N) / N) + angle_offset
    kx = (r[:, None] * jnp.cos(theta_shifted)).ravel()
    ky = (r[:, None] * jnp.sin(theta_shifted)).ravel()

    return kx, ky


@partial(jax.vmap, in_axes=(0, 0, None, None, None))
def _gaussian_kernel(kx_i: Array, ky_i: Array, x: Array, y: Array, sigma: float = 0.3) -> Array:
    """Gaussian kernel for gridding."""
    distances = ((x - kx_i) ** 2 + (y - ky_i) ** 2) / (2 * sigma**2)
    return jnp.exp(-distances)


@partial(jax.jit, static_argnums=(2,))
def _grid(kx: Array, ky: Array, size: tuple[int, int], sigma: float = 0.3, sharpness: float = 10.0) -> Array:
    """Grid spiral trajectory to Cartesian grid."""
    y, x = jnp.mgrid[0 : size[0], 0 : size[1]]
    y = y.astype(float)
    x = x.astype(float)

    scale = min(size[0], size[1]) / 2 - 1

    kx_scaled = kx * scale + size[1] / 2
    ky_scaled = ky * scale + size[0] / 2

    grid = _gaussian_kernel(kx_scaled, ky_scaled, x, y, sigma).sum(axis=0)

    grid = jax.nn.sigmoid(sharpness * (grid - 1))
    return grid


@dataclass
class SpiralMask(MRIMaskForwardModel):
    """Spiral k-space sampling pattern using regular FFT (not NUFFT)."""

    num_samples: int = 1000
    sigma: float = 0.3
    n_arms: int = 1
    sharpness: float = 10.0
    max_angle: Optional[float] = None
    design_bounds: tuple[float, float] = (2.0, 4.0)

    def sample_design(self, rng_key: PRNGKeyArray) -> Array:
        """Sample spiral design parameters: FOV, k_max, angle_offset."""
        minval, maxval = self.design_bounds
        return jax.random.uniform(rng_key, shape=(3,), minval=minval, maxval=maxval)

    def make(self, xi: Array, rng_key: Optional[PRNGKeyArray] = None) -> Array:
        """Generate spiral mask from design parameters."""
        del rng_key
        xi = jnp.asarray(xi)
        if xi.shape != (3,):
            raise ValueError(f"Expected design parameters of shape (3,), got {xi.shape}")

        xi = jax.nn.sigmoid(xi)
        fov = xi[0]
        k_max = xi[1]
        angle_offset = xi[2]

        kx, ky = _generate_spiral_2D(self.num_samples, k_max, fov, angle_offset, self.max_angle, self.n_arms)
        grid_mask_soft = _grid(kx, ky, self.img_shape[:-1], self.sigma, self.sharpness)
        grid_mask_hard = grid_mask_soft > 0.5

        # Straight-through estimator
        return grid_mask_soft + jax.lax.stop_gradient(grid_mask_hard - grid_mask_soft)


__all__ = ["SpiralMask"]
