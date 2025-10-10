# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
import jax
import jax.numpy as jnp
from diffuse.examples.gaussian_mixtures.mixture import MixState


def init_simple_mixture(key, d=1, n_components=3):
    """
    Create a simple Gaussian mixture with random components.

    Args:
        key: JAX random key
        d: Dimensionality
        n_components: Number of mixture components

    Returns:
        MixState: Initialized mixture state
    """
    keys = jax.random.split(key, 3)

    # Random means in reasonable range
    means = jax.random.uniform(keys[0], (n_components, d), minval=-3, maxval=3)

    # Small random covariances
    if d == 1:
        covs = 0.1 * jax.random.uniform(keys[1], (n_components, 1, 1), minval=0.5, maxval=1.5)
    else:
        # For multivariate case, use identity matrices with small random scaling
        identity = jnp.eye(d)
        scales = 0.1 * jax.random.uniform(keys[1], (n_components, 1, 1), minval=0.5, maxval=1.5)
        covs = scales * identity[None, :, :]

    # Normalized random weights
    weights = jax.random.uniform(keys[2], (n_components,))
    weights = weights / jnp.sum(weights)

    return MixState(means, covs, weights)


def init_grid_mixture(key, d=2, grid_size=5):
    """
    Create a mixture with components arranged on a grid (for high-dimensional cases).

    Args:
        key: JAX random key
        d: Dimensionality
        grid_size: Grid size (total components = grid_size^2)

    Returns:
        MixState: Initialized mixture state
    """
    n_components = grid_size * grid_size

    # Create grid positions
    grid_1d = jnp.linspace(-2, 2, grid_size)
    grid_positions = jnp.array([(i, j) for i in grid_1d for j in grid_1d])

    # Extend to higher dimensions by tiling
    if d > 2:
        repeats = (d + 1) // 2
        means = jnp.tile(grid_positions, (1, repeats))[:, :d]
    else:
        means = grid_positions

    # Identity covariance matrices
    covs = jnp.repeat(jnp.eye(d)[None, :, :], n_components, axis=0)

    # Random weights
    weights = jax.random.uniform(key, (n_components,))
    weights = weights / jnp.sum(weights)

    return MixState(means, covs, weights)


def init_fixed_mixture(d=2):
    """
    Create a fixed mixture for reproducible testing.

    Args:
        d: Dimensionality

    Returns:
        MixState: Fixed mixture state
    """
    if d == 1:
        means = jnp.array([[-2.0], [0.0], [2.0]])
        covs = jnp.array([[[0.5]], [[0.3]], [[0.7]]])
    elif d == 2:
        means = jnp.array([[-1.0, -1.0], [1.0, 1.0], [2.0, -2.0]])
        covs = jnp.array(
            [
                [[0.5, 0.1], [0.1, 0.5]],
                [[0.7, -0.1], [-0.1, 0.7]],
                [[0.3, 0.0], [0.0, 1.0]],
            ]
        )
    else:
        # For higher dimensions, extend 2D case
        means_2d = jnp.array([[-1.0, -1.0], [1.0, 1.0], [2.0, -2.0]])
        padding = jnp.zeros((3, d - 2))
        means = jnp.concatenate([means_2d, padding], axis=1)

        # Identity covariance for extra dimensions
        covs = jnp.repeat(jnp.eye(d)[None, :, :], 3, axis=0)

    weights = jnp.array([0.3, 0.4, 0.3])

    return MixState(means, covs, weights)


def init_bimodal_setup(key, d=2):
    """
    Create mixture and observation setup designed to produce bimodal posterior.

    Args:
        key: JAX random key
        d: Dimensionality (must be 2)

    Returns:
        Tuple of (mix_state, A, y_target, sigma_y)
    """
    if d != 2:
        raise ValueError("Bimodal setup currently only supports d=2")

    # Create a mixture with very well-separated components in 4 quadrants
    means = jnp.array(
        [
            [-8.0, -8.0],  # Bottom-left quadrant
            [8.0, -8.0],  # Bottom-right quadrant
            [-8.0, 8.0],  # Top-left quadrant
            [8.0, 8.0],  # Top-right quadrant
        ]
    )

    # Wide covariances for spread modes
    covs = jnp.array(
        [
            [[1.5, 0.0], [0.0, 1.5]],  # Bottom-left
            [[1.5, 0.0], [0.0, 1.5]],  # Bottom-right
            [[1.5, 0.0], [0.0, 1.5]],  # Top-left
            [[1.5, 0.0], [0.0, 1.5]],  # Top-right
        ]
    )

    weights = jnp.array([0.25, 0.25, 0.25, 0.25])  # Equal weights
    mix_state = MixState(means, covs, weights)

    # Design observation matrix to create multimodal posterior
    # Use a constraint that measures the sum of squares (distance from origin)
    A = jnp.array([[1.0, 1.0]])  # Measure x + y

    # Target observation of zero means we want points where x + y = 0
    # This creates a diagonal line that intersects different quadrants
    y_target = jnp.array([0.0])

    # Moderate noise to allow multiple modes to contribute
    sigma_y = 1.0

    return mix_state, A, y_target, sigma_y


def init_circular_setup(key, d=2, n_components=6, radius=1.0):
    """
    Create mixture and observation setup with components arranged in a circle.

    Args:
        key: JAX random key
        d: Dimensionality (must be 2)
        n_components: Number of components arranged in circle
        radius: Radius of the circle

    Returns:
        Tuple of (mix_state, A, y_target, sigma_y)
    """
    if d != 2:
        raise ValueError("Circular setup currently only supports d=2")

    keys = jax.random.split(key, 2)

    # Create circular arrangement with very small radius for overlapping components
    angles = jnp.linspace(0, 2 * jnp.pi, n_components, endpoint=False)
    means = jnp.stack([radius * jnp.cos(angles), radius * jnp.sin(angles)], axis=1)

    # Very large covariances to create significant overlap and saddle/camel back shape
    cov_scale = 1.2
    covs = cov_scale * jnp.repeat(jnp.eye(d)[None, :, :], n_components, axis=0)

    # Equal weights with small random perturbation
    base_weight = 1.0 / n_components
    perturbation = 0.1 * jax.random.uniform(keys[1], (n_components,), minval=-1, maxval=1)
    weights = base_weight + perturbation * base_weight
    weights = weights / jnp.sum(weights)  # Renormalize

    mix_state = MixState(means, covs, weights)

    # Design observation matrix to create interesting posterior
    # Use a weaker constraint to avoid skinny posterior
    A = jnp.array([[0.3, 0.7]])  # Measure weighted combination of x and y

    # Target observation creates a diagonal line constraint
    y_target = jnp.array([0.0])

    # Higher noise to make posterior covariance less skinny
    sigma_y = 2.0

    return mix_state, A, y_target, sigma_y
