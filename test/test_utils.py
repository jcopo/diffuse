# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Common utilities for diffusion tests.

This module provides shared utilities for plotting, validation, and metrics
across different test files to reduce code duplication.
"""

from collections import defaultdict
from typing import List

import jax.numpy as jnp
import warnings

from diffuse.examples.gaussian_mixtures.plotting import (
    display_trajectories,
    display_trajectories_at_times,
    plot_2d_mixture_and_samples,
    display_2d_trajectories_at_times,
)


# Global dict to store MMD results across all tests
mmd_results = defaultdict(list)


def create_plots(test_config, hist_position, plot_title, plot_if_enabled):
    """Create plots for test configuration.

    Unified plotting function that handles both 1D and 2D cases.

    Args:
        test_config: Test configuration object
        hist_position: History of positions from sampling
        plot_title: Title for the plots
        plot_if_enabled: Plotting fixture from conftest
    """
    if test_config.d == 1:
        plot_if_enabled(lambda: display_trajectories(hist_position, 100, title=plot_title))
        plot_if_enabled(
            lambda: display_trajectories_at_times(
                hist_position,
                test_config.timer,
                test_config.n_steps,
                test_config.space,
                test_config.perct,
                lambda x, t: test_config.pdf(x, test_config.t_final - t),
                title=plot_title,
            )
        )
    elif test_config.d == 2:
        # For conditional tests, use posterior_state if available, otherwise mix_state
        state_for_plotting = getattr(test_config, "posterior_state", test_config.mix_state)
        plot_if_enabled(lambda: plot_2d_mixture_and_samples(state_for_plotting, hist_position, plot_title))
        plot_if_enabled(
            lambda: display_2d_trajectories_at_times(
                hist_position,
                test_config.timer,
                test_config.n_steps,
                test_config.perct,
                lambda x, t: test_config.pdf(x, test_config.t_final - t),
                title=plot_title,
                sde=test_config.model,
            )
        )


def rbf_kernel(x, y, gamma):
    """
    Compute RBF (Gaussian) kernel between two sets of samples.

    Args:
        x: First sample set (n_samples, d)
        y: Second sample set (m_samples, d)
        gamma: Kernel bandwidth parameter

    Returns:
        Kernel matrix (n_samples, m_samples)
    """
    # Compute squared Euclidean distances
    x_norm = jnp.sum(x**2, axis=1, keepdims=True)
    y_norm = jnp.sum(y**2, axis=1, keepdims=True)
    dist_sq = x_norm + y_norm.T - 2 * jnp.dot(x, y.T)

    # Apply RBF kernel
    return jnp.exp(-gamma * dist_sq)


def polynomial_kernel(x, y, degree=3, gamma=1.0, coef0=1.0):
    """
    Compute polynomial kernel between two sets of samples.

    Args:
        x: First sample set (n_samples, d)
        y: Second sample set (m_samples, d)
        degree: Polynomial degree
        gamma: Scaling parameter
        coef0: Constant term

    Returns:
        Kernel matrix (n_samples, m_samples)
    """
    dot_product = jnp.dot(x, y.T)
    return (gamma * dot_product + coef0) ** degree


def linear_kernel(x, y):
    """
    Compute linear kernel between two sets of samples.

    Args:
        x: First sample set (n_samples, d)
        y: Second sample set (m_samples, d)

    Returns:
        Kernel matrix (n_samples, m_samples)
    """
    return jnp.dot(x, y.T)


def median_heuristic(x, y):
    """
    Compute gamma parameter using median heuristic.

    Args:
        x: First sample set (n_samples, d)
        y: Second sample set (m_samples, d)

    Returns:
        gamma: Bandwidth parameter
    """
    # Combine samples
    combined = jnp.concatenate([x, y], axis=0)

    # Compute pairwise squared distances using broadcasting
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2<x_i, x_j>
    norms = jnp.sum(combined**2, axis=1, keepdims=True)
    dist_sq = norms + norms.T - 2 * jnp.dot(combined, combined.T)

    # Extract upper triangular part (excluding diagonal) to get unique pairwise distances
    n = combined.shape[0]
    mask = jnp.triu(jnp.ones((n, n)), k=1)  # Upper triangular mask excluding diagonal
    pairwise_dists = jnp.sqrt(jnp.maximum(dist_sq * mask, 0.0))  # Ensure non-negative before sqrt

    # Get non-zero distances (from upper triangular part)
    valid_dists = pairwise_dists[mask > 0]
    median_dist = jnp.median(valid_dists)

    # Gamma = 1 / (2 * median_distance^2)
    return 1.0 / (2.0 * median_dist**2 + 1e-8)  # Add small epsilon for numerical stability


def compute_mmd(samples_x, samples_y, kernel="rbf", gamma=None, **kernel_kwargs):
    """
    Compute Maximum Mean Discrepancy (MMD) between two sample sets.

    MMD²(P,Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    where x,x' ~ P and y,y' ~ Q, and k is a kernel function.

    Args:
        samples_x: First sample set (n_samples, d)
        samples_y: Second sample set (m_samples, d)
        kernel: Kernel type ('rbf', 'poly', 'linear')
        gamma: Kernel parameter (auto-select if None)
        **kernel_kwargs: Additional kernel parameters

    Returns:
        MMD value (float)
    """
    samples_x = jnp.asarray(samples_x)
    samples_y = jnp.asarray(samples_y)

    # Ensure samples are 2D
    if samples_x.ndim == 1:
        samples_x = samples_x.reshape(-1, 1)
    if samples_y.ndim == 1:
        samples_y = samples_y.reshape(-1, 1)

    # Auto-select gamma using median heuristic for RBF kernel
    if kernel == "rbf" and gamma is None:
        gamma = median_heuristic(samples_x, samples_y)

    # Select kernel function
    if kernel == "rbf":
        def kernel_func(x, y):
            return rbf_kernel(x, y, gamma)
    elif kernel == "poly" or kernel == "polynomial":
        degree = kernel_kwargs.get("degree", 3)
        gamma = gamma or 1.0
        coef0 = kernel_kwargs.get("coef0", 1.0)
        def kernel_func(x, y):
            return polynomial_kernel(x, y, degree, gamma, coef0)
    elif kernel == "linear":
        kernel_func = linear_kernel
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Compute kernel matrices
    K_xx = kernel_func(samples_x, samples_x)
    K_yy = kernel_func(samples_y, samples_y)
    K_xy = kernel_func(samples_x, samples_y)

    # Compute unbiased MMD estimate
    n = samples_x.shape[0]
    m = samples_y.shape[0]

    # Remove diagonal elements for unbiased estimate
    K_xx_no_diag = K_xx - jnp.diag(jnp.diag(K_xx))
    K_yy_no_diag = K_yy - jnp.diag(jnp.diag(K_yy))

    # MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    term1 = jnp.sum(K_xx_no_diag) / (n * (n - 1))
    term2 = 2 * jnp.mean(K_xy)
    term3 = jnp.sum(K_yy_no_diag) / (m * (m - 1))

    mmd_squared = term1 - term2 + term3

    # Return MMD (take square root, ensure non-negative)
    return jnp.sqrt(jnp.maximum(mmd_squared, 0.0))


def compute_and_store_mmd(test_config, state, reference_data, result_key_parts: List[str], print_result: bool = False, kernel: str = "rbf") -> float:
    """Compute Maximum Mean Discrepancy (MMD) and store results.

    Args:
        test_config: Test configuration object
        state: Final state from sampling
        reference_data: Reference samples to compare against
        result_key_parts: Parts to create result key for storage
        print_result: Whether to print the result
        kernel: Kernel type for MMD computation ('rbf', 'poly', 'linear')

    Returns:
        MMD value as float
    """
    generated_samples = state.integrator_state.position
    mmd_distance = compute_mmd(generated_samples, reference_data, kernel=kernel)

    result_key = "_".join(result_key_parts)
    mmd_results[result_key].append(float(mmd_distance))

    if print_result:
        print(f"\nMMD distance for {result_key}: {float(mmd_distance):.6f}")

    return float(mmd_distance)


def print_mmd_summary():
    """Print summary of all MMD distances collected during tests."""
    if mmd_results:
        print("\nMMD Distance Summary:")
        print("-" * 80)
        print(f"{'Configuration':<50} {'Distances':<10}")
        print("-" * 80)
        for config, distances in mmd_results.items():
            print(f"{config:<50} {distances}")


def assert_mmd_threshold(distance: float, threshold: float = 0.05):
    """Assert that MMD distance is below threshold.

    Raises a warning if distance is between 0.01 and threshold.
    """
    if 0.02 < distance < threshold:
        warnings.warn(f"MMD distance {distance} is close to the threshold {threshold}.", UserWarning)
    assert distance < threshold, f"MMD distance {distance} exceeds threshold {threshold}"
