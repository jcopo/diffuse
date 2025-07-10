"""Common utilities for diffusion tests.

This module provides shared utilities for plotting, validation, and metrics
across different test files to reduce code duplication.
"""

from collections import defaultdict
from typing import Callable, List, Optional, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import ott

from examples.gaussian_mixtures.mixture import cdf_t, sampler_mixtr, rho_t
from examples.gaussian_mixtures.plotting import (
    display_trajectories,
    display_trajectories_at_times,
    plot_2d_mixture_and_samples,
    display_2d_trajectories_at_times,
)


# Global dict to store Wasserstein results across all tests
wasserstein_results = defaultdict(list)


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
                sde=test_config.sde,
            )
        )


def validate_distributions(
    position,
    timer,
    n_steps: int,
    perct: List[float],
    cdf_func: Callable,
    key: Optional[jax.random.PRNGKey] = None,
    method: Optional[Any] = None,
    t_final: float = 1.0,
    forward: bool = True,
    p_threshold: float = 0.01,
):
    """Validate sample distributions against theoretical ones using KS test.

    Args:
        position: Array of positions to validate
        timer: Timer object that converts steps to time
        n_steps: Number of steps
        perct: Percentiles to check
        cdf_func: Cumulative distribution function to test against
        key: Optional PRNG key for sampling (used in backward validation)
        method: Optional method name for error messages
        t_final: Final time (defaults to 1.0 for forward process)
        forward: Whether this is forward or backward process
        p_threshold: P-value threshold for test acceptance
    """
    for i, x in enumerate(perct):
        k = int(x * n_steps) + (1 if key is None else 0)  # +1 only for forward process
        t = t_final - timer(k + 1)

        if key is not None:  # backward process case
            samples = position[:, k].squeeze()
            sample_indices = jax.random.choice(key, samples.shape[0], shape=(300,))
            samples = samples[sample_indices]
        else:  # forward process case
            samples = np.array(position[:, k].squeeze())

        time = t if forward else t_final - t
        _, p_value = sp.stats.kstest(
            samples,
            lambda x: cdf_func(x, time),
        )

        error_msg = f"Sample distribution does not match theoretical (p-value: {p_value}, t: {t}, k: {k})"
        if method:
            error_msg = f"Sample distribution does not match theoretical (method: {method.__name__}, p-value: {p_value}, t: {t}, k: {k})"

        assert p_value > p_threshold, error_msg


def compute_and_store_wasserstein(
    test_config, state, reference_data, result_key_parts: List[str], print_result: bool = False
) -> float:
    """Compute Wasserstein distance and store results.

    Args:
        test_config: Test configuration object
        state: Final state from sampling
        reference_data: Reference samples to compare against
        result_key_parts: Parts to create result key for storage
        print_result: Whether to print the result

    Returns:
        Wasserstein distance as float
    """
    wasserstein_distance, _ = ott.tools.sliced.sliced_wasserstein(state.integrator_state.position, reference_data)

    result_key = "_".join(result_key_parts)
    wasserstein_results[result_key].append(float(wasserstein_distance))

    if print_result:
        print(f"\nWasserstein distance for {result_key}: {float(wasserstein_distance):.6f}")

    return float(wasserstein_distance)


def validate_conditional_distributions(
    position,
    timer,
    n_steps: int,
    perct: List[float],
    cdf,
    key: jax.random.PRNGKey,
    method: Optional[Any] = None,
    t_final: float = 1.0,
):
    """Validate conditional distributions - wrapper for backward validation."""
    for i, x in enumerate(perct):
        k = int(x * n_steps)
        t = timer(0) - timer(k + 1)
        samples = position[:, k].squeeze()
        sample_indices = jax.random.choice(key, samples.shape[0], shape=(300,))
        samples = samples[sample_indices]

        # Use subset of samples for faster testing
        if len(samples) > 200:
            sample_indices = jax.random.choice(key, samples.shape[0], shape=(200,))
            samples = samples[sample_indices]

        ks_statistic, p_value = sp.stats.kstest(
            np.array(samples),
            lambda x: cdf(x, t),
        )

        error_msg = f"Sample distribution does not match theoretical (p-value: {p_value}, t: {t}, k: {k})"
        if method:
            error_msg = f"Sample distribution does not match theoretical (method: {method.__name__}, p-value: {p_value}, t: {t}, k: {k})"

        assert p_value > 0.01, error_msg


def print_wasserstein_summary():
    """Print summary of all Wasserstein distances collected during tests."""
    if wasserstein_results:
        print("\nWasserstein Distance Summary:")
        print("-" * 80)
        print(f"{'Configuration':<50} {'Distances':<10}")
        print("-" * 80)
        for config, distances in wasserstein_results.items():
            print(f"{config:<50} {distances}")


def assert_wasserstein_threshold(distance: float, threshold: float = 0.1):
    """Assert that Wasserstein distance is below threshold."""
    assert distance < threshold, f"Wasserstein distance {distance} exceeds threshold {threshold}"


def plot_debug(generated_samples, reference_samples, test_config, plot_title: str, plot_if_enabled):
    """Minimal debug plot with mixture PDF overlay."""
    def create_debug_plot():
        fig, ax = plt.subplots(figsize=(8, 6))

        # Get final samples
        gen = generated_samples[-1] if generated_samples.ndim == 3 else generated_samples
        ref = reference_samples[-1] if reference_samples.ndim == 3 else reference_samples

        if test_config.d == 1:
            # 1D: histograms + PDF
            range_val = jnp.max(jnp.abs(jnp.concatenate([gen.flatten(), ref.flatten()]))) * 1.1
            bins = jnp.linspace(-range_val, range_val, 40)
            ax.hist(ref.flatten(), bins=bins, alpha=0.5, color="blue", density=True, label="Reference")
            ax.hist(gen.flatten(), bins=bins, alpha=0.5, color="red", density=True, label="Generated")

            # Plot mixture PDF
            if hasattr(test_config, 'posterior_state') or hasattr(test_config, 'mix_state'):
                state = getattr(test_config, 'posterior_state', test_config.mix_state)
                x = jnp.linspace(-range_val, range_val, 200)
                pdf_vals = jax.vmap(lambda xi: test_config.pdf(xi, test_config.t_final))(x)
                ax.plot(x, pdf_vals, 'k-', linewidth=2, label='Mixture PDF')

        elif test_config.d == 2:
            # 2D: scatter + contours
            range_val = jnp.max(jnp.abs(jnp.concatenate([gen, ref]))) * 1.1
            ax.scatter(ref[:, 0], ref[:, 1], alpha=0.5, c="blue", s=10, label="Reference")
            ax.scatter(gen[:, 0], gen[:, 1], alpha=0.5, c="red", s=10, label="Generated")

            # Plot mixture PDF contours
            if hasattr(test_config, 'posterior_state') or hasattr(test_config, 'mix_state'):
                x = jnp.linspace(-range_val, range_val, 50)
                X, Y = jnp.meshgrid(x, x)
                points = jnp.stack([X.flatten(), Y.flatten()], axis=1)
                # pdf = lambda x, t: rho_t(x, t, test_config.posterior_state, test_config.sde)
                pdf_vals = jax.vmap(lambda p: test_config.pdf(p, 0.0))(points).reshape(X.shape)
                ax.contour(X, Y, pdf_vals, levels=8, colors='black', alpha=0.7, linewidths=1)
            ax.set_aspect('equal')

        ax.legend()
        ax.set_title(plot_title)
        plt.tight_layout()
        return fig

    plot_if_enabled(create_debug_plot)
