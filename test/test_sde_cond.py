from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import ott
import pytest
import scipy as sp

from examples.gaussian_mixtures.plotting import (
    display_trajectories,
    display_trajectories_at_times,
    plot_2d_mixture_and_samples,
    display_2d_trajectories_at_times,
)
from .config import get_parametrized_configs, get_conditional_configs


# Simple dict to store results
wasserstein_results = defaultdict(list)


@pytest.fixture(autouse=True)
def collect_wasserstein():
    """Automatically collect Wasserstein distances from all tests"""
    yield

    # At the end of all tests, print summary
    if wasserstein_results:
        print("\nWasserstein Distance Summary:")
        print("-" * 80)
        print(f"{'Configuration':<50} {'Distances':<10}")
        print("-" * 80)
        for config, distances in wasserstein_results.items():
            print(f"{config:<50} {distances}")


def validate_distributions(position, timer, n_steps, perct, cdf, key=None, method=None, forward=True):
    """Helper function to validate sample distributions against theoretical ones."""
    for i, x in enumerate(perct):
        k = int(x * n_steps)
        t = timer(0) - timer(k + 1)
        samples = position[:, k].squeeze()
        sample_indices = jax.random.choice(key, samples.shape[0], shape=(300,))
        samples = samples[sample_indices]

        if key is not None:  # Use subset of samples if key provided
            sample_indices = jax.random.choice(key, samples.shape[0], shape=(200,))
            samples = samples[sample_indices]

        # time = t if forward else timer(0) - t
        ks_statistic, p_value = sp.stats.kstest(
            np.array(samples),
            lambda x: cdf(x, t),
        )

        error_msg = f"Sample distribution does not match theoretical (p-value: {p_value}, t: {t}, k: {k})"
        if method:
            error_msg = f"Sample distribution does not match theoretical (method: {method.__name__}, p-value: {p_value}, t: {t}, k: {k})"

        assert p_value > 0.01, error_msg


@pytest.mark.parametrize("test_config", get_parametrized_configs(), indirect=True)
def test_backward_sde_conditional_mixture(test_config, plot_if_enabled):
    """Test backward SDE for conditional mixture - everything from config."""
    # Generate random keys
    key_gen = jax.random.split(test_config.key, 1)[0]

    # Use pre-configured denoiser directly
    state, hist_position = test_config.basic_denoiser.generate(key_gen, test_config.n_steps, test_config.n_samples)
    hist_position = hist_position.squeeze()

    # Visualization
    plot_title = f"{test_config.integrator_class.__name__} (Timer: {test_config.timer_name}, Schedule: {test_config.schedule_name})"

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
        plot_if_enabled(lambda: plot_2d_mixture_and_samples(test_config.posterior_state, hist_position, plot_title))
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

    # Compute Wasserstein distance
    wasserstein_distance, _ = ott.tools.sliced.sliced_wasserstein(
        state.integrator_state.position, test_config.posterior_state.means
    )

    # Store results
    result_key = f"{test_config.integrator_class.__name__}_{test_config.schedule_name}_{test_config.timer_name}"
    wasserstein_results[result_key].append(float(wasserstein_distance))

    assert wasserstein_distance.item() < 0.1


@pytest.mark.parametrize("test_config", get_conditional_configs(), indirect=True)
def test_backward_CondDenoisers(test_config, plot_if_enabled):
    """Test conditional denoisers - everything from config."""
    # Generate random keys
    key_meas, key_gen = jax.random.split(test_config.key, 2)

    # Generate observation for measurement state
    x_star = test_config.generate_samples(key_meas, 1)[0]
    y = test_config.forward_model.measure(key_meas, x_star)
    measurement_state = test_config.create_measurement_state(y)

    # Use pre-configured denoiser directly
    state, hist_position = test_config.denoiser.generate(
        key_gen, measurement_state, test_config.n_steps, test_config.n_samples
    )
    hist_position = hist_position.squeeze()

    # Visualization
    plot_title = f"{test_config.denoiser_class.__name__} (Integrator: {test_config.integrator_class.__name__}, Timer: {test_config.timer_name}, Schedule: {test_config.schedule_name})"

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
        plot_if_enabled(lambda: plot_2d_mixture_and_samples(test_config.posterior_state, hist_position, plot_title))
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

    # Compute Wasserstein distance
    wasserstein_distance, _ = ott.tools.sliced.sliced_wasserstein(state.integrator_state.position, x_star[None, :])

    # Store results
    result_key = f"{test_config.denoiser_class.__name__}_{test_config.integrator_class.__name__}_{test_config.schedule_name}_{test_config.timer_name}"
    wasserstein_results[result_key].append(float(wasserstein_distance))

    print(f"\nWasserstein distance for {result_key}: {float(wasserstein_distance):.6f}")

    assert wasserstein_distance.item() < 0.1
