from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import ott
import pytest
import scipy as sp

from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, DPSDenoiser, FPSDenoiser, TMPDenoiser
from diffuse.denoisers.denoiser import Denoiser
from examples.gaussian_mixtures.cond_mixture import (
    compute_posterior,
    compute_xt_given_y,
)
from examples.gaussian_mixtures.mixture import (
    cdf_mixtr,
    pdf_mixtr,
    sampler_mixtr,
)
from examples.gaussian_mixtures.plotting import (
    display_trajectories,
    display_trajectories_at_times,
    plot_2d_mixture_and_samples,
    display_2d_trajectories_at_times,
)
from examples.gaussian_mixtures.test_config import (
    CONFIG,
    create_basic_setup,
    create_sde,
    create_timer,
)


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


@pytest.fixture
def test_setup():
    """Single fixture that provides all necessary test configuration and objects"""
    return create_basic_setup()


def create_score_functions(posterior_state, sde):
    """Helper to create pdf, cdf and score functions for the posterior"""

    def pdf(x, t):
        mix_state_t = compute_xt_given_y(posterior_state, sde, t)
        return pdf_mixtr(mix_state_t, x)

    def cdf(x, t):
        mix_state_t = compute_xt_given_y(posterior_state, sde, t)
        return cdf_mixtr(mix_state_t, x)

    def score(x, t):
        return jax.grad(pdf)(x, t) / pdf(x, t)

    return pdf, cdf, score


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



@pytest.mark.parametrize("schedule_name", ["LinearSchedule", "CosineSchedule"])
@pytest.mark.parametrize("integrator_class,integrator_params", CONFIG["integrators"])
@pytest.mark.parametrize("timer_name,timer_fn", CONFIG["timers"])
def test_backward_sde_conditional_mixture(
    test_setup,
    plot_if_enabled,
    integrator_class,
    integrator_params,
    schedule_name,
    timer_name,
    timer_fn,
):
    # Unpack setup
    key = test_setup["key"]
    key_samples, key_meas, key_gen = jax.random.split(key, 3)
    mix_state = test_setup["mix_state"]
    forward_model = test_setup["forward_model"]
    A = test_setup["A"]
    t_init = test_setup["t_init"]
    t_final = test_setup["t_final"]
    n_steps = test_setup["n_steps"]
    n_samples = test_setup["n_samples"]
    sigma_y = test_setup["sigma_y"]
    perct = test_setup["perct"]

    # Create SDE and timer
    sde = create_sde(schedule_name, t_final)
    timer = create_timer(timer_name, n_steps, t_final)


    # Generate observation
    x_star = sampler_mixtr(key_samples, mix_state, 1)[0]
    y = forward_model.measure(key_meas, x_star)

    # Compute theoretical posterior and score functions
    posterior_state = compute_posterior(mix_state, y, A, sigma_y)
    pdf, cdf, score = create_score_functions(posterior_state, sde)

    # Setup denoising process with timer
    integrator = integrator_class(sde=sde, timer=timer, **integrator_params)
    denoise = Denoiser(integrator=integrator, sde=sde, score=score, x0_shape=x_star.shape)

    # Generate samples
    state, hist_position = denoise.generate(key_gen, n_steps, n_samples)
    hist_position = hist_position.squeeze()

    # Visualization
    space = jnp.linspace(-10, 10, 100)
    plot_title = f"{integrator_class.__name__} (Timer: {timer_name}, Schedule: {schedule_name})"

    if test_setup["d"] == 1:
        plot_if_enabled(lambda: display_trajectories(hist_position, 100, title=plot_title))
        plot_if_enabled(
            lambda: display_trajectories_at_times(
                hist_position,
                timer,
                n_steps,
                space,
                perct,
                lambda x, t: pdf(x, t_final - t),
                title=plot_title,
            )
        )
    elif test_setup["d"] == 2:
        plot_if_enabled(lambda: plot_2d_mixture_and_samples(posterior_state, hist_position, plot_title))
        plot_if_enabled(
            lambda: display_2d_trajectories_at_times(
                hist_position,
                timer,
                n_steps,
                perct,
                lambda x, t: pdf(x, t_final - t),
                title=plot_title,
                sde=sde,
            )
        )

    # Compute Wasserstein distance
    wasserstein_distance, _ = ott.tools.sliced.sliced_wasserstein(
        state.integrator_state.position,
        posterior_state.means,
    )

    # Create a composite key for the results
    result_key = f"{integrator_class.__name__}_{schedule_name}_{timer_name}"
    wasserstein_results[result_key].append(float(wasserstein_distance))

    # assert wasserstein_distance < 0.1
    assert wasserstein_distance.item() < 0.1

    # # Validate distributions
    # validate_distributions(
    #     hist_position,
    #     timer,
    #     n_steps,
    #     perct,
    #     lambda x, t: cdf(x, t_final - t),
    #     key=key,
    #     method=integrator_class,
    #     forward=False
    # )


@pytest.mark.parametrize("schedule_name", ["LinearSchedule", "CosineSchedule"])
@pytest.mark.parametrize("integrator_class,integrator_params", CONFIG["integrators"])
@pytest.mark.parametrize("timer_name,timer_fn", CONFIG["timers"])
@pytest.mark.parametrize("denoiser_class", CONFIG["cond_denoisers"])
def test_backward_CondDenoisers(
    test_setup,
    plot_if_enabled,
    integrator_class,
    integrator_params,
    schedule_name,
    timer_name,
    timer_fn,
    denoiser_class,
):
    # Unpack setup
    key = test_setup["key"]
    key_samples, key_meas, key_gen = jax.random.split(key, 3)
    mix_state = test_setup["mix_state"]
    forward_model = test_setup["forward_model"]
    A = test_setup["A"]
    t_init = test_setup["t_init"]
    t_final = test_setup["t_final"]
    n_steps = test_setup["n_steps"]
    n_samples = test_setup["n_samples"]
    sigma_y = test_setup["sigma_y"]
    perct = test_setup["perct"]

    # Create SDE and timer
    sde = create_sde(schedule_name, t_final)
    timer = create_timer(timer_name, n_steps, t_final)

    # Generate observation
    x_star = sampler_mixtr(key_samples, mix_state, 1)[0]
    y = forward_model.measure(key_meas, x_star)

    # Compute theoretical posterior and score functions
    posterior_state = compute_posterior(mix_state, y, A, sigma_y)
    pdf, cdf, score = create_score_functions(posterior_state, sde)

    # Setup denoising process with timer and integrator parameters
    integrator = integrator_class(sde=sde, timer=timer, **integrator_params)
    denoise = denoiser_class(
        integrator=integrator,
        sde=sde,
        score=score,
        forward_model=forward_model,
        x0_shape=x_star.shape,
    )

    # Create measurement state with mask
    mask = jnp.ones_like(y)
    measurement_state = MeasurementState(y=y, mask_history=mask)

    # Generate samples
    state, hist_position = denoise.generate(key_gen, measurement_state, n_steps, n_samples)
    hist_position = hist_position.squeeze()

    # Visualization
    space = jnp.linspace(-10, 10, 100)
    plot_title = f"{denoiser_class.__name__} (Integrator: {integrator_class.__name__}, Timer: {timer_name}, Schedule: {schedule_name})"

    if test_setup["d"] == 1:
        plot_if_enabled(lambda: display_trajectories(hist_position, 100, title=plot_title))
        plot_if_enabled(
            lambda: display_trajectories_at_times(
                hist_position,
                timer,
                n_steps,
                space,
                perct,
                lambda x, t: pdf(x, t_final - t),
                title=plot_title,
            )
        )
    elif test_setup["d"] == 2:
        plot_if_enabled(lambda: plot_2d_mixture_and_samples(posterior_state, hist_position, plot_title))
        print(hist_position.shape)
        plot_if_enabled(
            lambda: display_2d_trajectories_at_times(
                hist_position,
                timer,
                n_steps,
                perct,
                lambda x, t: pdf(x, t_final - t),
                title=plot_title,
                sde=sde,
            )
        )

    # Compute Wasserstein distance
    wasserstein_distance, _ = ott.tools.sliced.sliced_wasserstein(
        state.integrator_state.position,
        x_star[None, :],
    )

    # Create a composite key for the results
    result_key = f"{denoiser_class.__name__}_{integrator_class.__name__}_{schedule_name}_{timer_name}"
    wasserstein_results[result_key].append(float(wasserstein_distance))

    print(f"\nWasserstein distance for {result_key}: {float(wasserstein_distance):.6f}")

    assert wasserstein_distance.item() < 0.1
