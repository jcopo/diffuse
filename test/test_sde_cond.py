from collections import defaultdict
from functools import partial
from test.test_sde_mixture import display_trajectories_at_times

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import ott
import pytest
import scipy as sp

from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, DPSDenoiser, FPSDenoiser, TMPDenoiser
from diffuse.denoisers.denoiser import Denoiser
from diffuse.diffusion.sde import SDE, CosineSchedule, LinearSchedule
from diffuse.integrator.deterministic import (
    DDIMIntegrator,
    DPMpp2sIntegrator,
    EulerIntegrator,
    HeunIntegrator,
)
from diffuse.integrator.stochastic import EulerMaruyamaIntegrator
from diffuse.timer.base import DDIMTimer, HeunTimer, VpTimer
from examples.gaussian_mixtures.cond_mixture import (
    compute_posterior,
    compute_xt_given_y,
    init_gaussian_mixture,
)
from examples.gaussian_mixtures.forward_models.matrix_product import MatrixProduct
from examples.gaussian_mixtures.mixture import (
    cdf_mixtr,
    display_trajectories,
    pdf_mixtr,
    sampler_mixtr,
)

# Global configurations
CONFIG = {
    "schedules": {
        "LinearSchedule": {
            "params": {"b_min": 0.1, "b_max": 20.0, "t0": 0.001, "T": 1.0},
            "class": LinearSchedule
        },
        "CosineSchedule": {
            "params": {"b_min": 0.1, "b_max": 20.0, "t0": 0.001, "T": 1.0},
            "class": CosineSchedule
        }
    },
    "integrators": [
        EulerMaruyamaIntegrator,
        DDIMIntegrator,
        HeunIntegrator,
        DPMpp2sIntegrator,
        EulerIntegrator
    ],
    "space": {
        "t_init": 0.001,
        "t_final": 1.0,
        "n_samples": 5000,
        "n_steps": 300,
        "d": 1,  # dimensionality
        "sigma_y": 0.01  # observation noise
    },
    "percentiles": [0.0, 0.05, 0.1, 0.3, 0.6, 0.7, 0.73, 0.75, 0.8, 0.9],
    "timers": [
        ("vp", lambda n_steps, t_final: VpTimer(n_steps=n_steps, eps=0.001, tf=t_final)),
        ("heun", lambda n_steps, t_final: HeunTimer(n_steps=n_steps, rho=7.0, sigma_min=0.002, sigma_max=1.0)),
    ],
}

# Simple dict to store results
wasserstein_results = defaultdict(list)

@pytest.fixture(autouse=True)
def collect_wasserstein():
    """Automatically collect Wasserstein distances from all tests"""
    yield

    # At the end of all tests, print summary
    if wasserstein_results:
        print("\nWasserstein Distance Summary:")
        print("-" * 80)  # Increased width for longer keys
        print(f"{'Configuration':<50} {'Distances':<10}")
        print("-" * 80)
        for config, distances in wasserstein_results.items():
            print(f"{config:<50} {distances}")

@pytest.fixture
def test_setup():
    """Single fixture that provides all necessary test configuration and objects"""
    key = jax.random.PRNGKey(42)
    space_config = CONFIG["space"]

    # Initialize the Gaussian mixture prior
    mix_state = init_gaussian_mixture(key, space_config["d"])

    def create_sde(schedule_name):
        schedule_config = CONFIG["schedules"][schedule_name]
        beta = schedule_config["class"](**schedule_config["params"])
        return SDE(beta=beta, tf=space_config["t_final"])

    # Create observation setup
    key_meas, key_obs = jax.random.split(key)
    A = jax.random.normal(key_obs, (1, space_config["d"]))
    forward_model = MatrixProduct(A=A, std=space_config["sigma_y"])

    return {
        "key": key,
        "key_meas": key_meas,
        "mix_state": mix_state,
        "forward_model": forward_model,
        "A": A,
        "create_sde": create_sde,
        "perct": CONFIG["percentiles"],
        **space_config
    }

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
        t = timer(0) - timer(k+1)
        samples = position[:, k].squeeze()
        sample_indices = jax.random.choice(key, samples.shape[0], shape=(300,))
        samples = samples[sample_indices]


        if key is not None:  # Use subset of samples if key provided
            sample_indices = jax.random.choice(key, samples.shape[0], shape=(200,))
            samples = samples[sample_indices]

        #time = t if forward else timer(0) - t
        ks_statistic, p_value = sp.stats.kstest(
            np.array(samples),
            lambda x: cdf(x, t),
        )

        error_msg = f"Sample distribution does not match theoretical (p-value: {p_value}, t: {t}, k: {k})"
        if method:
            error_msg = f"Sample distribution does not match theoretical (method: {method.__name__}, p-value: {p_value}, t: {t}, k: {k})"

        assert p_value > 0.01, error_msg

@pytest.mark.parametrize("schedule_name", ["LinearSchedule", "CosineSchedule"])
@pytest.mark.parametrize("integrator_class", CONFIG["integrators"])
@pytest.mark.parametrize("timer_name,timer_fn", CONFIG["timers"])
def test_backward_sde_conditional_mixture(test_setup, plot_if_enabled, integrator_class, schedule_name, timer_name, timer_fn):
    # Unpack setup
    key = test_setup["key"]
    key_meas = test_setup["key_meas"]
    mix_state = test_setup["mix_state"]
    forward_model = test_setup["forward_model"]
    A = test_setup["A"]
    t_init = test_setup["t_init"]
    t_final = test_setup["t_final"]
    n_steps = test_setup["n_steps"]
    n_samples = test_setup["n_samples"]
    sigma_y = test_setup["sigma_y"]
    perct = test_setup["perct"]

    # Create SDE
    sde = test_setup["create_sde"](schedule_name)

    # Create timer
    timer = timer_fn(n_steps, t_final)

    # Generate observation
    key_samples = jax.random.split(key_meas)[0]
    x_star = sampler_mixtr(key_samples, mix_state, 1)[0]
    y = forward_model.measure(key_meas, None, x_star)

    # Compute theoretical posterior and score functions
    posterior_state = compute_posterior(mix_state, y, A, sigma_y)
    pdf, cdf, score = create_score_functions(posterior_state, sde)

    # Setup denoising process with timer
    integrator = integrator_class(sde=sde, timer=timer)
    denoise = Denoiser(
        integrator=integrator,
        sde=sde,
        score=score,
        x0_shape=x_star.shape
    )

    # Generate samples
    key_gen = jax.random.split(key_samples)[0]
    state, hist_position = denoise.generate(key_gen, n_steps, n_samples)
    hist_position = hist_position.squeeze().T

    # Visualization
    space = jnp.linspace(-10, 10, 100)
    plot_title = f"{integrator_class.__name__} (Timer: {timer_name}, Schedule: {schedule_name})"

    plot_if_enabled(lambda: display_trajectories(hist_position, 100, title=plot_title))
    plot_if_enabled(
        lambda: display_trajectories_at_times(
            hist_position,
            timer,
            n_steps,
            space,
            perct,
            lambda x, t: pdf(x, t_final - t),
            title=plot_title
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





# Add the denoiser classes to the CONFIG
CONFIG["cond_denoisers"] = [
    DPSDenoiser,
    TMPDenoiser,
    FPSDenoiser
]

@pytest.mark.parametrize("schedule_name", ["LinearSchedule", "CosineSchedule"])
@pytest.mark.parametrize("integrator_class", CONFIG["integrators"])
@pytest.mark.parametrize("timer_name,timer_fn", CONFIG["timers"])
@pytest.mark.parametrize("denoiser_class", CONFIG["cond_denoisers"])
def test_backward_CondDenoisers(test_setup, plot_if_enabled, integrator_class, schedule_name, timer_name, timer_fn, denoiser_class):
    # Unpack setup
    key = test_setup["key"]
    key_meas = test_setup["key_meas"]
    mix_state = test_setup["mix_state"]
    forward_model = test_setup["forward_model"]
    A = test_setup["A"]
    t_init = test_setup["t_init"]
    t_final = test_setup["t_final"]
    n_steps = test_setup["n_steps"]
    n_samples = test_setup["n_samples"]
    sigma_y = test_setup["sigma_y"]
    perct = test_setup["perct"]

    # Create SDE
    sde = test_setup["create_sde"](schedule_name)

    # Create timer
    timer = timer_fn(n_steps, t_final)

    # Generate observation
    key_samples = jax.random.split(key_meas)[0]
    x_star = sampler_mixtr(key_samples, mix_state, 1)[0]
    y = forward_model.measure(key_meas, None, x_star)

    # Compute theoretical posterior and score functions
    posterior_state = compute_posterior(mix_state, y, A, sigma_y)
    pdf, cdf, score = create_score_functions(posterior_state, sde)

    # Setup denoising process with timer
    integrator = integrator_class(sde=sde, timer=timer)
    denoise = denoiser_class(
        integrator=integrator,
        sde=sde,
        score=score,
        forward_model=forward_model
    )

    # Create measurement state with mask
    mask = jnp.ones_like(y)
    measurement_state = MeasurementState(y=y, mask_history=mask)

    # Generate samples
    key_gen = jax.random.split(key_samples)[0]
    state, hist_position = denoise.generate(key_gen, measurement_state, n_steps, n_samples)
    hist_position = hist_position.squeeze().T

    # Visualization
    space = jnp.linspace(-10, 10, 100)
    plot_title = f"{denoiser_class.__name__} (Integrator: {integrator_class.__name__}, Timer: {timer_name}, Schedule: {schedule_name})"

    plot_if_enabled(lambda: display_trajectories(hist_position, 100, title=plot_title))
    plot_if_enabled(
        lambda: display_trajectories_at_times(
            hist_position,
            timer,
            n_steps,
            space,
            perct,
            lambda x, t: pdf(x, t_final - t),
            title=plot_title
        )
    )

    # Compute Wasserstein distance
    wasserstein_distance, _ = ott.tools.sliced.sliced_wasserstein(
        state.integrator_state.position,
        posterior_state.means,
    )

    # Create a composite key for the results
    result_key = f"{denoiser_class.__name__}_{integrator_class.__name__}_{schedule_name}_{timer_name}"
    wasserstein_results[result_key].append(float(wasserstein_distance))

    print(f"\nWasserstein distance for {result_key}: {float(wasserstein_distance):.6f}")

    # assert wasserstein_distance < 0.1
    assert wasserstein_distance.item() < 0.1