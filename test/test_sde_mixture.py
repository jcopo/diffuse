from functools import partial


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy as sp

from examples.gaussian_mixtures.mixture import (
    MixState,
    cdf_t,
    display_histogram,
    display_trajectories,
    rho_t,
    sampler_mixtr,
)
from diffuse.diffusion.sde import SDE, LinearSchedule, CosineSchedule, SDEState
from diffuse.denoisers.denoiser import Denoiser
from diffuse.integrator.stochastic import EulerMaruyamaIntegrator
from diffuse.integrator.deterministic import DDIMIntegrator, EulerIntegrator, HeunIntegrator, DPMpp2sIntegrator
from diffuse.timer.base import VpTimer, HeunTimer, DDIMTimer
# float64 accuracy
jax.config.update("jax_enable_x64", True)

# Global configurations
CONFIG = {
    "schedules": {
        "LinearSchedule": {
            "params": {"b_min": 0.02, "b_max": 5.0, "t0": 0.0, "T": 1.0},
            "class": LinearSchedule
        },
        "CosineSchedule": {
            "params": {"b_min": 0.1, "b_max": 20.0, "t0": 0.0, "T": 1.0},
            "class": CosineSchedule
        }
    },
    "integrators": [
        EulerMaruyamaIntegrator,
        DDIMIntegrator,
        EulerIntegrator,
        HeunIntegrator,
        DPMpp2sIntegrator
    ],
    "timers": [
        ("vp", lambda n_steps, t_final: VpTimer(n_steps=n_steps, eps=0.001, tf=t_final)),
        ("heun", lambda n_steps, t_final: HeunTimer(n_steps=n_steps, rho=7.0, sigma_min=0.002, sigma_max=1.0)),
        # ("ddim", lambda n_steps, t_final: DDIMTimer(n_steps=n_steps, n_time_training=1000, c_1=0.001, c_2=0.008, j0=8))
    ],
    "percentiles": [0.03, 0.06, 0.08, 0.1, 0.3, 0.7, 0.8, 0.9, 1],
    "space": {
        "t_init": 0.0,
        "t_final": 1.0,
        "n_samples": 5000,
        "n_steps": 300,
        "space_min": -5,
        "space_max": 5,
        "space_points": 300
    }
}

@pytest.fixture
def test_setup():
    """Single fixture that provides all necessary test configuration and objects"""
    key = jax.random.PRNGKey(42)
    space_config = CONFIG["space"]
    t_final = space_config["t_final"]

    # Create mixture state
    n_mixt = 3
    d = 1
    means = jax.random.uniform(key, (n_mixt, d), minval=-3, maxval=3)
    covs = 0.1 * (jax.random.normal(key + 1, (n_mixt, d, d))) ** 2
    mix_weights = jax.random.uniform(key + 2, (n_mixt,))
    mix_weights /= jnp.sum(mix_weights)
    mix_state = MixState(means, covs, mix_weights)

    # Create space setup
    ts = jnp.linspace(space_config["t_init"], t_final, space_config["n_steps"])
    space = jnp.linspace(space_config["space_min"], space_config["space_max"], space_config["space_points"])

    def create_sde(schedule_name, t_final):
        schedule_config = CONFIG["schedules"][schedule_name]
        params = schedule_config["params"].copy()
        params["T"] = t_final
        beta = schedule_config["class"](**params)
        return SDE(beta=beta, tf=t_final)

    # Create default timer for forward process
    forward_timer = CONFIG["timers"][0][1](space_config["n_steps"], t_final=t_final)  # Using VP timer

    return {
        "key": key,
        "mix_state": mix_state,
        "ts": ts,
        "space": space,
        "create_sde": create_sde,
        "perct": CONFIG["percentiles"],
        "forward_timer": forward_timer,
        **space_config
    }

@pytest.mark.parametrize("schedule_name", ["LinearSchedule", "CosineSchedule"])
def test_forward_sde_mixture(test_setup, plot_if_enabled, schedule_name):
    t_final = test_setup["t_final"]
    sde = test_setup["create_sde"](schedule_name, t_final)
    n_samples = test_setup["n_samples"]
    n_steps = test_setup["n_steps"]
    ts = test_setup["ts"]
    space = test_setup["space"]
    key = test_setup["key"]
    mix_state = test_setup["mix_state"]
    perct = test_setup["perct"]
    timer = test_setup["forward_timer"]

    # Generate initial samples
    samples_mixt = sampler_mixtr(key, mix_state, n_samples)

    # Setup noising process
    keys = jax.random.split(key, n_samples * n_steps).reshape((n_samples, n_steps, -1))
    t0 = jnp.array([test_setup["t_init"]] * n_samples)
    state_mixt = SDEState(position=samples_mixt, t=t0)

    # Run forward process
    noised_samples = jax.vmap(
        jax.vmap(sde.path, in_axes=(0, None, 0)), in_axes=(0, 0, None)
    )(keys, state_mixt, ts)

    # Setup visualization
    pdf = partial(rho_t, init_mix_state=mix_state, sde=sde)
    plot_if_enabled(
        lambda: display_trajectories(noised_samples.position.squeeze(), 100)
    )

    plot_if_enabled(
        lambda: display_trajectories_at_times(
            noised_samples.position.squeeze(),
            timer,
            n_steps,
            space,
            perct,
            pdf,
        )
    )

    # Validate distributions
    validate_distributions(
        noised_samples.position[1:, :],
        timer,
        n_steps,
        perct,
        mix_state,
        sde
    )

@pytest.mark.parametrize("integrator_class", CONFIG["integrators"])
@pytest.mark.parametrize("timer_name,timer_fn", CONFIG["timers"])
@pytest.mark.parametrize("schedule_name", ["LinearSchedule", "CosineSchedule"])
def test_backward_sde_mixture(
    test_setup, plot_if_enabled,
    integrator_class, timer_name, timer_fn, schedule_name
):
    n_samples = test_setup["n_samples"]
    n_steps = test_setup["n_steps"]
    space = test_setup["space"]
    key = test_setup["key"]
    mix_state = test_setup["mix_state"]
    perct = test_setup["perct"]
    t_final = test_setup["t_final"]  # Use the global t_final for backward process

    # Create timer with explicit t_final
    timer = timer_fn(n_steps, t_final)

    # Create SDE with explicit t_final
    sde = test_setup["create_sde"](schedule_name, t_final)

    # Setup score function
    pdf = partial(rho_t, init_mix_state=mix_state, sde=sde)
    score = lambda x, t: jax.grad(lambda x: jnp.log(pdf(x, t)))(x)

    # Setup denoising process
    integrator = integrator_class(sde=sde, timer=timer)
    denoise = Denoiser(
        integrator=integrator, sde=sde, score=score, x0_shape=(1,)
    )

    # Generate samples
    key_samples, _ = jax.random.split(key)
    _, hist_position = denoise.generate(key_samples, n_steps, n_samples)
    hist_position = hist_position.squeeze().T

    # Visualize results
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

    # Validate distributions
    validate_distributions(
        hist_position,
        timer,
        n_steps,
        perct,
        mix_state,
        sde,
        key=key,
        method=integrator_class,
        t_final=t_final,
        forward=False
    )

def display_trajectories_at_times(
    particles, timer, n_steps, space, perct, pdf, title=None
):
    n_plots = len(perct)
    d = particles.shape[-1]
    fig, axs = plt.subplots(n_plots, 1, figsize=(10 * n_plots, n_plots))
    if title:
        fig.suptitle(title)
    for i, x in enumerate(perct):
        k = int(x * n_steps)
        t = timer(0) - timer(k+1)
        # plot histogram and true density
        display_histogram(particles[:, k], axs[i])
        axs[i].plot(space, jax.vmap(pdf, in_axes=(0, None))(space, t))

def validate_distributions(position, timer, n_steps, perct, mix_state, sde, key=None, method=None, t_final=1.0, forward=True):
    """Helper function to validate sample distributions against theoretical ones.

    Args:
        position: Array of positions to validate
        timer: Timer object that converts steps to time
        n_steps: Number of steps
        perct: Percentiles to check
        mix_state: Mixture state
        sde: SDE object
        key: Optional PRNG key for sampling (used in backward validation)
        method: Optional method name for error messages
        t_final: Final time (defaults to 1.0 for forward process)
    """
    for i, x in enumerate(perct):
        k = int(x * n_steps) + (1 if key is None else 0)  # +1 only for forward process
        t = t_final - timer(k+1)

        if key is not None:  # backward process case
            samples = position[:, k].squeeze()
            sample_indices = jax.random.choice(key, samples.shape[0], shape=(300,))
            samples = samples[sample_indices]
        else:  # forward process case
            samples = np.array(position[:, k].squeeze())

        time = t if forward else t_final - t
        _, p_value = sp.stats.kstest(
            samples,
            lambda x: cdf_t(x, time, mix_state, sde),
        )

        error_msg = f"Sample distribution does not match theoretical (p-value: {p_value}, t: {t}, k: {k})"
        if method:
            error_msg = f"Sample distribution does not match theoretical (method: {method.__name__}, p-value: {p_value}, t: {t}, k: {k})"

        assert p_value > 0.01, error_msg

