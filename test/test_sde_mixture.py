from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy as sp

from examples.gaussian_mixtures.mixture import (
    MixState,
    cdf_t,
    rho_t,
    sampler_mixtr,
)
from examples.gaussian_mixtures.plotting import (
    display_histogram,
    display_trajectories,
    display_trajectories_at_times,
    plot_2d_mixture_and_samples,
    display_2d_trajectories_at_times,
)
from examples.gaussian_mixtures.test_config import CONFIG, create_basic_setup, create_sde, create_timer
from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.denoiser import Denoiser
# float64 accuracy
jax.config.update("jax_enable_x64", True)

@pytest.fixture
def test_setup():
    """Single fixture that provides all necessary test configuration and objects"""
    return create_basic_setup()

@pytest.mark.parametrize("schedule_name", ["LinearSchedule", "CosineSchedule"])
def test_forward_sde_mixture(test_setup, plot_if_enabled, schedule_name):
    t_final = test_setup["t_final"]
    n_samples = test_setup["n_samples"]
    n_steps = test_setup["n_steps"]
    ts = test_setup["ts"]
    space = test_setup["space"]
    key = test_setup["key"]
    mix_state = test_setup["mix_state"]
    perct = test_setup["perct"]

    # Create SDE and timer
    sde = create_sde(schedule_name, t_final)
    timer = create_timer("vp", n_steps, t_final)  # Use VP timer for forward process

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

    if test_setup["d"] == 1:
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
    elif test_setup["d"] == 2:
        plot_if_enabled(
            lambda: plot_2d_mixture_and_samples(
                mix_state,
                noised_samples.position[-1],  # Final timestep samples
                f"Forward SDE - {schedule_name}"
            )
        )
        # Create score function for forward process
        score_fn = lambda x, t: jax.grad(lambda pos: jnp.log(pdf(pos, t)))(x)

        plot_if_enabled(
            lambda: display_2d_trajectories_at_times(
                noised_samples.position.transpose(1, 0, 2),  # Reshape to (n_particles, n_steps, 2)
                timer,
                n_steps,
                perct,
                pdf,
                f"Forward Evolution - {schedule_name}",
                score=score_fn
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

@pytest.mark.parametrize("integrator_class,integrator_params", CONFIG["integrators"])
@pytest.mark.parametrize("timer_name,timer_fn", CONFIG["timers"])
@pytest.mark.parametrize("schedule_name", ["LinearSchedule", "CosineSchedule"])
def test_backward_sde_mixture(
    test_setup, plot_if_enabled,
    integrator_class, integrator_params, timer_name, timer_fn, schedule_name
):
    n_samples = test_setup["n_samples"]
    n_steps = test_setup["n_steps"]
    space = test_setup["space"]
    key = test_setup["key"]
    mix_state = test_setup["mix_state"]
    perct = test_setup["perct"]
    t_final = test_setup["t_final"]  # Use the global t_final for backward process
    x0_shape = mix_state.means.shape[1]

    # Create SDE and timer
    sde = create_sde(schedule_name, t_final)
    timer = create_timer(timer_name, n_steps, t_final)

    # Setup score function
    pdf = partial(rho_t, init_mix_state=mix_state, sde=sde)
    score = lambda x, t: jax.grad(lambda x: jnp.log(pdf(x, t)))(x)

    # Setup denoising process with timer and churn parameters
    integrator = integrator_class(sde=sde, timer=timer, **integrator_params)
    denoise = Denoiser(
        integrator=integrator, sde=sde, score=score, x0_shape=(x0_shape,)
    )

    # Generate samples
    key_samples, _ = jax.random.split(key)
    _, hist_position = denoise.generate(key_samples, n_steps, n_samples)
    hist_position = hist_position.squeeze()

    # Visualize results
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
    elif test_setup["d"] == 2:
        # Get final generated samples
        plot_if_enabled(
            lambda: plot_2d_mixture_and_samples(
                mix_state,
                hist_position,
                plot_title
            )
        )
        plot_if_enabled(
            lambda: display_2d_trajectories_at_times(
                hist_position,  # Already in shape (n_particles, n_steps, d)
                timer,
                n_steps,
                perct,
                lambda x, t: pdf(x, t_final - t),
                f"Backward Evolution - {plot_title}",
                score=lambda x, t: score(x, t_final - t)  # Use opposite time for score
            )
        )



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

