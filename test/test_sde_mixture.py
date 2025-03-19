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
from diffuse.diffusion.sde import SDE, LinearSchedule, SDEState
from diffuse.denoisers.denoiser import Denoiser
from diffuse.integrator.stochastic import EulerMaruyama
from diffuse.integrator.deterministic import DDIMIntegrator, HeunIntegrator, DPMpp2sIntegrator

# float64 accuracy
jax.config.update("jax_enable_x64", True)

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def init_mixture(key):
    n_mixt = 3
    d = 1
    means = jax.random.uniform(key, (n_mixt, d), minval=-3, maxval=3)
    covs = 0.1 * (jax.random.normal(key + 1, (n_mixt, d, d))) ** 2
    mix_weights = jax.random.uniform(key + 2, (n_mixt,))
    mix_weights /= jnp.sum(mix_weights)

    return MixState(means, covs, mix_weights)


@pytest.fixture
def get_percentiles():
    perct = [0, 0.03, 0.06, 0.08, 0.1, 0.3, 0.7, 0.8, 0.9, 1]
    return perct


def display_trajectories_at_times(
    particles, t_init, t_final, n_steps, space, perct, pdf, title=None
):
    n_plots = len(perct)
    d = particles.shape[-1]
    fig, axs = plt.subplots(n_plots, 1, figsize=(10 * n_plots, n_plots))
    if title:
        fig.suptitle(title)
    for i, x in enumerate(perct):
        k = int(x * n_steps)
        t = t_init + (k+1) * (t_final - t_init) / n_steps

        # plot histogram and true density
        display_histogram(particles[:, k], axs[i])
        axs[i].plot(space, jax.vmap(pdf, in_axes=(0, None))(space, t))

@pytest.fixture
def sde_setup():
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
    sde = SDE(beta=beta, tf=2.0)
    return sde


@pytest.fixture
def time_space_setup():
    t_init = 0.0
    t_final = 2.0
    n_samples = 5000
    n_steps = 100
    ts = jnp.linspace(t_init, t_final, n_steps)
    space = jnp.linspace(-3, 3, 100)
    dts = jnp.array([t_final / n_steps] * (n_steps))
    return t_init, t_final, n_samples, n_steps, ts, space, dts


def test_forward_sde_mixture(
    sde_setup, time_space_setup, plot_if_enabled, get_percentiles, init_mixture, key
):
    sde = sde_setup
    t_init, t_final, n_samples, n_steps, ts, space, _ = time_space_setup
    perct = get_percentiles
    # samples from univariate gaussian
    mix_state = init_mixture
    samples_mixt = sampler_mixtr(key, mix_state, n_samples)

    # sample from noising process
    keys = jax.random.split(key, n_samples * n_steps).reshape((n_samples, n_steps, -1))
    t0 = jnp.array([t_init] * n_samples)
    state_mixt = SDEState(position=samples_mixt, t=t0)
    noised_samples = jax.vmap(
        jax.vmap(sde.path, in_axes=(0, None, 0)), in_axes=(0, 0, None)
    )(keys, state_mixt, ts)

    pdf = partial(rho_t, init_mix_state=mix_state, sde=sde)
    # plot if enabled
    plot_if_enabled(
        lambda: display_trajectories(noised_samples.position.squeeze(), 100)
    )
    plot_if_enabled(
        lambda: display_trajectories_at_times(
            noised_samples.position.squeeze(),
            t_init,
            t_final,
            n_steps,
            space,
            perct,
            pdf,
        )
    )

    # assert samples are distributed according to the true density
    for i, x in enumerate(perct):
        k = int(x * n_steps) + 1
        t = t_init + (k+1) * (t_final - t_init) / n_steps
        ks_statistic, p_value = sp.stats.kstest(
            np.array(noised_samples.position[:, k].squeeze()),
            lambda x: cdf_t(x, t, mix_state, sde),
        )
        assert (
            p_value > 0.05
        ), f"Sample distribution does not match theoretical (p-value: {p_value}, t: {t}, k: {k})"

@pytest.mark.parametrize("integrator_class", [EulerMaruyama, DDIMIntegrator, HeunIntegrator, DPMpp2sIntegrator])
def test_backward_sde_mixture(
    sde_setup, time_space_setup, plot_if_enabled, get_percentiles, init_mixture, key, integrator_class
):
    sde = sde_setup
    t_init, t_final, n_samples, n_steps, ts, space, dts = time_space_setup
    perct = get_percentiles
    mix_state = init_mixture

    pdf = partial(rho_t, init_mix_state=mix_state, sde=sde)
    score = lambda x, t: jax.grad(pdf)(x, t) / pdf(x, t)

    # init_samples = jax.random.normal(key_samples, (n_samples, 1))
    keys = jax.random.split(key, n_samples)

    # define Intergator and Denoiser
    integrator = integrator_class(sde=sde)
    denoise = Denoiser(
        integrator=integrator, sde=sde, score=score, x0_shape=(1,)
    )

    # generate samples
    key_samples, _ = jax.random.split(key)
    state, hist_position = denoise.generate(key_samples, n_steps, n_samples)
    hist_position = hist_position.squeeze().T

    # plot if enabled
    plot_if_enabled(lambda: display_trajectories(hist_position, 100, title=integrator_class.__name__))
    plot_if_enabled(
        lambda: display_trajectories_at_times(
            hist_position,
            t_init,
            t_final,
            n_steps,
            space,
            perct,
            lambda x, t: pdf(x, t_final - t),
            title=integrator_class.__name__
        )
    )

    # assert samples are distributed according to the true density
    for i, x in enumerate(perct):
        k = int(x * n_steps)
        t = t_init + (k+1) * (t_final - t_init) / n_steps
        ks_statistic, p_value = sp.stats.kstest(
            hist_position[:, k].squeeze(),
            lambda x: cdf_t(x, t_final - t, mix_state, sde),
        )
        assert (
            p_value > 0.05
        ), f"Sample distribution does not match theoretical (method: {integrator_class.__name__}, p-value: {p_value}, t: {t}, k: {k})"
