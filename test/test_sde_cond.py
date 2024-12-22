from functools import partial
import pdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from diffuse.diffusion.sde import SDE, LinearSchedule
from examples.gaussian_mixtures.cond_mixture import NoiseMask, posterior_distribution
from examples.gaussian_mixtures.mixture import MixState, rho_t, display_histogram, sampler_mixtr
from diffuse.integrator.stochastic import EulerMaruyama
from diffuse.denoisers.denoiser import Denoiser
from diffuse.denoisers.cond_denoiser import CondDenoiser
from examples.gaussian_mixtures.mixture import display_trajectories


@pytest.fixture
def noise_mask():
    return NoiseMask(alpha=0.8, std=1.3)

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
def key():
    return jax.random.PRNGKey(666)

@pytest.fixture
def sde_setup():
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
    sde = SDE(beta=beta, tf=2.0)
    return sde


def test_posterior_distribution_visual(noise_mask, init_mixture, plot_if_enabled, sde_setup, key):
    mix_state = init_mixture
    sample = sampler_mixtr(key, mix_state, 1).squeeze()
    y_meas = noise_mask.measure(key, sample)
    post_state = posterior_distribution(mix_state, noise_mask, y_meas)
    sde = sde_setup

    # samples from posterior
    n_samples = 1000
    post_samples = sampler_mixtr(key, post_state, n_samples)

    #
    space = jnp.linspace(-3, 3, 100)
    prior_pdf = partial(rho_t, init_mix_state=init_mixture, sde=sde)
    post_pdf = partial(rho_t, init_mix_state=post_state, sde=sde)
    # plot
    def plot_posterior():
        fig, ax = plt.subplots()
        display_histogram(post_samples, ax)
        ax.plot(space, jax.vmap(prior_pdf, in_axes=(0, None))(space, 0.0))
        ax.plot(space, jax.vmap(post_pdf, in_axes=(0, None))(space, 0.0))
        ax.set_title("Posterior Distribution")
        ax.grid(True)
        plt.tight_layout()

    plot_if_enabled(plot_posterior)


def test_denoise_mixture(noise_mask, init_mixture, plot_if_enabled, sde_setup, key):
    mix_state = init_mixture
    sample = sampler_mixtr(key, mix_state, 1).squeeze()
    y_meas = noise_mask.measure(key, sample)
    post_state = posterior_distribution(mix_state, noise_mask, y_meas)
    sde = sde_setup
    alpha, std_noise = noise_mask.alpha, noise_mask.std

    pdf = partial(rho_t, init_mix_state=mix_state, sde=sde)
    post_pdf = partial(rho_t, init_mix_state=post_state, sde=sde)
    score = lambda x, t: jax.grad(post_pdf)(x, t) / post_pdf(x, t)
    space = jnp.linspace(-3, 3, 100)

    # define Intergator and Denoiser
    integrator = EulerMaruyama(sde=sde)
    denoise = Denoiser(integrator=integrator, logpdf=pdf, sde=sde, score=score)

    init_samples = jax.random.normal(key, (1000, 1))
    keys = jax.random.split(key, init_samples.shape[0])

    # test if init and step work
    state = jax.vmap(denoise.init, in_axes=(0, 0, None))(init_samples, keys, 0.01)
    state = jax.vmap(denoise.step)(state)

    # generate samples
    state, hist = jax.vmap(denoise.generate, in_axes=(0, None, None, None))(keys, 0.01, 2.0, (1,))
    samples = state.integrator_state.position

    # plot samples
    def plot_samples():
        fig, ax = plt.subplots()
        display_histogram(samples, ax)
        ax.set_title("Samples from Posterior")
        ax.plot(space, jax.vmap(post_pdf, in_axes=(0, None))(space, 0.0))
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_trajectories():
        fig, ax = plt.subplots()
        display_trajectories(hist.squeeze(), 100)
        ax.set_title("Samples from Posterior")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    plot_if_enabled(plot_samples)
    plot_if_enabled(plot_trajectories)
