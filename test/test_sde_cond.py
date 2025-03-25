from functools import partial
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest
import scipy as sp
import numpy as np
import ott

from diffuse.diffusion.sde import SDE, LinearSchedule, CosineSchedule
from examples.gaussian_mixtures.cond_mixture import NoiseMask, posterior_distribution
from examples.gaussian_mixtures.mixture import (
    MixState,
    rho_t,
    display_histogram,
    sampler_mixtr,
)
from examples.gaussian_mixtures.forward_models.matrix_product import MatrixProduct
from examples.gaussian_mixtures.cond_mixture import compute_posterior, compute_xt_given_y, init_gaussian_mixture
from examples.gaussian_mixtures.mixture import cdf_mixtr, pdf_mixtr
from test.test_sde_mixture import display_trajectories_at_times
from diffuse.integrator.stochastic import EulerMaruyama
from diffuse.integrator.deterministic import DDIMIntegrator, HeunIntegrator, DPMpp2sIntegrator, EulerIntegrator
from diffuse.denoisers.denoiser import Denoiser
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
def sde_setup():
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
    sde = SDE(beta=beta, tf=2.0)
    return sde


@pytest.mark.parametrize("schedule", [LinearSchedule, CosineSchedule])
@pytest.mark.parametrize("integrator_class", [EulerMaruyama, DDIMIntegrator, HeunIntegrator, DPMpp2sIntegrator, EulerIntegrator])
@pytest.mark.parametrize("key", [jax.random.PRNGKey(42), jax.random.PRNGKey(666), jax.random.PRNGKey(1234)])
def test_backward_sde_conditional_mixture(integrator_class, plot_if_enabled, key, schedule):
    d = 1  # Dimensionality (can use d=200)
    sigma_y = 0.01

    # Initialize the Gaussian mixture prior
    key_init, key_mix = jax.random.split(key)
    mix_state = init_gaussian_mixture(key_init, d)

    # Define the SDE
    t_init, t_final, n_steps = 0.001, 2.0, 500
    beta = schedule(b_min=0.1, b_max=20.0, t0=t_init, T=t_final)
    sde = SDE(beta=beta, tf=t_final)

    # Generate observation (similar to main())
    key_meas, key_obs, key_samples, key_gen = jax.random.split(key_mix, 4)
    A = jax.random.normal(key_obs, (1, d))
    forward_model = MatrixProduct(A=A, std=sigma_y)
    x_star = sampler_mixtr(key_samples, mix_state, 1)[0]
    print(f"x_star shape: {x_star.shape}")
    print(f"True x* (first 5 dims): {x_star[:5]}")
    y = forward_model.measure(key_meas, None, x_star)

    # Compute theoretical posterior
    posterior_state = compute_posterior(mix_state, y, A, sigma_y)

    # Define score function using posterior distribution
    def pdf(x, t):
        mix_state_t = compute_xt_given_y(posterior_state, sde, t)
        return pdf_mixtr(mix_state_t, x)
    def cdf(x, t):
        mix_state_t = compute_xt_given_y(posterior_state, sde, t)
        return cdf_mixtr(mix_state_t, x)
    def score(x, t):
        return jax.grad(pdf)(x, t) / pdf(x, t)


    # Define Integrator and Denoiser
    integrator = integrator_class(sde=sde)
    denoise = Denoiser(
        integrator=integrator, sde=sde, score=score, x0_shape=x_star.shape
    )

    # Generate samples
    n_samples = 5000
    state, hist_position = denoise.generate(key_gen, n_steps, n_samples)
    hist_position = hist_position.squeeze().T

    # assert end time is < t_final
    assert state.integrator_state.t[0] < t_final


    # Visualization
    perct = [0., 0.05, 0.1, 0.3, 0.6, 0.7, .73, .75, 0.8, 0.9]
    space = jnp.linspace(-10, 10, 100)
    plot_if_enabled(lambda: display_trajectories(hist_position, 100, title=integrator_class.__name__))
    plt.show()
    # plt.close()
    plot_if_enabled(lambda: display_trajectories_at_times(
        hist_position,
        t_init,
        t_final,
        n_steps,
        space,
        perct,
        lambda x, t: pdf(x, t_final - t),
        title=integrator_class.__name__
    ))
    plt.show()

    # assert samples are distributed according to the true density
    for i, x in enumerate(perct):
        k = int(x * n_steps)
        t = t_init + (k+1) * (t_final - t_init) / n_steps
        ks_statistic, p_value = sp.stats.kstest(
            np.array(hist_position[:, k]),
            lambda x: cdf(x, t_final - t),
        )
        assert p_value > 0.05, f"Sample distribution does not match theoretical (method: {integrator_class.__name__}, p-value: {p_value}, t: {t}, k: {k})"

    # test for last element
    k = -1
    t = state.integrator_state.t[0]
    ks_statistic, p_value = sp.stats.kstest(
        np.array(hist_position[:, k]),
        lambda x: cdf(x, t_final - t),
    )
    assert p_value > 0.05, f"Sample distribution does not match theoretical (method: {integrator_class.__name__}, p-value: {p_value}, t: {t}, k: {k})"

    # compute Wasserstein distance between gen samples and true posterior samples
    wasserstein_distance, _ = ott.tools.sliced.sliced_wasserstein(
        state.integrator_state.position,
        posterior_state.means,
    )
    print(f"method: {integrator_class.__name__}, Wasserstein distance: {wasserstein_distance}, n_samples: {n_samples}, n_steps: {n_steps}")


