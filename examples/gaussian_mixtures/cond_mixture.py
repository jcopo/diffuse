from dataclasses import dataclass
import pdb

import matplotlib.pyplot as plt
import jax.scipy.stats as stats
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray
from functools import partial
import numpy as np
import scipy as sp
import ott

from examples.gaussian_mixtures.mixture import MixState, pdf_mixtr, sampler_mixtr, rho_t, cdf_mixtr
from diffuse.diffusion.sde import SDE, LinearSchedule
from diffuse.integrator.stochastic import EulerMaruyama
from diffuse.integrator.deterministic import DDIMIntegrator, HeunIntegrator, DPMpp2sIntegrator
from diffuse.denoisers.denoiser import Denoiser
from examples.gaussian_mixtures.mixture import display_trajectories
from test.test_sde_mixture import display_trajectories_at_times
from examples.gaussian_mixtures.forward_models.matrix_product import MatrixProduct

# float64 accuracy
jax.config.update("jax_enable_x64", True)


@dataclass
class NoiseMask:
    A: Array
    alpha: float
    std: float

    def measure(self, key: PRNGKeyArray, x: Array) -> Array:
        return self.A @ x + jax.random.normal(key, shape=x.shape) * self.std

    def restore(self, measured: Array) -> Array:
        return self.A.T @ measured


def posterior_distribution(
    mix_state: MixState, noise_mask: NoiseMask, y_meas: Array
) -> MixState:
    r"""
    Close form of p(theta | y) for y = alpha * theta + noise and theta ~ \sum_i w_i N(means[i], covs[i])
    """
    means, covs, weights = mix_state
    alpha, std = noise_mask.alpha, noise_mask.std

    new_covs = (covs ** (-1) + alpha**2 / std**2) ** (-1)
    new_means = new_covs[:, 0] * (means / covs[:, 0] + alpha * y_meas / std**2)
    new_weights = weights * jax.scipy.stats.norm.pdf(y_meas, alpha * means[:, 0], std)

    new_weights = new_weights / new_weights.sum()
    return MixState(new_means, new_covs, new_weights)


def init_gaussian_mixture(key, d=20):
    n_mixt = 25
    grid = jnp.array([(i, j) for i in range(-2, 3) for j in range(-2, 3)])  # Shape: (25, 2)
    grid_scaled = 8 * grid
    repeats = (d + 1) // 2
    means = jnp.tile(grid_scaled, (1, repeats))[:, :d]  # Shape: (25, d)
    covs = jnp.repeat(jnp.eye(d)[None, :, :], n_mixt, axis=0)  # Shape: (25, d, d)
    key_weights = jax.random.split(key, 1)[0]
    mix_weights = jax.random.uniform(key_weights, (n_mixt,))
    mix_weights = mix_weights / jnp.sum(mix_weights)
    return MixState(means, covs, mix_weights)


# Generate observation
# def generate_observation(key, mix_state: MixState, d=20, sigma_y=0.05):
#     key_A, key_x, key_noise = jax.random.split(key, 3)
#     x_star = sampler_mixtr(key_x, mix_state, 1)[0]  # Shape: (d,)
#     A = jax.random.normal(key_A, (1, d))  # Shape: (1, d)
#     epsilon = jax.random.normal(key_noise, (1,))
#     y = A @ x_star + sigma_y * epsilon  # Shape: (1,)
#     return y, A, x_star


def compute_xt_given_y(mix_state_posterior: MixState, sde:SDE, t: float):
    means, covs, weights = mix_state_posterior
    alpha_t = jnp.exp(-sde.beta.integrate(t, 0.0))
    means_xt = jnp.sqrt(alpha_t) * means
    covs_xt = alpha_t * covs + (1 - alpha_t) * jnp.eye(covs.shape[-1])
    return MixState(means_xt, covs_xt, weights)


# Compute the theoretical posterior
def compute_posterior(mix_state: MixState, y: Array, A: Array, sigma_y=0.05):
    means, covs, weights = mix_state
    d = means.shape[-1]

    # Compute posterior parameters
    AAT = A @ A.T  # Scalar (1x1)
    Sigma_bar = jnp.linalg.inv(jnp.eye(d) + (1 / sigma_y**2) * (A.T @ A))  # Shape: (d, d)
    covs_bar = jnp.repeat(Sigma_bar[None, :, :], len(weights), axis=0)  # Shape: (25, d, d)

    # Posterior means
    term1 = (1 / sigma_y**2) * (A.T @ y)  # Shape: (d, 1) -> (d,)
    means_bar = jax.vmap(lambda m: Sigma_bar @ (term1 + m))(means)  # Shape: (25, d)

    # Unnormalized posterior weights
    likelihood_var = sigma_y**2 + AAT.item()  # Scalar
    y_pred = jax.vmap(lambda m: A @ m)(means).squeeze()  # Shape: (25,)
    log_likelihood = stats.norm.logpdf(y, loc=y_pred, scale=jnp.sqrt(likelihood_var))
    weights_bar = weights * jnp.exp(log_likelihood)  # Unnormalized
    weights_bar = weights_bar / jnp.sum(weights_bar)  # Normalize

    return MixState(means_bar, covs_bar, weights_bar)


# Visualization function including posterior
def plot_distributions(mix_state, posterior_state, sde, t_values, x_grid, samples_prior):
    fig, axes = plt.subplots(1, len(t_values) + 2, figsize=(18, 5))

    # Plot prior distribution (t=0)
    pdf_prior = pdf_mixtr(mix_state, x_grid)
    axes[0].plot(x_grid[:, 0], pdf_prior, label="Prior PDF")
    axes[0].hist(samples_prior[:, 0], bins=50, density=True, alpha=0.5, color="red", label="Samples")
    axes[0].set_title("Prior (t=0)")
    axes[0].legend()

    # Plot posterior distribution
    pdf_posterior = pdf_mixtr(posterior_state, x_grid)
    axes[1].plot(x_grid[:, 0], pdf_posterior, label="Posterior PDF", color="green")
    samples_posterior = sampler_mixtr(jax.random.PRNGKey(43), posterior_state, 1000)
    axes[1].hist(samples_posterior[:, 0], bins=50, density=True, alpha=0.5, color="green", label="Posterior Samples")
    axes[1].set_title("Theoretical Posterior")
    axes[1].legend()

    # Plot noised distributions
    for i, t in enumerate(t_values):
        pdf_t = rho_t(x_grid, t, mix_state, sde)
        axes[i + 2].plot(x_grid[:, 0], pdf_t, label=f"PDF at t={t}", color="blue")
        axes[i + 2].set_title(f"Noised (t={t})")
        axes[i + 2].legend()

    plt.tight_layout()
    plt.show()


def test_backward_sde_conditional_mixture(integrator_class):
    key = jax.random.PRNGKey(42)
    d = 1  # Dimensionality (can use d=200)
    sigma_y = 0.05

    # Initialize the Gaussian mixture prior
    mix_state = init_gaussian_mixture(key, d)

    # Define the SDE
    t_init, t_final, n_steps = 0.001, 2.0, 100
    beta = LinearSchedule(b_min=0.1, b_max=20.0, t0=t_init, T=t_final)
    sde = SDE(beta=beta, tf=t_final)

    # Generate observation (similar to main())
    key_meas, key_obs, key_samples = jax.random.split(key, 3)
    sigma_y = 0.01
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
    n_samples = 1000
    state, hist_position = denoise.generate(key_samples, n_steps-2, n_samples)
    hist_position = hist_position.squeeze().T


    # Visualization
    perct = [0, 0.05, 0.1, 0.3, 0.6, 0.7, .73, .75, 0.8, 0.9, 1.]
    space = jnp.linspace(-10, 10, 100)
    display_trajectories(hist_position, 100, title=integrator_class.__name__)
    plt.show()
    # plt.close()
    display_trajectories_at_times(
        hist_position,
        t_init,
        t_final,
        n_steps,
        space,
        perct,
        lambda x, t: pdf(x, t_final - t),
        title=integrator_class.__name__
    )
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

    # compute Wasserstein distance between gen samples and true posterior samples
    wasserstein_distance, _ = ott.tools.sliced.sliced_wasserstein(
        state.integrator_state.position,
        posterior_state.means,
    )
    print(f"method: {integrator_class.__name__}, Wasserstein distance: {wasserstein_distance}, n_samples: {n_samples}, n_steps: {n_steps}")


    return state, hist_position, posterior_state


if __name__ == "__main__":
    integrators = [EulerMaruyama, DDIMIntegrator, HeunIntegrator, DPMpp2sIntegrator]
    for integrator_class in integrators:
        test_backward_sde_conditional_mixture(integrator_class)
    #main()