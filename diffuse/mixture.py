from typing import NamedTuple
from jaxtyping import PyTreeDef
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax


class MixState(NamedTuple):
    """
    Represents the state of the mixture optimization algorithm.

    Attributes:
        means (PyTreeDef): The means of the mixture components.
        cov (PyTreeDef): Covariances components of the mixture
        mix_weights (PyTreeDef): The mixture weights.
        grad_state (GradState, optional): The gradient state. Defaults to GradState().
        info (INFO, optional): Hyperparameters
    """

    means: PyTreeDef
    cov: PyTreeDef
    mix_weights: PyTreeDef


def pdf_mixtr(state: MixState, x: PyTreeDef):
    """
    Calculate the probability density function (PDF) of a multivariate normal distribution
    mixture given a state and input data.

    Args:
        state (MixState): The state of the mixture model, including means, cholesky factors,
                          mixture weights, and other parameters.
        x (PyTreeDef): The input data.

    Returns:
        float: The PDF of the multivariate normal distribution mixture.
    """
    means, sigmas, weights = state

    def pdf_multivariate_normal(mean, sigma):
        return jax.scipy.stats.multivariate_normal.pdf(x, mean, sigma)

    pdf = jax.vmap(pdf_multivariate_normal)(means, sigmas)
    return weights @ pdf


# def pdf_mixtr(state:MixState, x):
#     mu, sigma, weights = state
#     pdfs = jax.scipy.stats.multivariate_normal.pdf(x, mu, sigma)
#     return  weights @ pdfs


def init_mixture(key):
    n_mixt = 5
    d = 1
    means = jax.random.uniform(key, (n_mixt, d), minval=-3, maxval=3)
    covs = 0.1 * (jax.random.normal(key, (n_mixt, d, d))) ** 2
    mix_weights = jax.random.uniform(key, (n_mixt,))
    mix_weights /= jnp.sum(mix_weights)

    return MixState(means, covs, mix_weights)


def sampler_mixtr(key, state: MixState, N):
    """
    Sampler from the mixture
    """
    mu, sigma, weights = state
    key1, key2 = jax.random.split(key)
    idx = jax.random.choice(key1, jnp.arange(len(weights)), shape=(N,), p=weights)
    noise = jax.random.normal(key2, shape=(N, 1))
    return mu[idx] + jnp.einsum("nij, ni->nj", sigma[idx], noise)


def mixtr_sample(state: MixState, key):
    """
    Samples from a mixture distribution defined by the given state.

    Args:
        state (MixState): The state of the mixture distribution, containing means, cholesky factors,
                          and mixture weights.
        key (PRNGKey): The random key used for sampling.

    Returns:
        sample: A sample from the mixture distribution.
    """
    means, sigmas, mix_weights, *_ = state
    idx = jax.random.categorical(key, mix_weights, shape=())
    key, subkey = jax.random.split(key)
    return jax.random.multivariate_normal(subkey, means[idx], sigmas[idx])


def noise_process(key, sample, schedule) -> jnp.ndarray:
    """
    Add noise to the sample
    """
    alphas = 1 - schedule
    a_prod = jnp.prod(alphas)
    noise = jax.random.normal(key, shape=sample.shape)
    print(a_prod)
    return jnp.sqrt(a_prod) * sample + jnp.sqrt(1 - a_prod) * noise


def _noise_process(key, sample, schedule) -> jnp.ndarray:
    """
    Add noise to the sample
    """

    def step(sample, tup):
        key, schedule = tup
        key1, key2 = jax.random.split(key)
        noise = jax.random.normal(key1, shape=sample.shape)
        return jnp.sqrt(schedule) * noise + jnp.sqrt(1 - schedule) * sample, None

    keys = jax.random.split(key, schedule.shape[0])

    samples, _ = jax.lax.scan(step, sample, (keys, schedule))
    return samples


def noiser_pdf(state: MixState, x, schedule):
    """
    Compute the pdf of the noise process
    """
    mu, sigma, weights = state
    alpha = 1 - schedule
    a_prod = jnp.prod(alpha)
    pdfs = jax.scipy.stats.norm.pdf(x, jnp.sqrt(a_prod) * mu, (1 - a_prod) * sigma)
    return weights @ pdfs


xmax = 4
nbins = 200


def display_histogram(samples, ax):
    nb = samples.flatten().shape[0]
    h0, b = jnp.histogram(samples.flatten(), bins=nbins, range=[-xmax, xmax])
    h0 = h0 / nb * nbins / (2 * xmax)
    ax.bar(
        jnp.linspace(-xmax, xmax, nbins),
        h0,
        width=2 * xmax / (nbins - 1),
        align="center",
        color="red",
    )


if __name__ == "__main__":
    key = jax.random.PRNGKey(1)
    n_mixt = 3
    d = 1
    means = 4 * jax.random.normal(key, (n_mixt, d))
    covs = 2 * jax.random.normal(key, (n_mixt, d, d))
    mix_weights = jax.random.uniform(key, (n_mixt,))
    mix_weights /= jnp.sum(mix_weights)

    state = MixState(means, covs, mix_weights)

    n_samples = 5000
    samples = sampler_mixtr(key, state, n_samples)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    space = jnp.linspace(-xmax, xmax, 200)
    pdf = jax.vmap(pdf_mixtr, in_axes=(None, 0))(state, space)
    axs[0].plot(space, pdf)
    display_histogram(samples, axs[0])

    # noise samples
    schedule = jnp.linspace(0.001, 0.999, 100)
    samples = noise_process(key, samples, schedule)
    pdf = jax.vmap(noiser_pdf, in_axes=(None, 0, None))(state, space, schedule)
    unit_pdf_norm = jax.scipy.stats.norm.pdf(space, 0, 1)
    axs[1].plot(space, pdf)
    axs[1].plot(space, unit_pdf_norm)
    display_histogram(samples, axs[1])
    plt.show()
