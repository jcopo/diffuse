# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTreeDef

from diffuse.diffusion.sde import SDE


class MixState(NamedTuple):
    """
    Represents the state of the mixture optimization algorithm.

    Attributes:
        means (PyTreeDef): The means of the mixture components.
        cov (PyTreeDef): Covariances components of the mixture
        weights (PyTreeDef): The mixture weights.
        grad_state (GradState, optional): The gradient state. Defaults to GradState().
        info (INFO, optional): Hyperparameters
    """

    means: PyTreeDef
    covariances: PyTreeDef
    weights: PyTreeDef


def cdf_mixtr(mix_state: MixState, x: Array) -> Array:
    """
    Calculate the cumulative distribution function (CDF) of a mixture of Gaussian distributions.

    Args:
        mix_state (MixState): The state of the mixture model, including means, covariances,
                              and mixture weights.
        x (jnp.ndarray): The input values at which to evaluate the CDF.

    Returns:
        jnp.ndarray: The CDF values of the mixture distribution at the input points.

    Note:
        This function assumes that the mixture components are univariate Gaussian distributions.
        It uses jax.scipy.stats.norm.cdf for calculating individual CDFs.
    """
    means, covs, weights = mix_state
    stds = jnp.sqrt(covs)

    def single_cdf(mean, std, weight):
        return weight * jax.scipy.stats.norm.cdf((x - mean) / std)

    cdfs = jax.vmap(single_cdf)(means, stds, weights)
    return jnp.sum(cdfs, axis=0).squeeze()


def pdf_mixtr(mix_state: MixState, x: Array) -> Array:
    """
    Calculate the probability density function (PDF) of a multivariate normal distribution
    mixture given a state and input data.

    Args:
        state (MixState): The state of the mixture model, including means, cholesky factors,
                          mixture weights, and other parameters.
        x (Array): The input data.

    Returns:
        float: The PDF of the multivariate normal distribution mixture.
    """
    means, sigmas, weights = mix_state

    def pdf_multivariate_normal(mean, sigma):
        return jax.scipy.stats.multivariate_normal.pdf(x, mean, sigma)

    pdf = jax.vmap(pdf_multivariate_normal)(means, sigmas)
    return weights @ pdf


def rho_t(x: Array, t: Array, init_mix_state: MixState, sde: SDE) -> Array:
    """
    Compute p_t(x_t) where x_t follows the noising process defined by sde
    """
    means, covs, weights = transform_mixture_params(init_mix_state, sde, t)
    return pdf_mixtr(MixState(means, covs, weights), x)


def cdf_t(x: Array, t: Array, init_mix_state: MixState, sde: SDE) -> Array:
    """
    Compute cdf_t(x_t) where x_t follows the noising process defined by sde
    """
    means, covs, weights = transform_mixture_params(init_mix_state, sde, t)
    return cdf_mixtr(MixState(means, covs, weights), x)


def sampler_mixtr(key, state: MixState, N):
    """
    Sampler from the mixture
    """
    mu, sigma, weights = state
    d = mu.shape[-1]
    key1, key2 = jax.random.split(key)
    idx = jax.random.choice(key1, jnp.arange(len(weights)), shape=(N,), p=weights)
    noise = jax.random.normal(key2, shape=(N, d))

    chol = jnp.linalg.cholesky(sigma)
    noise_scaled = jnp.einsum("nij, nj->ni", chol[idx], noise)

    return mu[idx] + noise_scaled


# xmax = 4
nbins = 120


def transform_mixture_params(state, sde, t):
    """
    Close form solution of VP SDE for Gaussian Mixture
    """
    means, covs, weights = state
    alpha_t = sde.signal_level(t)
    sigma_t_squared = sde.noise_level(t) ** 2
    means = alpha_t * means
    covs = alpha_t**2 * covs + sigma_t_squared * jnp.eye(covs.shape[-1])
    return means, covs, weights
