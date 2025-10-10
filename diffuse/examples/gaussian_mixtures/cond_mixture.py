# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""
Conditional Gaussian Mixture Models for Diffusion-Based Inverse Problems

This module implements Bayesian inference with Gaussian mixture models (GMMs)
in the context of diffusion processes and linear inverse problems.

Mathematical Background:
======================

1. GAUSSIAN MIXTURE MODEL (GMM)
   A GMM represents a probability distribution as a weighted sum of K Gaussian components:

   p(x) = Σᵢ₌₁ᴷ wᵢ N(x; μᵢ, Σᵢ)

   where:
   - wᵢ ≥ 0, Σᵢ wᵢ = 1 (mixture weights)
   - μᵢ ∈ ℝᵈ (component means)
   - Σᵢ ∈ ℝᵈˣᵈ (component covariance matrices)

2. DIFFUSION PROCESS
   The diffusion process follows the SDE:
   dx(t) = -0.5 β(t) x(t) dt + √β(t) dW(t)

   The solution has the form:
   x(t) = √α(t) x₀ + √(1-α(t)) ε,  where ε ~ N(0, I)

   where α(t) = exp(-∫₀ᵗ β(s) ds) is the signal preservation ratio.

3. CLOSED-FORM SOLUTION FOR GMM + DIFFUSION
   When the prior is a GMM, the diffused distribution remains a GMM:

   pₜ(xₜ) = Σᵢ wᵢ N(xₜ; μᵢ(t), Σᵢ(t))

   where:
   - μᵢ(t) = √α(t) μᵢ(0)
   - Σᵢ(t) = α(t) Σᵢ(0) + (1-α(t)) I
   - wᵢ(t) = wᵢ(0) (weights unchanged)

4. BAYESIAN POSTERIOR WITH LINEAR MEASUREMENTS
   Given measurement: y = Ax + ε, where ε ~ N(0, σ_y² I)

   The posterior is also a GMM:
   p(x|y) = Σᵢ w̄ᵢ N(x; μ̄ᵢ, Σ̄)

   where:
   - Σ̄ = (I + (1/σ_y²) A^T A)^(-1)
   - μ̄ᵢ = Σ̄((1/σ_y²) A^T y + μᵢ)
   - w̄ᵢ ∝ wᵢ × p(y|μᵢ)
   - p(y|μᵢ) = N(y; Aμᵢ, σ_y² + AA^T)

This conjugacy property makes GMMs particularly useful for diffusion-based
inverse problems, as the posterior can be computed analytically at each step.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from jax import Array
from jaxtyping import PRNGKeyArray

from diffuse.diffusion.sde import SDE
from diffuse.examples.gaussian_mixtures.mixture import (
    MixState,
)

# float64 accuracy for numerical stability
jax.config.update("jax_enable_x64", True)


@dataclass
class NoiseMask:
    """
    Represents a linear measurement operator with additive noise.

    This implements the measurement model: y = Ax + ε, where ε ~ N(0, σ²I)

    Attributes:
        A: Measurement matrix of shape (m, d) where m is number of measurements
        alpha: Scaling factor for the measurement strength
        std: Standard deviation of measurement noise
    """

    A: Array
    alpha: float
    std: float

    def measure(self, key: PRNGKeyArray, x: Array) -> Array:
        """
        Apply the measurement operator: y = Ax + ε

        Args:
            key: JAX PRNG key for noise generation
            x: Input signal of shape (d,)

        Returns:
            y: Measurement of shape (m,)
        """
        return self.A @ x + jax.random.normal(key, shape=x.shape) * self.std

    def restore(self, measured: Array) -> Array:
        """
        Apply the adjoint operator: x̃ = A^T y

        Note: This is NOT the inverse operation, just the adjoint.
        For proper reconstruction, use Bayesian inference.

        Args:
            measured: Measurement y of shape (m,)

        Returns:
            x̃: Adjoint result of shape (d,)
        """
        return self.A.T @ measured


def compute_xt_given_y(mix_state_posterior: MixState, sde: SDE, t: float):
    """
    Transform a Gaussian mixture through the diffusion process.

    Given a GMM at time 0: p₀(x) = Σᵢ wᵢ N(x; μᵢ, Σᵢ)
    Compute the GMM at time t: pₜ(xₜ) = Σᵢ wᵢ N(xₜ; μᵢ(t), Σᵢ(t))

    Mathematical derivation:
    - The diffusion process preserves the mixture structure
    - Each component transforms as: N(x; μ, Σ) → N(xₜ; √αₜ μ, αₜ Σ + (1-αₜ)I)
    - Weights remain unchanged: wᵢ(t) = wᵢ(0)

    Args:
        mix_state_posterior: GMM state (means, covs, weights)
        sde: Stochastic differential equation defining the diffusion
        t: Time parameter

    Returns:
        MixState: Transformed GMM parameters
    """
    means, covs, weights = mix_state_posterior

    # Compute signal preservation ratio: α(t) = exp(-∫₀ᵗ β(s) ds)
    # alpha_t = jnp.exp(-sde.beta.integrate(t, 0.0))
    alpha_t = sde.signal_level(t)

    # Transform means: μᵢ(t) = αₜ μᵢ(0)
    means_xt = alpha_t * means

    # Transform covariances: Σᵢ(t) = αₜ² Σᵢ(0) + σₜ²I
    sigma_t_squared = sde.noise_level(t) ** 2
    covs_xt = alpha_t**2 * covs + sigma_t_squared * jnp.eye(covs.shape[-1])

    return MixState(means_xt, covs_xt, weights)


def compute_posterior(mix_state: MixState, y: Array, A: Array, sigma_y=0.05):
    """
    Compute the Bayesian posterior for a GMM given linear measurements.

    Measurement model: y = Ax + ε, where ε ~ N(0, σ_y² I)
    Prior: p(x) = Σᵢ wᵢ N(x; μᵢ, Σᵢ)
    Posterior: p(x|y) = Σᵢ w̄ᵢ N(x; μ̄ᵢ, Σ̄)

    Mathematical derivation:
    1. Posterior covariance: Σ̄ = (I + (1/σ_y²) A^T A)^(-1)
    2. Posterior means: μ̄ᵢ = Σ̄((1/σ_y²) A^T y + μᵢ)
    3. Posterior weights: w̄ᵢ ∝ wᵢ × p(y|μᵢ)
       where p(y|μᵢ) = N(y; Aμᵢ, σ_y² + AA^T)

    This exploits the conjugacy of Gaussian likelihoods with GMM priors.

    Args:
        mix_state: Prior GMM state (means, covs, weights)
        y: Measurement vector of shape (m,)
        A: Measurement matrix of shape (m, d)
        sigma_y: Measurement noise standard deviation

    Returns:
        MixState: Posterior GMM parameters
    """
    means, covs, weights = mix_state
    means.shape[-1]

    # Step 1: Compute posterior covariance matrix for each component
    # Correct formula: Σ̄ᵢ = (Σᵢ⁻¹ + (1/σ_y²) A^T A)^(-1)
    A @ A.T  # Shape: (m, m) - for scalar measurements this is (1, 1)
    measurement_precision = (1 / sigma_y**2) * (A.T @ A)  # Shape: (d, d)

    # Compute posterior covariance for each component
    def compute_posterior_cov(prior_cov):
        prior_precision = jnp.linalg.inv(prior_cov)
        posterior_precision = prior_precision + measurement_precision
        return jnp.linalg.inv(posterior_precision)

    covs_bar = jax.vmap(compute_posterior_cov)(covs)  # Shape: (K, d, d)

    # Step 2: Compute posterior means
    # μ̄ᵢ = Σ̄ᵢ(Σᵢ⁻¹μᵢ + (1/σ_y²) A^T y)
    measurement_term = (1 / sigma_y**2) * (A.T @ y)  # Shape: (d,)

    def compute_posterior_mean(prior_mean, prior_cov, posterior_cov):
        prior_precision = jnp.linalg.inv(prior_cov)
        precision_weighted_mean = prior_precision @ prior_mean + measurement_term
        return posterior_cov @ precision_weighted_mean

    means_bar = jax.vmap(compute_posterior_mean)(means, covs, covs_bar)  # Shape: (K, d)

    # Step 3: Compute posterior weights
    # w̄ᵢ ∝ wᵢ × p(y|μᵢ, Σᵢ)
    # where p(y|μᵢ, Σᵢ) = N(y; Aμᵢ, σ_y²I + AΣᵢA^T)
    def compute_likelihood(prior_mean, prior_cov):
        y_pred = A @ prior_mean  # Shape: (m,)
        likelihood_cov = sigma_y**2 * jnp.eye(A.shape[0]) + A @ prior_cov @ A.T  # Shape: (m, m)
        return stats.multivariate_normal.logpdf(y, mean=y_pred, cov=likelihood_cov)

    log_likelihood = jax.vmap(compute_likelihood)(means, covs)  # Shape: (K,)

    # Unnormalized posterior weights
    weights_bar = weights * jnp.exp(log_likelihood)  # Shape: (K,)

    # Normalize to ensure they sum to 1
    weights_bar = weights_bar / jnp.sum(weights_bar)

    return MixState(means_bar, covs_bar, weights_bar)
