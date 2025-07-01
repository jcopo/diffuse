from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from jax import Array
from jaxtyping import PRNGKeyArray

from diffuse.diffusion.sde import SDE
from examples.gaussian_mixtures.mixture import (
    MixState,
)

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


# Import initialization from the dedicated module

# Generate observation
# def generate_observation(key, mix_state: MixState, d=20, sigma_y=0.05):
#     key_A, key_x, key_noise = jax.random.split(key, 3)
#     x_star = sampler_mixtr(key_x, mix_state, 1)[0]  # Shape: (d,)
#     A = jax.random.normal(key_A, (1, d))  # Shape: (1, d)
#     epsilon = jax.random.normal(key_noise, (1,))
#     y = A @ x_star + sigma_y * epsilon  # Shape: (1,)
#     return y, A, x_star


def compute_xt_given_y(mix_state_posterior: MixState, sde: SDE, t: float):
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
