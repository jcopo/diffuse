from dataclasses import dataclass
import pdb

import jax
from jax import Array
from jaxtyping import PRNGKeyArray

from examples.gaussian_mixtures.mixture import MixState

@dataclass
class NoiseMask:
    alpha: float
    std: float

    def measure(self, key: PRNGKeyArray, x: Array) -> Array:
        return self.alpha * x + jax.random.normal(key, shape=x.shape) * self.std

    def restore(self, key: PRNGKeyArray, x: Array, measured: Array) -> Array:
        return self.alpha * x


def posterior_distribution(mix_state: MixState, noise_mask: NoiseMask, y_meas: Array) -> MixState:
    r"""
    Close form of p(theta | y) for y = alpha * theta + noise and theta ~ \sum_i w_i N(means[i], covs[i])
    """
    means, covs, weights = mix_state
    alpha, std = noise_mask.alpha, noise_mask.std

    new_covs = (covs**(-1) + alpha**2 / std**2)**(-1)
    new_means = new_covs[:, 0] * (means / covs[:, 0] + alpha * y_meas / std**2)
    new_weights = weights * jax.scipy.stats.norm.pdf(y_meas, alpha * means[:, 0], std)

    new_weights = new_weights / new_weights.sum()
    return MixState(new_means, new_covs, new_weights)
