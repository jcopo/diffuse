from dataclasses import dataclass

import jax
from jax import Array
from jax.random import PRNGKeyArray

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
    means, covs, weights = mix_state
    alpha, std = noise_mask

    new_covs = (covs**(-1) + alpha**2 / std**2)**(-1)
    new_means = new_covs * (means / covs + alpha * y_meas / std**2)

