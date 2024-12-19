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


def posterior_distribution(mix_state: MixState, noise_mask: NoiseMask) -> MixState:
    means, covs, weights = mix_state

