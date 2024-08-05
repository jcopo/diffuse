from flax import linen as nn
from typing import Sequence
import jax.numpy as jnp


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x, t):
        x = jnp.concatenate([x, t], axis=-1)
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x
