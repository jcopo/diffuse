from flax import linen as nn
from typing import Sequence


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x, t):
        t = nn.Dense(self.features[0])(t)
        x = nn.Dense(self.features[0])(x)
        x = x + t
        for i, feat in enumerate(self.features[1:]):
            x = nn.Dense(feat)(x)
            if i != len(self.features) - 1:
                x = nn.sigmoid(x)
        return x
