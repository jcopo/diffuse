from typing import Optional

from flax import nnx
from jax import Array
from jax.typing import ArrayLike


class TimestepBlock(nnx.Module):
    def __call__(self, x: ArrayLike, time_emb: Optional[ArrayLike] = None) -> Array:
        pass


class TimestepEmbedSequential(nnx.Sequential, TimestepBlock):
    def __call__(self, x: ArrayLike, time_emb: Optional[ArrayLike] = None) -> Array:
        for layer in self.layers:
            if isinstance(layer, TimestepBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        return x
