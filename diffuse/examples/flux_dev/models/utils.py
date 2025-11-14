from jax import Array
from jaxtyping import ArrayLike


def modulate(x: ArrayLike, shift: ArrayLike, scale: ArrayLike, base: float = 1.0) -> Array:
    shift = shift[:, None, :]
    scale = scale[:, None, :]
    return x * (base + scale) + shift
