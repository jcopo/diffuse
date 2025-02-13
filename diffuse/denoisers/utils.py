import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array


def ess(log_weights: Array) -> float:
    return jnp.exp(log_ess(log_weights))


def log_ess(log_weights: Array) -> float:
    return 2 * jsp.special.logsumexp(log_weights) - jsp.special.logsumexp(
        2 * log_weights
    )


def normalize_log_weights(log_weights: Array) -> Array:
    return jax.nn.log_softmax(log_weights, axis=0)
