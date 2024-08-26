import pdb
from diffuse.sde import SDE, SDEState, LinearSchedule
from diffuse.neural_networks import MLP
from diffuse.mixture import display_trajectories
from jaxtyping import PyTreeDef, PRNGKeyArray
import jax.numpy as jnp
import jax
from typing import Callable
import einops
import matplotlib.pyplot as plt
import optax

from diffuse.mixture import init_mixture, sampler_mixtr


def score_match_loss(
    nn_params: PyTreeDef,
    rng_key: PRNGKeyArray,
    x0_samples,
    sde: SDE,
    nt_samples: int,
    tf: float,
    lmbda: Callable,
    network: Callable,
):
    """
    Calculate the score matching loss. This version shares the batch of x0 and t. Meaning nt_samples and n_x0 must be the same.

    Args:
        nn_params (PRNGKeyArray): Parameters for the neural network.
        rng_key (PRNGKeyArray): Random number generator key.
        x0_samples: Samples of x0.
        sde (SDE): Stochastic differential equation.
        nt_samples (int): Number of time samples.
        tf (float): Final time.
        lmbda (Callable): Lambda function for weighting the loss.
        network (Callable): Neural network function.

    Returns:
        float: The score matching loss.

    """
    key_t, key_x = jax.random.split(rng_key)
    n_x0 = x0_samples.shape[0]
    # generate time samples
    ts = jax.random.uniform(key_t, (nt_samples - 1, 1), minval=1e-5, maxval=tf)
    ts = jnp.concatenate([ts, jnp.array([[tf]])], axis=0)

    # generate samples of x_t \sim p(x_t|x_0)
    state_0 = SDEState(x0_samples, jnp.zeros((n_x0, 1)))
    keys_x = jax.random.split(key_x, n_x0)
    state = jax.vmap(sde.path, in_axes=(0, 0, 0))(keys_x, state_0, ts)

    # nn eval
    nn_eval = network.apply(nn_params, state.position, ts)
    # evaluate log p(x_t|x_0)
    score_eval = jax.vmap(sde.score)(state, state_0)

    # reduce squared diff over all axis except batch
    sq_diff = einops.reduce(
        (nn_eval - score_eval) ** 2, "t ... -> t ", "mean"
    )  # (n_ts)

    return jnp.mean(lmbda(ts) * sq_diff, axis=0)
