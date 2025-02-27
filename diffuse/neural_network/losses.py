import jax
import jax.numpy as jnp
import einops
from typing import Callable
from jaxtyping import PyTreeDef, PRNGKeyArray
from diffuse.diffusion.sde import SDE, SDEState
from functools import partial


def score_match_loss(
    nn_params: PyTreeDef,
    rng_key: PRNGKeyArray,
    x0_samples,
    sde: SDE,
    network: Callable,
    lmbda: Callable = None,
):
    """
    Calculate the score matching loss. This version shares the batch of x0 and t. Meaning nt_samples and n_x0 must be the same.

    Args:
        nn_params (PRNGKeyArray): Parameters for the neural network.
        rng_key (PRNGKeyArray): Random number generator key.
        x0_samples: Samples of x0.
        sde (SDE): Stochastic differential equation.
        lmbda (Callable): Lambda function for weighting the loss.
        network (Callable): Neural network function.

    Returns:
        float: The score matching loss.

    """
    key_t, key_x = jax.random.split(rng_key)
    n_x0 = x0_samples.shape[0]
    # generate time samples
    ts = jax.random.uniform(key_t, (n_x0 - 1, 1), minval=1e-5, maxval=sde.tf)
    ts = jnp.concatenate([ts, jnp.array([[sde.tf]])], axis=0)

    # generate samples of x_t \sim p(x_t|x_0)
    state_0 = SDEState(x0_samples, jnp.zeros((n_x0, 1)))
    keys_x = jax.random.split(key_x, n_x0)
    state = jax.vmap(sde.path)(keys_x, state_0, ts)

    # nn eval
    nn_eval = network.apply(nn_params, state.position, ts)
    # evaluate log p(x_t|x_0)
    score_eval = jax.vmap(sde.score)(state, state_0)

    # reduce squared diff over all axis except batch
    sq_diff = einops.reduce(
        (nn_eval - score_eval) ** 2, "t ... -> t ", "mean"
    )  # (n_ts)

    return jnp.mean(lmbda(ts) * sq_diff, axis=0)

def weight_fun(t, sde: SDE):
    int_b = sde.beta.integrate(t, 0).squeeze()
    return 1 - jnp.exp(-int_b)


def noise_match_loss(
    nn_params: PyTreeDef,
    rng_key: PRNGKeyArray,
    x0_samples,
    sde: SDE,
    network: Callable,
):
    """
    Calculate the noise matching loss for learning to predict noise directly.
    This version shares the batch of x0 and t.

    Args:
        nn_params (PRNGKeyArray): Parameters for the neural network.
        rng_key (PRNGKeyArray): Random number generator key.
        x0_samples: Samples of x0.
        sde (SDE): Stochastic differential equation.
        tf (float): Final time.
        network (Callable): Neural network that predicts noise.

    Returns:
        float: The noise matching loss.
    """
    key_t, key_x = jax.random.split(rng_key)
    n_x0 = x0_samples.shape[0]

    # Generate time samples
    ts = jax.random.randint(key_t, (n_x0, 1), minval=0, maxval=100)
    ts = ts * (sde.tf / 1000)

    # Generate samples of x_t and get the noise used
    state_0 = SDEState(x0_samples, jnp.zeros((n_x0, 1)))
    keys_x = jax.random.split(key_x, n_x0)
    state, noise = jax.vmap(partial(sde.path, return_noise=True))(keys_x, state_0, ts)

    # Predict noise with neural network
    noise_pred = network.apply(nn_params, state.position, ts)

    # Calculate MSE between predicted and true noise
    mse = jnp.mean((noise_pred - noise) ** 2, axis=list(range(1, noise.ndim)))

    return jnp.mean(mse)

def kl_loss(nn_params, rng_key, x0_samples, beta: float, network: Callable):
    posterior, sample = network.apply(nn_params, x0_samples, rngs=rng_key)
    kl_value = posterior.kl()
    mse_value = einops.reduce((sample - x0_samples) ** 2, "t ... -> t ", "mean")
    return jnp.mean(beta * kl_value + mse_value)