import pdb
from diffuse.sde import SDE, SDEState, LinearSchedule
from diffuse.neural_networks import MLP
from jaxtyping import PyTreeDef, PRNGKeyArray
import jax.numpy as jnp
import jax
from typing import Callable

from diffuse.mixture import init_mixture, sampler_mixtr


def loss(
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
    Calculate the score matching loss.

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
    # sample one x_t for each x_0 x t

    key_t, key_x = jax.random.split(rng_key)
    # p(x0), n_x0
    n_x0 = x0_samples.shape[0]
    # t \sim U[0, T], (n_t, )
    ts = jnp.sort(
        jax.random.uniform(key_t, (nt_samples, 1), minval=0, maxval=tf), axis=0
    )
    # ts = jax.scipy.stats.uniform.ppf(jnp.arange(0,nt_samples)/nt_samples + 1/(2*nt_samples))
    dts = ts[1:] - ts[:-1]
    # (n_ts, n_x0, ...)
    state_0 = SDEState(x0_samples, jnp.zeros((n_x0, 1)))
    # p(xt|x0), (n_t, n_x0, ...)
    keys_x = jax.random.split(key_x, n_x0)
    _, conditional_path = jax.vmap(sde.path, in_axes=(0, 0, None))(keys_x, state_0, dts)

    # None, (n_ts, n_x0, ...), (n_ts,) -> (n_ts, n_x0 ...)
    pdb.set_trace()
    nn_eval = network.apply(nn_params, conditional_path, ts)

    # (n_ts, n_x0, ...)
    state = SDEState(conditional_path, jnp.repeat(ts, n_x0, axis=0))
    # (n_ts, n_x0, ...), (n_ts, n_x0, ...) -> (n_ts, n_x0, ...)
    score_eval = sde.score(state, state_0)

    sq_diff = (nn_eval - score_eval) ** 2  # (n_ts, n_x0, ...)
    mean_sq_diff = jnp.mean(sq_diff, axis=1)  # (n_ts, ...)

    return jnp.mean(lmbda(ts) * mean_sq_diff, axis=0)


if __name__ == "__main__":
    model = MLP([20, 32, 4])
    x = jnp.ones((1, 1))
    t = jnp.ones((1, 1))
    init = model.init(jax.random.PRNGKey(0), x, t)
    res = model.apply(init, x, t)

    sde = SDE(beta=LinearSchedule(0.1, 1.0, 0.0, 1.0))
    key = jax.random.PRNGKey(0)

    n_samples = 500
    mixt_state = init_mixture(key)
    samples_mixt = sampler_mixtr(key, mixt_state, n_samples)

    ls = loss(
        init, key, samples_mixt, sde, 100, 2.0, lambda x: jnp.ones(x.shape), model
    )

    pdb.set_trace()
