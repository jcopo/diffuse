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


def rho_t(x, t):
    means, covs, weights = state
    int_b = sde.beta.integrate(t, 0.0)
    alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)
    means = alpha * means
    covs = alpha**2 * covs + beta
    return pdf_mixtr(MixState(means, covs, weights), x)


# score = jax.grad(rho_t)
score = lambda x, t: jax.grad(rho_t)(x, t) / rho_t(x, t)


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
        jax.random.uniform(key_t, (nt_samples, 1), minval=1e-5, maxval=tf), axis=0
    )
    # ts = jax.scipy.stats.uniform.ppf(jnp.arange(0,nt_samples)/nt_samples + 1/(2*nt_samples))
    # (n_x0, n_ts, ...)
    state_0 = SDEState(x0_samples, jnp.zeros((n_x0, 1)))
    # p(xt|x0), (n_x0, n_ts, ...)
    keys_x = jax.random.split(key_x, n_x0)
    # _, conditional_path = jax.vmap(sde.path, in_axes=(0, 0, None))(keys_x, state_0, dts)
    # all_paths = conditional_path.position

    state = jax.vmap(sde.path, in_axes=(0, 0, 0))(keys_x, state_0, ts)
    all_paths = state.position

    # None, (n_x0, n_ts,, ...), (n_ts,) -> (n_x0, n_ts, ...)
    # nn_eval = jax.vmap(network.apply, in_axes=(None, 0, 0))(
    #     nn_params, all_paths, ts
    # )
    nn_eval = network.apply(nn_params, all_paths, ts)
    score_eval = jax.vmap(sde.score)(state, state_0)
    pdb.set_trace()
    # (n_x0, n_ts, ...)
    # state = SDEState(all_paths, einops.repeat(ts, "n -> new_axis n", new_axis=n_x0))
    # (n_x0, n_ts, ...), (n_x0, n_ts, ...) -> (n_x0, n_ts, ...)
    # score_eval = jax.vmap(sde.score, in_axes=(1, None), out_axes=1)(state, state_0)

    sq_diff = einops.reduce(
        (nn_eval - score_eval) ** 2, "t n ... -> t ", "mean"
    )  # (n_ts)

    # mean_sq_diff = jnp.mean(sq_diff, axis=0)  # (n_ts, ...)

    return jnp.mean(lmbda(ts) * sq_diff, axis=0)
    # return jnp.mean(lmbda(ts) * mean_sq_diff, axis=0)


if __name__ == "__main__":
    model = MLP([300, 100, 10, 1])
    x = jnp.ones((1, 1))
    t = jnp.ones((1, 1))
    init = model.init(jax.random.PRNGKey(0), x, t)
    res = model.apply(init, x, t)

    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
    sde = SDE(beta)
    key = jax.random.PRNGKey(0)

    n_samples = 500
    mixt_state = init_mixture(key)
    samples_mixt = sampler_mixtr(key, mixt_state, n_samples)

    ls = score_match_loss(
        init,
        key,
        samples_mixt,
        sde,
        100,
        2.0,
        lambda x: jnp.ones(x.shape).squeeze(),
        model,
    )

    def weight_fun(t):
        int_b = sde.beta.integrate(t, 0).squeeze()
        return 1 - jnp.exp(-int_b)

    def loss_(nn_params, key):
        samples_mixt = sampler_mixtr(key, mixt_state, n_samples)
        return score_match_loss(
            nn_params,
            key,
            samples_mixt,
            sde,
            200,
            2.0,
            jax.vmap(weight_fun),
            model,
        ).squeeze()

    tx = optax.adam(1e-3)

    @jax.jit
    def train_step(nn_params, opt_state, key):
        g = jax.grad(loss_)(nn_params, key)
        updates, new_state = tx.update(g, opt_state)
        return updates, new_state

    opt_state = tx.init(init)
    nn_params = init
    key = jax.random.PRNGKey(0)
    for i in range(1000):
        key, subkey = jax.random.split(key)
        updates, opt_state = train_step(nn_params, opt_state, subkey)
        nn_params = optax.apply_updates(nn_params, updates)
        print(loss_(nn_params, key))

    pdb.set_trace()
