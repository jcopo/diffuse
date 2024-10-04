import jax
import jax.numpy as jnp
import jax.random as random
from diffuse.mixture import (
    MixState,
    pdf_mixtr,
    sampler_mixtr,
    display_histogram,
    init_mixture,
    display_trajectories,
)
from functools import partial
import einops

import pdb
from diffuse.sde import SDE, SDEState, LinearSchedule
import matplotlib.pyplot as plt


jax.config.update("jax_enable_x64", True)

# def test_score():
#     beta = LinearSchedule(b_min=0.5, b_max=1.5, t0=0.0, T=1.0)
#     sde = SDE(beta=beta)

#     state = SDEState(position=jnp.array([1.0, 2.0]), t=0.0)
#     state_0 = SDEState(position=jnp.array([0.0, 0.0]), t=0.0)

#     expected_score = jnp.array([-0.5, -1.0])
#     assert jnp.allclose(sde.score(state, state_0), expected_score)


def test_path():
    beta = LinearSchedule(b_min=0.5, b_max=1.5, t0=0.0, T=1.0)
    sde = SDE(beta=beta)

    key = random.PRNGKey(0)
    state = SDEState(position=jnp.array([1.0, 2.0]), t=0.0)
    dt = 0.1

    new_state = sde.path(key, state, dt)

    assert new_state.t == state.t + dt
    assert new_state.position.shape == state.position.shape

    # Additional assertions for the position values can be added here


def test_mixture():
    key = jax.random.PRNGKey(666)
    state = init_mixture(key)
    print(state)

    def rho_t(x, t):
        means, covs, weights = state
        int_b = sde.beta.integrate(t, 0.0)
        alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)
        means = alpha * means
        covs = alpha**2 * covs + beta
        return pdf_mixtr(MixState(means, covs, weights), x)

    # score
    score = lambda x, t: jax.grad(rho_t)(x, t) / rho_t(x, t)

    # samples from univariate gaussian
    n_samples = 5000

    t_final = 2.0

    # move to the mixture
    n_steps = 100
    dts = jnp.array([t_final / n_steps] * (n_steps))
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
    sde = SDE(beta=beta)

    # reverse rpoceess
    init_samples = jax.scipy.stats.norm.ppf(
        jnp.arange(0, n_samples) / n_samples + 1 / (2 * n_samples)
    )[:, None]
    tf = jnp.array([t_final] * n_samples)
    state_f = SDEState(position=init_samples, t=tf)
    keys = jax.random.split(key, n_samples)
    revert_sde = jax.jit(jax.vmap(partial(sde.reverso, score=score, dts=dts)))
    state_0, state_Ts = revert_sde(keys, state_f)

    # noise proccess
    ts = jnp.linspace(0, t_final, n_steps)
    keys = jax.random.split(key, n_samples * n_steps).reshape((n_samples, n_steps, -1))
    samples_mixt = sampler_mixtr(key, state, n_samples)
    t0 = jnp.array([0.0] * n_samples)
    state_mixt = SDEState(position=samples_mixt, t=t0)
    sample_mixt_T = jax.vmap(
        jax.vmap(sde.path, in_axes=(0, None, 0)), in_axes=(0, 0, None)
    )(keys, state_mixt, ts)

    # plot samples
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    space = jnp.linspace(-3, 3, 200)

    display_trajectories(sample_mixt_T.position.squeeze(), 100)
    plt.show()
    display_trajectories(state_Ts.position.squeeze(), 100)
    plt.show()

    # PLOT FORWARD TRAJECTORIES
    perct = [0, 0.03, 0.06, 0.08, 0.1, 0.3, 0.7, 0.8, 0.9, 1]
    n_plots = len(perct)
    fig, axs = plt.subplots(n_plots, 1, figsize=(10 * n_plots, n_plots))
    end_particles = sample_mixt_T.position
    for i, x in enumerate(perct):
        k = int(x * n_steps)
        t = k * t_final / n_steps
        display_histogram(end_particles[:, k], axs[i])
        state_t = SDEState(position=end_particles[:, k], t=jnp.array([t] * n_samples))
        # axs[i].plot(space, jax.vmap(lambda x: sde.score(SDEState(position=x, t=jnp.array([t])),
        #                                                 state_mixt))(space))

        axs[i].plot(space, jax.vmap(lambda x: rho_t(x, t))(space))
        # axs[i].scatter(end_particles[:, k], jnp.zeros_like(end_particles[:, k]), color='red')
    plt.show()

    # PLOT BACKWARD TRAJECTORIES
    perct = [0, 0.1, 0.3, 0.7, 0.8, 0.9, 0.93, 0.9, 0.99, 1]
    n_plots = len(perct)
    fig, axs = plt.subplots(n_plots, 1, figsize=(10 * n_plots, n_plots))
    end_particles = state_Ts.position
    for i, x in enumerate(perct):
        k = int(x * n_steps)
        t = k * t_final / n_steps
        display_histogram(end_particles[:, k], axs[i])
        axs[i].plot(space, jax.vmap(lambda x: rho_t(x, t_final - t))(space))
        # axs[i].scatter(end_particles[:, k], jnp.zeros_like(end_particles[:, k]), color='red')
    plt.show()


if __name__ == "__main__":
    test_mixture()
