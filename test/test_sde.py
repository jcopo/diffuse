import jax
import jax.numpy as jnp
import jax.random as random
from diffuse.mixture import (
    MixState,
    pdf_mixtr,
    sampler_mixtr,
    display_histogram,
    init_mixture,
)
from functools import partial

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

    # score = jax.grad(rho_t)
    score = lambda x, t: jax.grad(rho_t)(x, t) / rho_t(x, t)

    # samples from univariate gaussian
    n_samples = 5000
    # init_samples = jax.random.normal(key, (n_samples, 1))
    init_samples = jax.scipy.stats.norm.ppf(
        jnp.arange(0, n_samples) / n_samples + 1 / (2 * n_samples)
    )
    print(init_samples.shape)
    samples_mixt = sampler_mixtr(key, state, n_samples)

    t_final = 2.0
    ts = jnp.array([t_final] * n_samples)

    state_mixt = SDEState(position=samples_mixt, t=ts)

    # move to the mixture
    n_steps = 1000
    dts = jnp.array([t_final / n_steps] * (n_steps))
    print(jnp.sum(dts))
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
    sde = SDE(beta=beta)
    state_f = SDEState(position=init_samples, t=ts)
    keys = jax.random.split(key, n_samples)

    revert_sde = jax.jit(jax.vmap(partial(sde.reverso, score=score, dts=dts)))
    state_0, state_Ts = revert_sde(keys, state_f)
    sample_mixt_T = jax.vmap(sde.path, in_axes=(0, 0, None))(keys, state_mixt, dts)

    # plot samples
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    space = jnp.linspace(-3, 3, 200)
    # pdf = jax.vmap(lambda x: pdf_mixtr(state, x))(space)
    # axs[0].plot(space, pdf)
    # #display_histogram(state_f.position, axs[0])
    # display_histogram(samples_mixt, axs[0])
    # display_histogram(state_0.position, axs[1])
    # display_histogram(sample_mixt_T[0].position, axs[2])
    # #axs[2].scatter(sample_0_T[0].position, jnp.zeros_like(state_0[0].position), color='red')

    # print(state_0)

    # for t in [0.1, 0.5, 1.0, 2., 20.]:
    #     axs[1].plot(space, jax.vmap(lambda x: rho_t(x, t))(space))
    # plt.show()

    # for i in range(n_steps):
    #     fig, axs = plt.subplots()
    #     display_histogram(state_0[1].position[-i], axs)
    #     #wait
    #     plt.show(block=False)
    #     plt.pause(0.1)
    #     plt.close()

    # plot 1, 10 plots of display_histogram(state_0[1].position[t], axs[1]) in a same figute at 10 different times
    perct = [0, 0.1, 0.3, 0.7, 0.8, 0.9, 0.93, 0.9, 0.99, 1]
    n_plots = len(perct)
    fig, axs = plt.subplots(n_plots, 1, figsize=(10 * n_plots, n_plots))
    end_particles = jnp.vstack([state_f.position, state_Ts.position.T]).T
    for i, x in enumerate(perct):
        k = int(x * n_steps)
        t = k * t_final / n_steps
        display_histogram(end_particles[:, k], axs[i])
        axs[i].plot(space, jax.vmap(lambda x: rho_t(x, t_final - t))(space))
        # axs[i].scatter(end_particles[:, k], jnp.zeros_like(end_particles[:, k]), color='red')
    plt.show()


if __name__ == "__main__":
    test_mixture()
