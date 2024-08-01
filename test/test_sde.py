import jax
import jax.numpy as jnp
import jax.random as random
import pytest
from diffuse.mixture import MixState, pdf_mixtr, sampler_mixtr, display_histogram

from diffuse.sde import SDE, SDEState, LinearSchedule
import matplotlib.pyplot as plt


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
    key = jax.random.PRNGKey(1)
    n_mixt = 3
    means = 4*jax.random.normal(key, (n_mixt, 1))
    covs = 2*jax.random.normal(key, (n_mixt, 1))
    mix_weights = jax.random.uniform(key, (n_mixt,))
    mix_weights /= jnp.sum(mix_weights)

    state = MixState(means, covs, mix_weights)
    score = jax.grad(lambda x: pdf_mixtr(state, x))

    # samples from univariate gaussian
    n_samples = 1000
    init_samples = jax.random.normal(key, (n_samples, 1))
    
    # move to the mixture
    dts = jnp.array([0.0001] * 1000)
    beta = LinearSchedule(b_min=0.5, b_max=1.5, t0=0.0, T=1.0)
    sde = SDE(beta=beta)
    state_f = SDEState(position=init_samples, t=1.)
    state_0 = sde.reverso(key, state_f, score, dts)

    # plot samples
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    space = jnp.linspace(-10, 10, 200)
    pdf = jax.vmap(lambda x: pdf_mixtr(state, x))(space)
    axs[0].plot(space, pdf)
    display_histogram(state_f.position, axs[0])

    display_histogram(state_0.position, axs[1])
    plt.show()