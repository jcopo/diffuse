import pdb
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from diffuse.mixture import (
    MixState,
    cdf_t,
    pdf_mixtr,
    rho_t,
    transform_mixture_params,
)
from diffuse.sde import SDE, LinearSchedule


def init_mixture(key, d=1):
    # Means
    means = jnp.array([
        [-1.0, -1.0],  # Bottom-left
        [1.0, 1.0],    # Top-right
        [2.0, -2.0]    # Bottom-right
    ])

    # Covariances
    covs = 1.5*jnp.array([
        [[0.5, 0.3], [0.3, 0.5]],     # Slightly correlated
        [[0.7, -0.2], [-0.2, 0.7]],   # Slightly anti-correlated
        [[.3, 0.0], [0.0, 1.0]]      # Stretched vertically
    ])

    # Weights
    weights = jnp.array([0.3, 0.4, 0.3])  # Slightly uneven weights

    return MixState(means, covs, weights)

    n_mixt = 3
    #means = jax.random.uniform(key, (n_mixt, d), minval=-3, maxval=3)
    means = jnp.array([[0.0, 0.0], [2.0, 2.0], [-2.0, -2.0]])
    covs = jnp.array(
        [
            [[1.0, -0.1], [-0.1, 1.0]],
            [[1.0, 0.8], [0.8, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    mix_weights = jax.random.uniform(key + 2, (n_mixt,))
    mix_weights /= jnp.sum(mix_weights)

    return MixState(means, covs, mix_weights)


def make_sde():
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=5.0)
    sde = SDE(beta=beta)
    return sde


def make_mixture():
    key = jax.random.PRNGKey(666)
    state = init_mixture(key, d=2)
    return state


def run_time_evolution_animation(sde, init_mix_state, num_frames=100, interval=50):
    pdf = partial(rho_t, init_mix_state=init_mix_state, sde=sde)
    score = lambda x, t: jax.grad(pdf)(x, t) / pdf(x, t)

    # Create 2D grid
    space = jnp.linspace(-5, 5, 100)
    x, y = jnp.meshgrid(space, space)
    xy = jnp.stack([x, y], axis=-1)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    contour = ax.contourf(x, y, jnp.zeros_like(x))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_title("Mixture Evolution")

    def update(frame):
        t = frame / num_frames * sde.beta.T
        pdf_grid = jax.vmap(jax.vmap(pdf, in_axes=(0, None)), in_axes=(0, None))(xy, t)

        ax.clear()
        contour = ax.contourf(x, y, pdf_grid, levels=20, cmap='viridis')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_title(f"Mixture Evolution (t = {t:.2f})")
        return contour,

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
    plt.show()

if __name__ == "__main__":
    sde = make_sde()
    state = make_mixture()
    run_time_evolution_animation(sde, state)
