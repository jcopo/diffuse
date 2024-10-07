import pdb
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from diffuse.sde import SDEState
from diffuse.mixture import (
    MixState,
    cdf_t,
    pdf_mixtr,
    rho_t,
    sampler_mixtr,
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
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
    sde = SDE(beta=beta)
    return sde


def make_mixture():
    key = jax.random.PRNGKey(666)
    state = init_mixture(key, d=2)
    return state


def run_forward_evolution_animation(sde, init_mix_state, num_frames=100, interval=500):
    key = jax.random.PRNGKey(666)
    pdf = partial(rho_t, init_mix_state=init_mix_state, sde=sde)
    score = lambda x, t: jax.grad(pdf)(x, t) / pdf(x, t)

    # sample mixture
    num_samples = 500
    samples = sampler_mixtr(key, init_mix_state, num_samples)


    # Create 2D grid
    space = jnp.linspace(-5, 5, 100)
    x, y = jnp.meshgrid(space, space)
    xy = jnp.stack([x, y], axis=-1)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    contour = ax.contourf(x, y, jnp.zeros_like(x))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=12)

    state = SDEState(position=samples, t=jnp.zeros((num_samples, 1)))

    def update(frame):
        t = frame / num_frames * sde.beta.T
        pdf_grid = jax.vmap(jax.vmap(pdf, in_axes=(0, None)), in_axes=(0, None))(xy, t)

        ax.clear()
        contour = ax.contourf(x, y, pdf_grid, levels=20, zorder=-1)
        # Update sample positions based on the SDE
        key = jax.random.PRNGKey(frame)  # Use frame as seed for reproducibility
        samples = sde.path(key, state, jnp.array([t])).position.squeeze()

        # Plot updated samples
        scatter = ax.scatter(samples[:, 0], samples[:, 1], zorder=1, marker='o', s=10, c='k')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.axis('off')

        # Update time text
        time_text = ax.text(0.02, 0.98, f'Time: {t:.2f}', transform=ax.transAxes, va='top', fontsize=12)

        return scatter, contour, time_text

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
    plt.show()


def run_backward_evolution_animation(sde, init_mix_state, num_frames=100, interval=500):
    key = jax.random.PRNGKey(666)
    pdf = partial(rho_t, init_mix_state=init_mix_state, sde=sde)
    score = lambda x, t: jax.grad(pdf)(x, t) / pdf(x, t)

    # sample from the diffused state (t=T)
    num_samples = 500
    T = sde.beta.T
    diffused_samples = sampler_mixtr(key, init_mix_state, num_samples)
    diffused_samples = sde.path(key, SDEState(position=diffused_samples, t=jnp.zeros((num_samples, 1))), jnp.array([T])).position.squeeze()

    # Create 2D grid
    space = jnp.linspace(-5, 5, 100)
    x, y = jnp.meshgrid(space, space)
    xy = jnp.stack([x, y], axis=-1)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    contour = ax.contourf(x, y, jnp.zeros_like(x))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=12)

    state = SDEState(position=diffused_samples, t=T * jnp.ones((num_samples, 1)))

    def update(frame):
        t = T - (frame / num_frames * T)
        pdf_grid = jax.vmap(jax.vmap(pdf, in_axes=(0, None)), in_axes=(0, None))(xy, t)

        ax.clear()
        contour = ax.contourf(x, y, pdf_grid, levels=20, zorder=-1)

        # Update sample positions based on the reverse SDE
        key = jax.random.PRNGKey(frame)  # Use frame as seed for reproducibility
        samples = sde.reverso(key, state, jnp.array([t])).position.squeeze()

        # Plot updated samples
        scatter = ax.scatter(samples[:, 0], samples[:, 1], zorder=1, marker='o', s=10, c='k')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.axis('off')

        # Update time text
        time_text = ax.text(0.02, 0.98, f'Time: {t:.2f}', transform=ax.transAxes, va='top', fontsize=12)

        return scatter, contour, time_text

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
    plt.show()

if __name__ == "__main__":
    sde = make_sde()
    state = make_mixture()
    #run_forward_evolution_animation(sde, state)
    run_backward_evolution_animation(sde, state)



if __name__ == "__main__":
    sde = make_sde()
    state = make_mixture()
    # run_forward_evolution_animation(sde, state)
    run_backward_evolution_animation(sde, state)
