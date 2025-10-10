# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from diffuse.diffusion.sde import SDEState
from diffuse.examples.gaussian_mixtures.mixture import (
    MixState,
    rho_t,
    sampler_mixtr,
)
from diffuse.diffusion.sde import SDE, LinearSchedule


def init_mixture(key, d=1):
    # Means
    means = jnp.array(
        [
            [-1.0, -1.0],  # Bottom-left
            [1.0, 1.0],  # Top-right
            [2.0, -2.0],  # Bottom-right
        ]
    )

    # Covariances
    covs = 1.5 * jnp.array(
        [
            [[0.5, 0.3], [0.3, 0.5]],  # Slightly correlated
            [[0.7, -0.2], [-0.2, 0.7]],  # Slightly anti-correlated
            [[0.3, 0.0], [0.0, 1.0]],  # Stretched vertically
        ]
    )

    # Weights
    weights = jnp.array([0.3, 0.4, 0.3])  # Slightly uneven weights

    return MixState(means, covs, weights)


def make_sde():
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
    sde = SDE(beta=beta)
    return sde


def make_mixture():
    key = jax.random.PRNGKey(666)
    state = init_mixture(key, d=2)
    return state


def run_forward_evolution_animation(sde, init_mix_state, num_frames=100, interval=200):
    key = jax.random.PRNGKey(666)
    pdf = partial(rho_t, init_mix_state=init_mix_state, sde=sde)
    def score(x, t):
        return jax.grad(pdf)(x, t) / pdf(x, t)

    # sample mixture
    num_samples = 100
    samples = sampler_mixtr(key, init_mix_state, num_samples)

    # Create 2D grid
    space = jnp.linspace(-5, 5, 100)
    x, y = jnp.meshgrid(space, space)
    xy = jnp.stack([x, y], axis=-1)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(x, y, jnp.zeros_like(x))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=12)

    state = SDEState(position=samples, t=jnp.zeros((num_samples, 1)))

    def update(frame):
        t = frame / num_frames * sde.beta.T
        pdf_grid = jax.vmap(jax.vmap(pdf, in_axes=(0, None)), in_axes=(0, None))(xy, t)

        # get score at current time
        # plot scores vectors

        ax.clear()
        ax.set_title("Forward Process")
        contour = ax.contourf(x, y, pdf_grid, levels=20, zorder=-1)
        # Update sample positions based on the SDE
        key = jax.random.PRNGKey(frame)  # Use frame as seed for reproducibility
        samples = sde.path(key, state, jnp.array([t])).position.squeeze()

        score_samples = jax.vmap(score, in_axes=(0, None))(samples, t)
        scores = ax.quiver(
            samples[:, 0],
            samples[:, 1],
            score_samples[:, 0],
            score_samples[:, 1],
            color="red",
        )

        # Plot updated samples
        scatter = ax.scatter(samples[:, 0], samples[:, 1], zorder=1, marker="o", s=10, c="k")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.axis("off")

        # Update time text
        time_text = ax.text(0.02, 0.98, f"Time: {t:.2f}", transform=ax.transAxes, va="top", fontsize=12)

        return scores, scatter, contour, time_text

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
    anim.save("forward_process.gif", writer="pillow")
    plt.show()


def run_backward_evolution_animation(sde, init_mix_state, num_frames=100, interval=200):
    key = jax.random.PRNGKey(666)
    pdf = partial(rho_t, init_mix_state=init_mix_state, sde=sde)
    def score(x, t):
        return jax.grad(pdf)(x, t) / pdf(x, t)

    # Sample from standard normal distribution
    num_samples = 100
    T = sde.beta.T
    init_samples = jax.random.normal(key, (num_samples, 2))

    # Create 2D grid
    space = jnp.linspace(-5, 5, 100)
    x, y = jnp.meshgrid(space, space)
    xy = jnp.stack([x, y], axis=-1)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(x, y, jnp.zeros_like(x))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=12)

    state = SDEState(position=init_samples, t=T * jnp.ones((num_samples, 1)))

    # Prepare reverse SDE function
    num_steps = 200
    dts = jnp.array([T / num_steps] * num_steps)
    keys = jax.random.split(key, num_samples)
    revert_sde = jax.jit(jax.vmap(partial(sde.reverso, score=score, dts=dts)))
    state_0, state_Ts = revert_sde(keys, state)

    def update(frame):
        t = frame / num_frames * T
        pdf_grid = jax.vmap(jax.vmap(lambda x, t: pdf(x, T - t), in_axes=(0, None)), in_axes=(0, None))(xy, t)

        ax.clear()
        ax.set_title("Backward Process")
        contour = ax.contourf(x, y, pdf_grid, levels=20, zorder=-1)

        # Update sample positions based on the reverse SDE
        idx_frame = int(t / T * num_steps)
        samples = state_Ts.position[:, idx_frame]
        score_samples = jax.vmap(score, in_axes=(0, None))(samples, T - t)
        scores = ax.quiver(
            samples[:, 0],
            samples[:, 1],
            score_samples[:, 0],
            score_samples[:, 1],
            color="red",
        )

        # Plot updated samples
        scatter = ax.scatter(samples[:, 0], samples[:, 1], zorder=1, marker="o", s=10, c="k")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.axis("off")

        # Update time text
        time_text = ax.text(
            0.02,
            0.98,
            f"Time: {T - t:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=12,
        )

        return scores, scatter, contour, time_text

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
    # save animation
    anim.save("backward_process.gif", writer="pillow")
    plt.show()


if __name__ == "__main__":
    sde = make_sde()
    state = make_mixture()
    run_forward_evolution_animation(sde, state)
    run_backward_evolution_animation(sde, state)
