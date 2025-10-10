# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection

from diffuse.examples.gaussian_mixtures.mixture import pdf_mixtr


def display_histogram(samples, ax):
    """Display histogram of 1D samples."""
    flat_samples = samples.flatten()
    nb = flat_samples.shape[0]
    xmax = jnp.max(jnp.abs(samples))
    nbins = 80

    # Freedman-Diaconis rule
    percentiles = jnp.array([75, 25])
    q75, q25 = jnp.percentile(flat_samples, percentiles)
    iqr = q75 - q25
    2 * iqr * (nb ** (-1 / 3))

    h0, b = jnp.histogram(flat_samples, bins=nbins, range=[-xmax, xmax])
    h0 = h0 / nb * nbins / (2 * xmax)
    ax.bar(
        jnp.linspace(-xmax, xmax, nbins),
        h0,
        width=2 * xmax / (nbins - 1),
        align="center",
        color="red",
    )


def display_trajectories(Y, m, title=None):
    """
    Display 1D particle trajectories with color coding.
    m: number of trajectories to plot
    """
    N, P = Y.shape
    idxs = jnp.round(jnp.linspace(0, P - 1, m)).astype(jnp.int32)
    sorted_idx = jnp.argsort(Y[-1, :])
    selected_indices = sorted_idx[idxs]

    for i, idx in enumerate(selected_indices):
        color_marker = i / (m - 1)
        plt.plot(Y[:, idx], c=[color_marker, 0, 1 - color_marker], alpha=0.3, linewidth=0.5)
    if title:
        plt.title(title)


def display_trajectories_at_times(particles, timer, n_steps, space, perct, pdf, title=None):
    """Display histograms vs theoretical PDFs at different time points."""
    n_plots = len(perct)
    fig, axs = plt.subplots(n_plots, 1, figsize=(10 * n_plots, n_plots))
    if title:
        fig.suptitle(title)

    for i, x in enumerate(perct):
        k = int(x * n_steps)
        t = timer(0) - timer(k + 1)
        display_histogram(particles[k, :], axs[i])
        axs[i].plot(space, jax.vmap(pdf, in_axes=(0, None))(space, t))


def plot_2d_mixture_and_samples(mixture_state, final_samples, title, ax=None, save_plot=False):
    """
    Plot 2D samples overlaid on theoretical mixture contours.
    Similar to display_histogram but for 2D case.

    Args:
        mixture_state: Mixture state to plot contours
        final_samples: Generated samples from diffusion
        title: Plot title
        ax: Optional matplotlib axis to plot on. If None, creates new figure.
        save_plot: If True, save the plot to the plots directory
    """
    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

    # plot whole trajectory for 100 particles randomly selected with different colors
    # colors depends on the last position of the particle
    idxs = jax.random.choice(jax.random.PRNGKey(0), final_samples.shape[0], (50,), replace=True)
    # idxs = jnp.arange(final_samples.shape[0])
    steps_to_plot = final_samples.shape[0] // 10

    # Track if we've added labels for legend
    trajectory_labeled = False
    discretization_labeled = False

    for i in idxs:
        # Subsample the trajectory
        x = final_samples[::steps_to_plot, i, 0]
        y = final_samples[::steps_to_plot, i, 1]
        points = jnp.stack([x, y], axis=1)
        points = jnp.array(points)

        # Create segments for LineCollection
        segments = jnp.stack([points[:-1], points[1:]], axis=1)
        segments = jnp.array(segments)

        # Normalize color along the trajectory
        norm = plt.Normalize(0, len(segments))
        lc = LineCollection(segments, norm=norm, linewidth=2, alpha=0.5)
        lc.set_array(jnp.arange(len(segments)))
        if not trajectory_labeled:
            lc.set_label("Sample trajectories")
            trajectory_labeled = True
        ax.add_collection(lc)

        # Add discretization points
        scatter_label = "Time discretization" if not discretization_labeled else ""
        ax.scatter(x, y, color="black", s=4, zorder=10, alpha=0.5, marker="|", label=scatter_label)
        if not discretization_labeled:
            discretization_labeled = True

    # Determine plot range based on samples
    sample_range = jnp.max(jnp.abs(final_samples)) * 1.2
    x_range = jnp.linspace(-sample_range, sample_range, 100)
    y_range = jnp.linspace(-sample_range, sample_range, 100)
    X, Y = jnp.meshgrid(x_range, y_range)
    grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    # Compute theoretical PDF on grid
    pdf_values = jax.vmap(lambda x: pdf_mixtr(mixture_state, x))(grid_points)
    pdf_grid = pdf_values.reshape(X.shape)

    # Plot contours of theoretical distribution
    ax.contour(X, Y, pdf_grid, levels=10, colors="blue", alpha=0.6, linewidths=1.5)

    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Save plot if requested
    if save_plot and title:
        # Create plots directory if it doesn't exist
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)

        # Create safe filename from title
        safe_filename = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).rstrip()
        safe_filename = safe_filename.replace(" ", "_")
        filepath = os.path.join(plots_dir, f"{safe_filename}_2d_final.png")

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {filepath}")


def display_2d_trajectories_at_times(particles, timer, n_steps, perct, pdf, title=None, score=None, sde=None, save_plot=False):
    """
    Display 2D particle evolution at different time points in a single horizontal line.
    Shows samples as scatter plots overlaid on theoretical PDF contours and score field.

    Args:
        particles: Array of shape (n_particles, n_steps, 2)
        timer: Timer object to convert steps to time
        n_steps: Total number of steps
        perct: Percentiles of time points to show
        pdf: PDF function that takes (x, t)
        title: Optional title for the plot
        score: Optional score function that takes (x, t) and returns gradient
        sde: Optional SDE object to compute alpha values
        save_plot: If True, save the plot to the plots directory
    """
    n_plots = len(perct)

    # Single row layout
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if title:
        fig.suptitle(title, y=0.98, fontsize=12)

    # Ensure axes is always an array
    if n_plots == 1:
        axes = [axes]

    # Determine overall plot range from all time points
    all_samples = particles.reshape(-1, 2)
    sample_range = jnp.max(jnp.abs(all_samples)) * 1.1

    for i, x in enumerate(perct):
        ax = axes[i]

        k = int(x * n_steps)
        t = timer(0) - timer(k + 1)
        samples_at_t = particles[k, :]  # Shape: (n_particles, 2)

        # Create grid for contours
        x_range = jnp.linspace(-sample_range, sample_range, 50)
        y_range = jnp.linspace(-sample_range, sample_range, 50)
        X, Y = jnp.meshgrid(x_range, y_range)
        grid_points = jnp.stack([X.ravel(), Y.ravel()], axis=1)

        # Compute theoretical PDF on grid
        pdf_values = jax.vmap(lambda pos: pdf(pos, t))(grid_points)
        pdf_grid = pdf_values.reshape(X.shape)

        # Plot contours
        ax.contour(X, Y, pdf_grid, levels=8, colors="blue", alpha=0.6, linewidths=1)

        # Plot score field if provided
        if score is not None:
            # Create a coarser grid for score vectors
            skip = 4  # Show every 4th point to avoid clutter
            X_coarse = X[::skip, ::skip]
            Y_coarse = Y[::skip, ::skip]
            grid_coarse = jnp.stack([X_coarse.ravel(), Y_coarse.ravel()], axis=1)

            # Compute score vectors
            score_vectors = jax.vmap(lambda pos: score(pos, t))(grid_coarse)
            score_x = score_vectors[:, 0].reshape(X_coarse.shape)
            score_y = score_vectors[:, 1].reshape(X_coarse.shape)

            # Plot score field as arrows
            ax.quiver(
                X_coarse,
                Y_coarse,
                score_x,
                score_y,
                alpha=0.4,
                scale=50,
                width=0.003,
                color="green",
                label="Score field" if i == 0 else "",
            )

        # Plot samples
        ax.scatter(
            samples_at_t[:, 0],
            samples_at_t[:, 1],
            alpha=0.7,
            s=12,
            c="red",
            label="Samples" if i == 0 else "",
            zorder=-1,
        )

        ax.set_xlim(-sample_range, sample_range)
        ax.set_ylim(-sample_range, sample_range)
        ax.set_aspect("equal")
        ax.axis("off")  # Remove axes and grid

        # Compute alpha if SDE is provided
        alpha_t = sde.signal_level(t)
        ax.set_title(f"t = {t:.2f}, step {k},\n α = {alpha_t:.3f}".replace("0.", "."), fontsize=8)

        if i == 0:  # Add legend to first subplot
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plot if requested
    if save_plot and title:
        # Create plots directory if it doesn't exist
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)

        # Create safe filename from title
        safe_filename = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).rstrip()
        safe_filename = safe_filename.replace(" ", "_")
        filepath = os.path.join(plots_dir, f"{safe_filename}_2d_evolution.png")

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {filepath}")
