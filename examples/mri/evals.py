import pdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import csv
from functools import partial
import dm_pix
from dataclasses import dataclass
from typing import Protocol
from jaxtyping import Array
from diffuse.base_forward_model import ForwardModel

def get_confusion_matrix_metrics(ground_truth: Array, x: Array) -> dict:
    """Calculate confusion matrix metrics between ground truth and predicted values.

    Args:
        ground_truth: Binary ground truth values (0 or 1)
        x: Binary predicted values (0 or 1)

    Returns:
        Dictionary containing TP, FP, TN, FN counts
    """
    # Ensure inputs are binary
    ground_truth = (ground_truth > 0.5).astype(jnp.float32)
    x = (x > 0.5).astype(jnp.float32)

    # Calculate each metric
    true_positives = jnp.sum(ground_truth * x)
    false_positives = jnp.sum((1 - ground_truth) * x)
    true_negatives = jnp.sum((1 - ground_truth) * (1 - x))
    false_negatives = jnp.sum(ground_truth * (1 - x))

    return {
        "TP": true_positives,
        "FP": false_positives,
        "TN": true_negatives,
        "FN": false_negatives
    }


def get_segmentation_metrics(ground_truth: Array, x: Array) -> dict:
    """Calculate DICE and IoU scores between ground truth and predicted segmentation.

    Args:
        ground_truth: Binary ground truth values (0 or 1)
        x: Binary predicted values (0 or 1)

    Returns:
        Dictionary containing DICE and IoU scores
    """
    metrics = get_confusion_matrix_metrics(ground_truth, x)
    tp, fp, fn = metrics["TP"], metrics["FP"], metrics["FN"]

    # Calculate DICE coefficient: 2*TP / (2*TP + FP + FN)
    dice = 2 * tp / jnp.maximum(2 * tp + fp + fn, 1e-8)

    # Calculate IoU (Jaccard index): TP / (TP + FP + FN)
    iou = tp / jnp.maximum(tp + fp + fn, 1e-8)

    return {
        "DICE": dice,
        "IoU": iou
    }

class Experiment(Protocol):
    def plot_measurement(self, measurement_state):
        pass

    def plot_samples(self, measurement_state, ground_truth, thetas, weights, n_meas, mask=None, logging_path=None):
        pass

    def evaluate_metrics(self, ground_truth, theta_infered, weights_infered):
        pass


@dataclass
class WMHExperiment(Experiment):
    mask: ForwardModel

    def plot_measurement(self, measurement_state):
        plot_measurement(measurement_state)

    def plot_samples(self, measurement_state, ground_truth, thetas, weights, n_meas, task="anomaly", logging_path=None):
        abs_thetas = jnp.abs(thetas[..., 0] + 1j * thetas[..., 1])
        abs_ground_truth = jnp.abs(ground_truth[..., 0] + 1j * ground_truth[..., 1])
        plot_channel(0, measurement_state.mask_history, measurement_state.y, abs_thetas, abs_ground_truth, weights, n_meas, self.mask, ground_truth, logging_path)
        if task == "anomaly":
            plot_channel(1, measurement_state.mask_history, measurement_state.y, thetas[..., 2], ground_truth[..., 2], weights, n_meas, self.mask, ground_truth, logging_path)

    def evaluate_metrics(self, ground_truth, theta_infered, weights_infered, task="anomaly"):
        jax.debug.print("weights_infered logger: {}", jax.scipy.special.logsumexp(weights_infered))
        weights_infered = jnp.exp(weights_infered)
        jax.debug.print("sum of weights_infered logger: {}", jnp.sum(weights_infered))
        # Convert to magnitude images and ensure correct shape
        abs_theta_infered = jnp.abs(theta_infered[..., 0] + 1j * theta_infered[..., 1])
        abs_ground_truth = jnp.abs(ground_truth[..., 0] + 1j * ground_truth[..., 1])

        # Add channel dimension
        abs_theta_infered = abs_theta_infered[..., None]
        abs_ground_truth = abs_ground_truth[..., None]

        max_val = jnp.maximum(jnp.max(abs_ground_truth), jnp.max(abs_theta_infered))
        psnr_array = jax.vmap(dm_pix.psnr, in_axes=(None, 0))(abs_ground_truth, abs_theta_infered)
        psnr_score = jnp.sum(psnr_array * weights_infered)

        ssim = partial(dm_pix.ssim, max_val=max_val, filter_size=7, filter_sigma=1.02)
        ssim_array = jax.vmap(ssim, in_axes=(None, 0))(abs_ground_truth, abs_theta_infered)
        ssim_score = jnp.sum(ssim_array * weights_infered)
        
        # Save the magnitude images - fixed callback usage
        # save_path = "/lustre/fswork/projects/rech/hlp/uha64uw/tmp_res/magnitude_images.npz"
        # def save_callback(gt, pred):
            # jnp.savez(save_path, ground_truth_=gt, prediction_=pred)
        
        # jax.experimental.io_callback(
            # save_callback,
            # None,
            # abs_ground_truth,
            # abs_theta_infered
        # )

        if task == "anomaly":
            segmentation_metrics = jax.vmap(get_segmentation_metrics, in_axes=(None, 0))(ground_truth[..., -1], theta_infered[..., -1])
            segmentation_metrics = jax.tree_map(lambda x: jnp.sum(x * weights_infered), segmentation_metrics)
            return psnr_score, ssim_score, segmentation_metrics
        else:
            return psnr_score, ssim_score, None




def plot_measurement(measurement_state):
    mask_history, y = measurement_state.mask_history, measurement_state.y
    print(f"% of space used: {jnp.sum(mask_history)/mask_history.size}")
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(mask_history, cmap="gray")
    ax[0].set_title("Mask")
    ax[0].axis("off")
    ax[1].imshow(jnp.log10(jnp.abs(y[..., 0] + 1j * y[..., 1]) + 1e-10), cmap="gray")
    ax[1].set_title("y")
    ax[1].axis("off")
    plt.show()


def plot_channel(
    i,
    mask_history,
    joint_y,
    thetas_i,
    ground_truth_i,
    weights,
    n_meas,
    mask,
    ground_truth,
    logging_path=None
):
    n = 20
    weights = jnp.exp(weights)
    best_idx = jnp.argsort(weights)[-n:][::-1]
    worst_idx = jnp.argsort(weights)[:n]

    # Calculate global min and max for consistent scaling
    restored_theta = mask.restore_from_mask(
        mask_history, jnp.zeros_like(ground_truth), joint_y
    )
    all_images = [
        ground_truth_i,
        thetas_i[best_idx],
        thetas_i[worst_idx]
    ]
    vmin = 0
    vmax = max(img.max() for img in all_images)

    # Create figure
    fig = plt.figure(figsize=(40, 10))
    fig.suptitle(
        "High weight (top) and low weight (bottom) Samples",
        fontsize=18,
        y=.92,
        x=0.62,
    )

    # Increase vertical spacing between rows
    gs = fig.add_gridspec(6, n, hspace=0.01)  # Changed from 0.0001 to 0.3

    # Ground truth subplot
    ax_large = fig.add_subplot(gs[:2, :2])
    ax_large.imshow(ground_truth_i, cmap="gray", vmin=vmin, vmax=vmax)
    ax_large.text(
        -0.05,  # Just outside the right edge of the axes
        0.5,  # Vertically centered
        f"Measurement {n_meas}",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        rotation="vertical",
        transform=ax_large.transAxes  # Use axes coordinates
    )
    ax_large.axis("off")
    ax_large.set_title("Ground Truth", fontsize=12)

    # Measurement subplot
    ax_large = fig.add_subplot(gs[:2, 2:4])
    ax_large.imshow(
        jnp.log10(jnp.abs(joint_y[..., 0] + 1j * joint_y[..., 1]) + 1e-10),
        cmap="gray",
    )
    ax_large.axis("off")
    ax_large.set_title(r"Measure $y$", fontsize=12)

    # Fourier subplot
    ax_large = fig.add_subplot(gs[:2, 4:6])
    ax_large.imshow(restored_theta[..., 0], cmap="gray")
    ax_large.axis("off")
    ax_large.set_title(r"$F^{-1}(y)$", fontsize=12)

    # Remaining sample subplots
    for idx in range(n - 6):
        ax1 = fig.add_subplot(gs[0, idx + 6])
        ax2 = fig.add_subplot(gs[1, idx + 6])

        ax1.imshow(thetas_i[best_idx[idx]], cmap="gray", vmin=vmin, vmax=vmax)
        ax2.imshow(thetas_i[worst_idx[idx]], cmap="gray", vmin=vmin, vmax=vmax)

        ax1.axis("off")
        ax2.axis("off")

    if logging_path:
        plt.savefig(f"{logging_path}/{i}_samples_{n_meas}.png", bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def show_samples_plot(
    measurement_state,
    ground_truth,
    thetas,
    weights,
    n_meas,
    mask=None,
    logging_path=None,
    size=7,
):
    weights = jnp.exp(weights)
    # Convert complex values to magnitude and phase
    thetas = jnp.stack([jnp.abs(thetas[..., 0] + 1j * thetas[..., 1]), thetas[..., -1]], axis=-1)
    ground_truth = jnp.stack([jnp.abs(ground_truth[..., 0] + 1j * ground_truth[..., 1]), ground_truth[..., -1]], axis=-1)

    # Plot both channels
    for i in [0, 1]:
        plot_channel(
            i,
            measurement_state.mask_history,
            measurement_state.y,
            thetas[..., i],
            ground_truth[..., i],
            weights,
            n_meas,
            mask,
            ground_truth,
            logging_path
        )

