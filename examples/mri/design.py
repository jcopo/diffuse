import csv
import os
import pdb
from functools import partial

import dm_pix
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jaxtyping import PRNGKeyArray
from torchio.utils import get_first_item


from diffuse.bayesian_design import ExperimentOptimizer
from diffuse.denoisers.cond_denoiser import CondDenoiser
from diffuse.diffusion.sde import SDE, LinearSchedule
from diffuse.integrator.deterministic import DPMpp2sIntegrator
from diffuse.integrator.stochastic import EulerMaruyama
from diffuse.neural_network.unet import UNet
from examples.mri.utils import maskSpiral, maskRadial
from diffuse.utils.plotting import (
    log_samples,
    plot_comparison,
    plot_lines,
    plotter_random,
    sigle_plot,
)
from examples.mnist.images import SquareMask
from examples.mri.brats.create_dataset import (
    get_train_dataloader as get_brats_train_dataloader,
)
from examples.mri.wmh.create_dataset import (
    get_train_dataloader as get_wmh_train_dataloader,
)

SIZE = 7


config_brats = {
    "path_dataset": "/lustre/fswork/projects/rech/hlp/uha64uw/diffuse/data/BRATS/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training",
    "save_path": "/lustre/fswork/projects/rech/hlp/uha64uw/diffuse/data/BRATS/models/",
    "batch_size": 32,
    "num_workers": 0,
}

USER = "uha64uw"

config_wmh = {
    "modality": "FLAIR",
    "slice_size_template": 49,
    "begin_slice": 26,
    "path_dataset": f"/lustre/fswork/projects/rech/hlp/{USER}/diffuse/data/WMH",
    "save_path": f"/lustre/fswork/projects/rech/hlp/{USER}/diffuse/data/WMH/models/",
    "batch_size": 32,
    "num_workers": 0,
}



def plot_measurement(measurement_state):
    mask_history, y = measurement_state.mask_history, measurement_state.y
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(mask_history, cmap="gray")
    ax[0].set_title("Mask")
    ax[0].axis('off')
    ax[1].imshow(jnp.log10(jnp.abs(y[..., 0] + 1j * y[..., 1]) + 1e-10), cmap="gray")
    ax[1].set_title("y")
    ax[1].axis('off')
    plt.show()

def show_samples_plot(
    measurement_state, ground_truth, thetas, weights, n_meas, size=7, mask=None
):
    for i in [0, 1]:
        mask_history, joint_y = measurement_state.mask_history, measurement_state.y
        thetas_i = thetas[..., i]
        #joint_y = joint_y[..., 0]
        n = 20
        best_idx = jnp.argsort(weights)[-n:][::-1]
        worst_idx = jnp.argsort(weights)[:n]

        # Create a figure with subplots
        fig = plt.figure(figsize=(40, 10))
        fig.suptitle(
            "High weight (top) and low weight (bottom) Samples", fontsize=18, y=0.67, x=0.6
        )

        # Create grid spec for layout with reduced vertical spacing
        gs = fig.add_gridspec(6, n, hspace=0.0001)

        # Add the larger subplot for the first 4 squares
        ax_large = fig.add_subplot(gs[:2, :2])
        ax_large.imshow(ground_truth[..., i], cmap="gray")
        ax_large.text(
            -2.3,
            9.0,
            f"Measurement {n_meas}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            rotation="vertical",
        )

        ax_large.axis("off")
        ax_large.set_title("Ground Truth", fontsize=12)

        # Add another large subplot
        ax_large = fig.add_subplot(gs[:2, 2:4])
        ax_large.imshow(jnp.log10(jnp.abs(joint_y[..., 0] + 1j * joint_y[..., 1]) + 1e-10), cmap="gray")
        ax_large.axis("off")
        ax_large.set_title("Measure $y$", fontsize=12)

        # Add another large subplot
        ax_large = fig.add_subplot(gs[:2, 4:6])
        restored_theta = mask.restore_from_mask(mask_history, jnp.zeros_like(ground_truth), joint_y)
        #ax_large.imshow(jnp.abs(restored_theta[..., 0] + 1j * restored_theta[..., 1]) , cmap="gray")
        ax_large.imshow(restored_theta[..., 0], cmap="gray")
        ax_large.axis("off")
        ax_large.set_title("Fourier($y$)", fontsize=12)
        # ax_large.scatter(opt_hist[-1, 0], opt_hist[-1, 1], marker="o", c="red")
        # add a square above the image. Around the design and 5 pixels from it


        # Add the remaining subplots
        for idx in range(n - 6):
            ax1 = fig.add_subplot(gs[0, idx + 6])
            ax2 = fig.add_subplot(gs[1, idx + 6])

            ax1.imshow(thetas_i[best_idx[idx]], cmap="gray")
            ax2.imshow(thetas_i[worst_idx[idx]], cmap="gray")

            ax1.axis("off")
            ax2.axis("off")

        #plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        plt.close()


def initialize_experiment(key: PRNGKeyArray, n_t: int):
    data_model = "wmh" # "wmh"

    if data_model == "brats":
        unet = "ann_480.npz"
        config = config_brats
        dataloader = get_brats_train_dataloader
    elif data_model == "wmh":
        unet = "ann_3955.npz"
        config = config_wmh
        dataloader = get_wmh_train_dataloader
    else:
        raise ValueError(f"Invalid data model: {data_model}")

    xs = get_first_item(dataloader(config))

    key, subkey = jax.random.split(key)
    ground_truth = xs[1]
    #ground_truth = jax.random.choice(key, xs)

    n_samples, tf = 150, 2.0
    dt = tf / n_t

    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)

    dt_embedding = tf / 32
    score_net = UNet(dt_embedding, 64, upsampling="pixel_shuffle")
    nn_trained = jnp.load( os.path.join(config["save_path"], unet), allow_pickle=True)
    params = nn_trained["params"].item()

    def nn_score(x, t):
        return score_net.apply(params, x, t)

    sde = SDE(beta=beta, tf=tf)
    shape = ground_truth.shape

    # mask = maskSpiral(img_shape=shape, num_spiral=1, num_samples=100000, sigma=.2)
    mask = maskRadial(img_shape=shape, num_lines=5)


    return sde, mask, ground_truth, dt, n_t, nn_score


def logger_metrics(psnr_score: float, ssim_score: float, n_meas: int, dir_path: str):
    """
    Log PSNR and SSIM metrics to a CSV file during optimization.

    Args:
        psnr_score: Peak Signal-to-Noise Ratio score
        ssim_score: Structural Similarity Index score
        n_meas: Current measurement number
    """
    # Assuming scores_file is defined in the global scope
    scores_file = f"{dir_path}/scores_during_opt.csv"

    # Create file with headers if it doesn't exist
    if not os.path.exists(scores_file):
        with open(scores_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Measurement", "PSNR", "SSIM"])

    # Append new scores
    with open(scores_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([n_meas, float(psnr_score), float(ssim_score)])


@jax.jit
def evaluate_metrics(grande_truite, theta_infered, weights_infered):
    psnr_array = jax.vmap(dm_pix.psnr, in_axes=(None, 0))(grande_truite, theta_infered)
    psnr_score = jnp.sum(psnr_array * weights_infered)
    ssim_array = jax.vmap(dm_pix.ssim, in_axes=(None, 0))(grande_truite, theta_infered)
    ssim_score = jnp.sum(ssim_array * weights_infered)
    return psnr_score, ssim_score


def plot_and_log_iteration(mask, ground_truth, optimal_state, measurement_state, hist, n_meas,
                          logger_metrics_fn=None, plotter_theta=None, plotter_contrastive=None):
    """Handle plotting and logging for each iteration of the experiment."""
    # Calculate metrics
    psnr_score, ssim_score = evaluate_metrics(
        ground_truth,
        optimal_state.denoiser_state.integrator_state.position,
        optimal_state.denoiser_state.weights
    )
    # Log metrics
    if logger_metrics_fn:
        jax.experimental.io_callback(
            logger_metrics_fn, None, psnr_score, ssim_score, n_meas
        )

    # Plot theta samples
    if plotter_theta:
        plotter_theta = partial(plotter_theta, mask=mask)
        jax.experimental.io_callback(
            plotter_theta,
            None,
            measurement_state,
            ground_truth,
            optimal_state.denoiser_state.integrator_state.position,
            optimal_state.denoiser_state.weights,
            n_meas,
        )

    # Plot contrastive samples
    if plotter_contrastive:
        jax.experimental.io_callback(
            plotter_contrastive,
            None,
            hist,
            ground_truth,
            measurement_state.y,
            optimal_state.cntrst_denoiser_state.integrator_state.position,
            optimal_state.cntrst_denoiser_state.weights,
            n_meas,
        )

def main(num_measurements: int, key: PRNGKeyArray, plot: bool = False,
         plotter_theta=None, plotter_contrastive=None, logger_metrics=None):
    # Initialize experiment forward model
    n_t = 150
    sde, mask, ground_truth, dt, n_t, nn_score = initialize_experiment(key, n_t)
    n_samples = 50
    n_samples_cntrst = 51
    n_loop_opt = 1
    n_opt_steps = n_t * n_loop_opt + (n_loop_opt - 1)

    # Conditional Denoiser
    # integrator = EulerMaruyama(sde)
    integrator = DPMpp2sIntegrator(sde)#, stochastic_churn_rate=0.1, churn_min=0.05, churn_max=1.95, noise_inflation_factor=.3)
    resample = True
    denoiser = CondDenoiser(integrator, sde, nn_score, mask, resample)

    # init design
    # measurement_state = mask.init_measurement()
    # measurement_state = mask.init_measurement(jnp.array([0.0, 0.0])) # For Spiral
    measurement_state = mask.init_measurement(jnp.array([0., .1, .2, .3, .4, .5])) # For Radial

    # ExperimentOptimizer
    optimizer = optax.chain(optax.adam(learning_rate=0.1), optax.scale(-1))
    experiment_optimizer = ExperimentOptimizer(
        denoiser, mask, optimizer, ground_truth.shape
    )

    exp_state = experiment_optimizer.init(key, n_samples, n_samples_cntrst, dt)

    def scan_step(carry, n_meas):
        exp_state, measurement_state, key = carry

        key, subkey = jax.random.split(key)
        jax.debug.print("design start: {}", exp_state.design)
        jax.experimental.io_callback(plot_measurement, None, measurement_state)

        optimal_state, hist = experiment_optimizer.get_design(
            exp_state, subkey, measurement_state, n_steps=n_opt_steps
        )
        jax.debug.print("design optimal: {}", optimal_state.design)

        # make new measurement
        new_measurement = mask.measure(optimal_state.design, ground_truth)
        measurement_state = mask.update_measurement(
            measurement_state, new_measurement, optimal_state.design
        )

        if plot:
            plot_and_log_iteration(
                mask,
                ground_truth,
                optimal_state,
                measurement_state,
                hist,
                n_meas,
                logger_metrics,
                plotter_theta,
                plotter_contrastive
            )

        exp_state = experiment_optimizer.init(subkey, n_samples, n_samples_cntrst, dt)
        return (exp_state, measurement_state, key), optimal_state

    init_carry = (exp_state, measurement_state, key)
    (exp_state, measurement_state, key), optimal_states = jax.lax.scan(
        scan_step,
        init_carry,
        jnp.arange(num_measurements)
    )

    return ground_truth, optimal_states[-1], measurement_state.y


if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--rng_key", type=int, default=0)
    parser.add_argument("--num_meas", type=int, default=3)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--space", type=str, default="runs")

    args = parser.parse_args()
    key_int = args.rng_key
    plot = args.plot
    num_meas = args.num_meas

    rng_key = jax.random.PRNGKey(key_int)
    dir_path = f"{args.space}/{args.prefix}/{key_int}_{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}"

    # Setup logging paths
    logging_path_theta = f"{dir_path}/theta"
    logging_path_contrastive = f"{dir_path}/contrastive"
    os.makedirs(logging_path_theta, exist_ok=True)
    os.makedirs(logging_path_contrastive, exist_ok=True)

    # Setup plotting functions
    plotter_theta = partial(log_samples, logging_path=logging_path_theta, size=SIZE)
    plotter_contrastive = partial(log_samples, logging_path=logging_path_contrastive, size=SIZE)
    logger_metrics_fn = partial(logger_metrics, dir_path=dir_path)

    ground_truth, optimal_state, final_measurement = main(
        num_meas,
        rng_key,
        plot=plot,
        plotter_theta=plotter_theta,
        plotter_contrastive=plotter_contrastive,
        logger_metrics=logger_metrics_fn
    )

    # Calculate final metrics
    final_samples = optimal_state.denoiser_state.integrator_state.position[-1, :optimal_state.weights.shape[0]]
    psnr_score, ssim_score = evaluate_metrics(ground_truth, final_samples, optimal_state.weights)
    print(f"PSNR: {psnr_score} SSIM: {ssim_score}")

    # Save final scores
    scores_file = f"{dir_path}/scores.csv"
    with open(scores_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "PSNR", "SSIM"])
        writer.writerow(["Optimized", float(psnr_score), float(ssim_score)])
