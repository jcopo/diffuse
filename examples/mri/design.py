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

from examples.mri.utils import get_first_item, get_latest_model

from diffuse.bayesian_design import ExperimentOptimizer, ExperimentRandom
from diffuse.denoisers.cond_denoiser import CondDenoiser
from diffuse.diffusion.sde import SDE, LinearSchedule
from diffuse.integrator.deterministic import DPMpp2sIntegrator
from diffuse.integrator.stochastic import EulerMaruyama
from diffuse.neural_network.unet import UNet
from diffuse.neural_network.unett import UNet as Unet
from examples.mri.forward_models import maskRadial, maskSpiral
from diffuse.utils.plotting import (
    log_samples,
    plot_comparison,
    plot_lines,
    plotter_random,
    sigle_plot,
)

from examples.mri.brats.create_dataset import get_dataloader as get_brats_dataloader
from examples.mri.fastMRI.create_dataset import get_dataloader as get_fastmri_dataloader
from examples.mri.wmh.create_dataset import get_dataloader as get_wmh_dataloader

from envyaml import EnvYAML

SIZE = 7


# get user from environment variable
USER = os.getenv("USER")
WORKDIR = os.getenv("WORK")

dataloader_zoo = {
    "WMH": lambda cfg: get_wmh_dataloader(cfg, train=True),
    "BRATS": lambda cfg: get_brats_dataloader(cfg, train=True),
    "fastMRI": lambda cfg: get_fastmri_dataloader(cfg, train=True),
}

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

    thetas = jnp.stack([jnp.abs(thetas[..., 0] + 1j * thetas[..., 1]), thetas[..., -1]], axis=-1)
    ground_truth = jnp.stack([jnp.abs(ground_truth[..., 0] + 1j * ground_truth[..., 1]), ground_truth[..., -1]], axis=-1)
    for i in [0, 1]:
        mask_history, joint_y = measurement_state.mask_history, measurement_state.y
        thetas_i = thetas[..., i]
        n = 20
        best_idx = jnp.argsort(weights)[-n:][::-1]
        worst_idx = jnp.argsort(weights)[:n]

        # Calculate global min and max for consistent scaling
        restored_theta = mask.restore_from_mask(
            mask_history, jnp.zeros_like(ground_truth), joint_y
        )
        all_images = [
            ground_truth[..., i],
            thetas_i[best_idx],
            thetas_i[worst_idx]
        ]
        vmin = 0
        vmax = max(img.max() for img in all_images)

        # Create figure
        # Calculate global min and max for consistent scaling
        restored_theta = mask.restore_from_mask(
            mask_history, jnp.zeros_like(ground_truth), joint_y
        )
        all_images = [
            ground_truth[..., i],
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
            y=0.67,
            x=0.6,
        )

        gs = fig.add_gridspec(6, n, hspace=0.0001)

        # Ground truth subplot
        # Ground truth subplot
        ax_large = fig.add_subplot(gs[:2, :2])
        ax_large.imshow(ground_truth[..., i], cmap="gray", vmin=vmin, vmax=vmax)
        ax_large.imshow(ground_truth[..., i], cmap="gray", vmin=vmin, vmax=vmax)
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

        # Measurement subplot
        # Measurement subplot
        ax_large = fig.add_subplot(gs[:2, 2:4])
        ax_large.imshow(
            jnp.log10(jnp.abs(joint_y[..., 0] + 1j * joint_y[..., 1]) + 1e-10),
            cmap="gray",
        )
        ax_large.axis("off")
        ax_large.set_title("Measure $y$", fontsize=12)

        # Fourier subplot
        # Fourier subplot
        ax_large = fig.add_subplot(gs[:2, 4:6])
        ax_large.imshow(restored_theta[..., 0], cmap="gray")
        ax_large.axis("off")
        ax_large.set_title("Fourier($y$)", fontsize=12)

        # Remaining sample subplots
        # Remaining sample subplots
        for idx in range(n - 6):
            ax1 = fig.add_subplot(gs[0, idx + 6])
            ax2 = fig.add_subplot(gs[1, idx + 6])

            ax1.imshow(thetas_i[best_idx[idx]], cmap="gray", vmin=vmin, vmax=vmax)
            ax2.imshow(thetas_i[worst_idx[idx]], cmap="gray", vmin=vmin, vmax=vmax)
            ax1.imshow(thetas_i[best_idx[idx]], cmap="gray", vmin=vmin, vmax=vmax)
            ax2.imshow(thetas_i[worst_idx[idx]], cmap="gray", vmin=vmin, vmax=vmax)

            ax1.axis("off")
            ax2.axis("off")

        if logging_path:
            plt.savefig(f"{logging_path}/{i}_samples_{n_meas}.png", bbox_inches="tight")
            plt.savefig(f"{logging_path}/{i}_samples_{n_meas}.png", bbox_inches="tight")
        plt.show()
        plt.close()



def initialize_experiment(key: PRNGKeyArray, config: dict):
    data_model = config['dataset']
    path_dataset = config['path_dataset']
    model_dir = config['model_dir']

    dataloader = dataloader_zoo[data_model](config)

    xs = get_first_item(dataloader)

    key, subkey = jax.random.split(key)
    ground_truth = jax.random.choice(key, xs)

    n_t = config['inference']['n_t']
    tf = config['sde']['tf']
    dt = tf / n_t

    beta = LinearSchedule(
        b_min=config['sde']['beta_min'],
        b_max=config['sde']['beta_max'],
        t0=config['sde']['t0'],
        T=tf
    )

    if config['score_model'] == "UNet":
        score_net = UNet(config["unet"]["dt_embedding"], config["unet"]["embedding_dim"], upsampling=config["unet"]["upsampling"])
    elif config['score_model'] == "UNett":
        score_net = Unet(dim=config['unet']['embedding_dim'])
    else:
        raise ValueError(f"Score model {config['score_model']} not found")

    nn_trained = jnp.load(os.path.join(model_dir, f"ann_{get_latest_model(config)}.npz"), allow_pickle=True)
    params = nn_trained["params"].item()

    def nn_score(x, t):
        return score_net.apply(params, x, t)

    sde = SDE(beta=beta, tf=tf)
    shape = ground_truth.shape

    if config['mask']['mask_type'] == 'spiral':
        mask = maskSpiral(img_shape=shape, task=config['task'], num_spiral=1, num_samples=100000, sigma=.2)
    else:  # radial
        mask = maskRadial(
            num_lines=config['mask']['num_lines'],
            img_shape=shape,
            task=config['task']
        )

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
    weights_infered = jnp.exp(weights_infered)
    grande_truite = jnp.stack([jnp.abs(grande_truite[..., 0] + 1j * grande_truite[..., 1]), grande_truite[..., -1]], axis=-1)
    theta_infered = jnp.stack([jnp.abs(theta_infered[..., 0] + 1j * theta_infered[..., 1]), theta_infered[..., -1]], axis=-1)
    psnr_array = jax.vmap(dm_pix.psnr, in_axes=(None, 0))(grande_truite, theta_infered)
    psnr_score = jnp.sum(psnr_array * weights_infered)
    ssim_array = jax.vmap(dm_pix.ssim, in_axes=(None, 0))(grande_truite, theta_infered)
    ssim_score = jnp.sum(ssim_array * weights_infered)
    return psnr_score, ssim_score


def plot_and_log_iteration(
    mask,
    ground_truth,
    optimal_state,
    measurement_state,
    n_meas,
    logger_metrics_fn=None,
    plotter_theta=None,
    plotter_contrastive=None,
):
    """Handle plotting and logging for each iteration of the experiment."""
    # Calculate metrics
    psnr_score, ssim_score = evaluate_metrics(
        ground_truth,
        optimal_state.denoiser_state.integrator_state.position,
        optimal_state.denoiser_state.weights,
    )
    jax.debug.print("PSNR: {}", psnr_score)
    jax.debug.print("SSIM: {}", ssim_score)
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
        plotter_contrastive = partial(plotter_contrastive, mask=mask)
        jax.experimental.io_callback(
            plotter_contrastive,
            None,
            measurement_state,
            ground_truth,
            optimal_state.cntrst_denoiser_state.integrator_state.position,
            optimal_state.cntrst_denoiser_state.weights,
            n_meas,
        )


def main(
    num_measurements: int,
    key: PRNGKeyArray,
    config: dict,
    plot: bool = False,
    plotter_theta=None,
    plotter_contrastive=None,
    logger_metrics=None,
    random: bool = False,
):
    # Initialize experiment forward model
    sde, mask, ground_truth, dt, n_t, nn_score = initialize_experiment(key, config)
    n_samples = config['inference']['n_samples']
    n_samples_cntrst = config['inference']['n_samples_cntrst']
    n_loop_opt = config['inference']['n_loop_opt']
    n_opt_steps = n_t * n_loop_opt + (n_loop_opt - 1)

    # Conditional Denoiser
    integrator = EulerMaruyama(sde)
    # integrator = DPMpp2sIntegrator(sde)#, stochastic_churn_rate=0.1, churn_min=0.05, churn_max=1.95, noise_inflation_factor=.3)
    resample = True
    denoiser = CondDenoiser(integrator, sde, nn_score, mask, resample)

    # init design and measurement
    xi = mask.init_design(key)
    measurement_state = mask.init_measurement(xi)

    # ExperimentOptimizer
    optimizer = optax.chain(
        optax.adam(learning_rate=config['inference']['lr']),
        optax.scale(-1)
    )
    experiment_optimizer = ExperimentOptimizer(
        denoiser, mask, optimizer, ground_truth.shape
    )
    if random:
        experiment_optimizer = ExperimentRandom(denoiser, mask, ground_truth.shape)

    exp_state = experiment_optimizer.init(key, n_samples, n_samples_cntrst, dt)

    def scan_step(carry, n_meas):
        exp_state, measurement_state, key = carry

        key, subkey = jax.random.split(key)
        jax.debug.print("design start: {}", exp_state.design)
        jax.experimental.io_callback(plot_measurement, None, measurement_state)

        optimal_state, _ = experiment_optimizer.get_design(
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
                n_meas,
                logger_metrics,
                plotter_theta,
                plotter_contrastive,
            )

        exp_state = experiment_optimizer.init(subkey, n_samples, n_samples_cntrst, dt)
        return (exp_state, measurement_state, key), optimal_state.denoiser_state

    init_carry = (exp_state, measurement_state, key)
    (exp_state, measurement_state, key), optimal_states = jax.lax.scan(
        scan_step, init_carry, jnp.arange(num_measurements)
    )

    return ground_truth, optimal_states, measurement_state.y


if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--rng_key", type=int, default=0)
    parser.add_argument("--num_meas", type=int, default=3)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--space", type=str, default="runs")
    parser.add_argument("--config", type=str, default="examples/mri/configs/config_fastMRI_inference.yaml")

    args = parser.parse_args()

    # Load configuration
    config = EnvYAML(args.config)

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
    plotter_theta = partial(
        show_samples_plot, logging_path=logging_path_theta, size=SIZE
    )
    plotter_contrastive = partial(
        show_samples_plot, logging_path=logging_path_contrastive, size=SIZE
    )
    logger_metrics_fn = partial(logger_metrics, dir_path=dir_path)

    ground_truth, optimal_state, final_measurement = main(
        num_meas,
        rng_key,
        config,
        plot=plot,
        plotter_theta=plotter_theta,
        plotter_contrastive=plotter_contrastive,
        logger_metrics=logger_metrics_fn,
    )

    # Calculate final metrics
    final_samples = optimal_state.integrator_state.position[-1]
    weights = optimal_state.weights[-1]
    psnr_score, ssim_score = evaluate_metrics(
        ground_truth, final_samples, weights
    )
    print(f"PSNR: {psnr_score} SSIM: {ssim_score}")

    # Save final scores
    scores_file = f"{dir_path}/scores.csv"
    with open(scores_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "PSNR", "SSIM"])
        writer.writerow(["Optimized", float(psnr_score), float(ssim_score)])

    # random experiment
    dir_path_random = f"{dir_path}/random"
    os.makedirs(dir_path_random, exist_ok=True)
    plotter_random = partial(show_samples_plot, logging_path=dir_path_random, size=SIZE)
    logger_metrics_fn_random = partial(logger_metrics, dir_path=dir_path_random)
    ground_truth_random, optimal_state_random, final_measurement_random = main(
        num_meas,
        rng_key,
        plot=plot,
        plotter_theta=plotter_random,
        logger_metrics=logger_metrics_fn_random,
        random=True
    )
