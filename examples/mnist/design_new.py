import jax
from jax import numpy as jnp
import dm_pix

import optax
import numpy as np
import os
import csv

from diffuse.neural_network.unet import UNet
from diffuse.diffusion.sde import SDE, LinearSchedule
from diffuse.integrator.stochastic import EulerMaruyama
from diffuse.denoisers.cond_denoiser import CondDenoiser
from diffuse.bayesian_design import ExperimentOptimizer
from examples.mnist.images import SquareMask
from diffuse.utils.plotting import plot_lines, sigle_plot
import matplotlib.pyplot as plt
from functools import partial
from diffuse.utils.plotting import log_samples, plot_comparison, plotter_random

from jaxtyping import PRNGKeyArray


SIZE = 7


def initialize_experiment(key: PRNGKeyArray):
    # Load MNIST dataset
    data = np.load("dataset/mnist.npz")
    xs = jnp.array(data["X"])
    xs = xs.reshape(xs.shape[0], xs.shape[1], xs.shape[2], 1)  # Add channel dimension

    # Initialize parameters
    tf = 2.0
    n_t = 300
    dt = tf / n_t

    # Define beta schedule and SDE
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)

    # Initialize ScoreNetwork
    score_net = UNet(dt, 64, upsampling="pixel_shuffle")
    nn_trained = jnp.load("weights/ann_2999.npz", allow_pickle=True)
    params = nn_trained["params"].item()

    # Define neural network score function
    def nn_score(x, t):
        return score_net.apply(params, x, t)

    # Set up mask and measurement
    ground_truth = jax.random.choice(key, xs)
    mask = SquareMask(SIZE, ground_truth.shape)

    # SDE
    sde = SDE(beta=beta, tf=tf)

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


def main(num_measurements: int, key: PRNGKeyArray, plot: bool = False,
         plotter_theta=None, plotter_contrastive=None, logger_metrics=None):
    # Initialize experiment forward model
    sde, mask, ground_truth, dt, n_t, nn_score = initialize_experiment(key)
    n_samples = 151
    n_samples_cntrst = 150

     # Conditional Denoiser
    integrator = EulerMaruyama(sde)
    resample = True
    denoiser = CondDenoiser(integrator, sde, nn_score, mask, resample)

    # init design
    measurement_state = mask.init_measurement()

    # ExperimentOptimizer
    optimizer = optax.chain(optax.adam(learning_rate=0.1), optax.scale(-1))
    experiment_optimizer = ExperimentOptimizer(
        denoiser, mask, optimizer, ground_truth.shape
    )

    exp_state = experiment_optimizer.init(key, n_samples, n_samples_cntrst, dt)

    for n_meas in range(num_measurements):
        optimal_state, hist = experiment_optimizer.get_design(
            exp_state, key, measurement_state, n_t
        )

        # make new measurement
        new_measurement = mask.measure(optimal_state.design, ground_truth)
        measurement_state = mask.update_measurement(
            measurement_state, new_measurement, optimal_state.design
        )

        sigle_plot(measurement_state.mask_history)
        sigle_plot(measurement_state.y)
        if plot:
            # Calculate metrics
            psnr_score, ssim_score = evaluate_metrics(
                ground_truth,
                optimal_state.denoiser_state.integrator_state.position,
                optimal_state.denoiser_state.weights
            )

            # Log metrics
            if logger_metrics:
                jax.experimental.io_callback(
                    logger_metrics, None, psnr_score, ssim_score, n_meas
                )

            # Plot theta samples
            if plotter_theta:
                jax.experimental.io_callback(
                    plotter_theta,
                    None,
                    hist,
                    ground_truth,
                    measurement_state.y,
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

        exp_state = experiment_optimizer.init(key, n_samples, n_samples_cntrst, dt)
        key, _ = jax.random.split(key)

    return ground_truth, optimal_state, measurement_state.y


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
