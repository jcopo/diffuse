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
from diffuse.utils.plotting import plot_lines

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


def main(num_measurements: int, key: PRNGKeyArray, plot: bool = False):
    # Initialize experiment forward model
    sde, mask, ground_truth, dt, n_t, nn_score = initialize_experiment(key)
    n_samples = 15
    n_samples_cntrst = 16

    # Conditional Denoiser
    integrator = EulerMaruyama(sde)
    denoiser = CondDenoiser(
        integrator, sde, nn_score, mask
    )  # , n_t, ground_truth.shape)

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

        exp_state = experiment_optimizer.init(key, n_samples, n_samples_cntrst, dt)
        if plot:
            # psnr_score, ssim_score = evaluate_metrics(
            #     ground_truth, optimal_state.denoiser_state.integrator_state.position[-1, :], optimal_state.weights
            # )
            # jax.experimental.io_callback(
            #     logger_metrics, None, psnr_score, ssim_score, n_meas
            # )
            print(hist.shape)
            jax.experimental.io_callback(
                plot_lines,
                None,
                hist[-1],
            )
            jax.experimental.io_callback(
                plot_lines,
                None,
                hist[-1],
            )


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    main(10, key, plot=True)
