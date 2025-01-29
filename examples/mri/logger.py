import os
import datetime
import csv
from functools import partial
from examples.mri.evals import Experiment
import jax
import jax.numpy as jnp
from typing import Protocol
from jaxtyping import Array
from diffuse.base_forward_model import ForwardModel, MeasurementState
from diffuse.bayesian_design import BEDState

class ExperimentLogger(Protocol):
    def log(self, mask, ground_truth, optimal_state, measurement_state, iteration):
        pass

class MRILogger:
    """Handles logging and plotting for MRI experiments."""

    def __init__(self, config, rng_key, experiment:Experiment, prefix="", space="runs", random=False, save_plots=True, experiment_name: str = ""):
        """
        Initialize the logger with experiment configuration.

        Args:
            config: Configuration dictionary/object containing experiment settings
            rng_key: Random key identifier for the experiment
            prefix: Optional prefix for the experiment directory
            space: Base directory for experiment logs
            random: Whether this is a random experiment
        """
        self.config = config
        self.random = random
        self.experiment = experiment
        self.save_plots = save_plots

        if random:
            experiment_name += "/random"

        self.dir_path = os.path.join(space, experiment_name)
        self.theta_path = os.path.join(self.dir_path, "theta")
        self.contrastive_path = os.path.join(self.dir_path, "contrastive")
        self.metrics_file = os.path.join(self.dir_path, "metrics.csv")

        print("Saving to \n \n", self.dir_path)
        # Create directories
        os.makedirs(self.theta_path, exist_ok=True)
        os.makedirs(self.contrastive_path, exist_ok=True)

        # Initialize metrics CSV file with headers
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "PSNR", "SSIM", "K-space"])

    def metrics_logger(self, iteration, psnr_score, ssim_score, kspace_percent):
        """Log metrics to CSV file."""
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([iteration, float(psnr_score), float(ssim_score), float(kspace_percent)])

    def log(self, ground_truth: Array, optimal_state: BEDState, measurement_state: MeasurementState, iteration: int):
        """
        Log all metrics and generate plots for the current iteration.

        Args:
            ground_truth: The ground truth image
            optimal_state: The current optimal state
            measurement_state: The current measurement state
            iteration: Current iteration number
        """
        # Plot measurement state using io_callback
        #jax.experimental.io_callback( self.experiment.plot_measurement, None, measurement_state)

        # Get current samples and weights
        current_samples = optimal_state.denoiser_state.integrator_state.position
        current_weights = optimal_state.denoiser_state.weights
        jax.debug.print("current_weights: {}", jnp.exp(current_weights))

        # Calculate metrics for current iteration
        psnr_score, ssim_score = self.experiment.evaluate_metrics(
            ground_truth, current_samples, current_weights
        )

        # Calculate k-space usage percentage
        total_points = measurement_state.mask_history.size
        used_points = jnp.sum(measurement_state.mask_history)
        kspace_percent = (used_points / total_points) * 100

        # Replace multiple print statements with a single combined print
        jax.debug.print(
            "Iteration {} | PSNR: {} | SSIM: {} | K-space: {}%",
            iteration, psnr_score, ssim_score, kspace_percent
        )
        plot_samples = partial(self.experiment.plot_samples, logging_path=self.theta_path) if self.save_plots else self.experiment.plot_samples
        # Plot theta samples using io_callback
        jax.experimental.io_callback(
            plot_samples,
            None,
            measurement_state,
            ground_truth,
            current_samples,
            current_weights,
            iteration,
        )

        # Plot contrastive samples only for non-random experiments
        # if not self.random and getattr(optimal_state, 'cntrst_denoiser_state', None) is not None:
            # plot_contrastive = partial(self.experiment.plot_samples, logging_path=self.contrastive_path) if self.save_plots else self.experiment.plot_samples
            # jax.experimental.io_callback(
                # plot_contrastive,
                # None,
                # measurement_state,
                # ground_truth,
                # optimal_state.cntrst_denoiser_state.integrator_state.position,
                # optimal_state.cntrst_denoiser_state.weights,
                # iteration
            # )

        # Log metrics to CSV using io_callback
        jax.experimental.io_callback(
            self.metrics_logger,
            None,
            iteration,
            psnr_score,
            ssim_score,
            kspace_percent
        )