# pylint: disable=assignment-from-no-return
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import ForwardModel
from diffuse.denoisers.cond_denoiser import CondDenoiser
from diffuse.design.bayesian_design import BEDState, MeasurementState


@dataclass
class ADSOptimizer:
    """Implements fastMRI-style mask optimization strategies in JAX"""
    denoiser: CondDenoiser
    mask: ForwardModel
    base_shape: Tuple[int, ...]
    strategy: str = "column_entropy"
    sigma: float = 1.0
    n_center: int = 10

    def init(self, rng_key: PRNGKeyArray, n_samples: int, n_samples_cntrst: int, dt: float):
        design = self.mask.init_design(rng_key)
        denoiser_state = self.denoiser.init(design, rng_key, dt)
        return BEDState(denoiser_state=denoiser_state, cntrst_denoiser_state=None, design=design, opt_state=None)

    def _compute_selection_metric(self, particles: Array) -> Array:
        """Compute selection metric based on strategy"""
        if self.strategy == "column_variance":
            return jnp.var(particles, axis=0)
        elif self.strategy == "column_entropy":
            # Compute pairwise differences for entropy estimation
            diffs = particles[:, None] - particles[None, :]  # [N,N,...]
            log_probs = -jnp.square(diffs) / (2 * self.sigma**2)
            return jnp.mean(log_probs, axis=(0,1))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _update_design(self, state: BEDState, particles: Array) -> Array:
        """Update mask based on posterior particles"""
        # Convert to k-space
        kspace_particles = self.mask.measure(state.design, particles)

        # Compute metric and prevent reselection
        metric = self._compute_selection_metric(kspace_particles)
        taken = state.design
        metric *= (1 - taken)  # Zero out already selected columns

        # Select top-k columns
        n_cols = metric.shape[-2]
        if self.strategy == "fastmri_baseline":
            # Fixed center + random selection
            center_start = (n_cols - self.n_center) // 2
            selected = jnp.concatenate([
                jnp.arange(center_start, center_start + self.n_center),
                jax.random.choice(
                    jax.random.PRNGKey(0),  # Seed handled externally
                    n_cols,
                    shape=(self.mask.num_samples - self.n_center,),
                    replace=False,
                    p=(1 - taken[0,0,:,0]) / (n_cols - self.n_center)
                )
            ])
        else:
            # Adaptive column selection
            column_metrics = jnp.mean(metric, axis=(-1, -3, -4))  # Avg over batches, coils, readout
            selected = jnp.argsort(column_metrics)[-self.mask.num_samples:]

        # Update mask - handle JAX array updates
        new_mask = state.design.at[:, :, selected, :].set(1)
        return new_mask

    def get_design(
        self,
        state: BEDState,
        rng_key: PRNGKeyArray,
        measurement_state: MeasurementState,
        n_steps: int,
    ) -> BEDState:
        n_particles = jax.device_count() * 5
        # Get posterior samples from denoiser state
        particles = state.denoiser_state.integrator_state.position

        # Update mask design
        new_design = self._update_design(state, particles)

        cond_denoiser_state, _ = self.denoiser.generate(rng_key, measurement_state, n_steps, n_particles)

        return BEDState(denoiser_state=cond_denoiser_state, cntrst_denoiser_state=cond_denoiser_state, design=new_design, opt_state=None), _
