import jax.numpy as jnp
from diffuse.design.bayesian_design import BEDState, MeasurementState
from diffuse.denoisers.cond_denoiser import CondDenoiser
from diffuse.base_forward_model import ForwardModel
from jaxtyping import Array, PRNGKeyArray
from dataclasses import dataclass
import jax

@dataclass
class ADSOptimizer:
    """Implements fastMRI-style mask optimization strategies in JAX"""
    denoiser: CondDenoiser
    mask: ForwardModel
    strategy: str = "column_entropy"
    sigma: float = 1.0
    n_center: int = 10

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
        kspace_particles = jnp.fft.fft2(particles, axes=(-3, -2))

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

    def step(
        self,
        state: BEDState,
        rng_key: PRNGKeyArray,
        measurement_state: MeasurementState,
    ) -> BEDState:
        # Get posterior samples from denoiser state
        particles = state.denoiser_state.integrator_state.position

        # Update mask design
        new_design = self._update_design(state, particles)

        # Update measurement state with new mask
        new_measurement_state = measurement_state._replace(
            mask_history=jnp.concatenate([
                measurement_state.mask_history,
                new_design[None]
            ], axis=0)
        )

        # Run standard denoising step with updated mask
        return super().step(
            state._replace(design=new_design),
            rng_key,
            new_measurement_state
        )

    def get_design(self, *args, **kwargs):
        """Disable optimization steps, use pure mask selection"""
        return super().get_design(*args, **kwargs)