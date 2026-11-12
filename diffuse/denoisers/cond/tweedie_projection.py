from dataclasses import dataclass
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from jaxtyping import PRNGKeyArray
import matplotlib.pyplot as plt
import numpy as np

from diffuse.base_forward_model import MeasurementState
from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.cond.base import CondDenoiser, CondDenoiserState


def _plot_debug_callback(
    step_idx: int,
    x_t: np.ndarray,
    x0_hat: np.ndarray,
    x0_corrected: np.ndarray,
    delta: np.ndarray,
    position_next: np.ndarray,
    residual: np.ndarray,
    residual_corrected: np.ndarray,
    measurements: np.ndarray,
    t_current: float,
    t_next: float,
    output_dir: str = "debug_tweedie_plots",
) -> None:
    """Callback function to plot debugging information from Tweedie projection step.

    This function is called via io_callback and plots the key variables at each step.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Helper to compute magnitude for complex data
    def get_magnitude(arr):
        if arr.ndim and arr.shape[-1] == 2:  # Complex as [real, imag]
            return np.linalg.norm(arr, axis=-1)
        return np.abs(arr)

    step_idx = int(step_idx)
    x_t = np.asarray(x_t)
    x0_hat = np.asarray(x0_hat)
    x0_corrected = np.asarray(x0_corrected)
    delta = np.asarray(delta)
    position_next = np.asarray(position_next)
    residual = np.asarray(residual)
    residual_corrected = np.asarray(residual_corrected)
    measurements = np.asarray(measurements)
    t_current = float(t_current)
    t_next = float(t_next)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(
        f"Tweedie Projection Debug - Step {step_idx} (t={t_current:.4f} -> {t_next:.4f})",
        fontsize=16,
    )

    # Row 1: Current state, Tweedie estimate, Corrected estimate
    im0 = axes[0, 0].imshow(get_magnitude(x_t), cmap="gray")
    axes[0, 0].set_title(f"x_t (noisy state)")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(get_magnitude(x0_hat), cmap="gray")
    axes[0, 1].set_title(r"$\hat{x}_0$ (Tweedie estimate)")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(get_magnitude(x0_corrected), cmap="gray")
    axes[0, 2].set_title("x0_corrected (after CG)")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Row 2: Delta correction, Next position, Difference before/after correction
    im3 = axes[1, 0].imshow(get_magnitude(delta), cmap="seismic")
    axes[1, 0].set_title("delta (CG correction)")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(get_magnitude(position_next), cmap="gray")
    axes[1, 1].set_title(f"x_next (re-embedded)")
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    diff_correction = get_magnitude(x0_corrected) - get_magnitude(x0_hat)
    im5 = axes[1, 2].imshow(diff_correction, cmap="seismic")
    axes[1, 2].set_title("|x0_corrected| - |x0_hat|")
    axes[1, 2].axis("off")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    # Row 3: Residual, Measurements, Statistics
    im6 = axes[2, 0].imshow(get_magnitude(residual), cmap="hot")
    axes[2, 0].set_title("Residual (y - A x_hat)")
    axes[2, 0].axis("off")
    plt.colorbar(im6, ax=axes[2, 0], fraction=0.046)

    im7 = axes[2, 1].imshow(get_magnitude(measurements), cmap="magma")
    axes[2, 1].set_title(f"Measurements (y)")
    axes[2, 1].axis("off")
    plt.colorbar(im7, ax=axes[2, 1], fraction=0.046)

    # Statistics text
    axes[2, 2].axis("off")
    stats_text = f"""Statistics:

||x_t||: {np.linalg.norm(x_t):.4f}
||x_hat||: {np.linalg.norm(x0_hat):.4f}
||x0_corrected||: {np.linalg.norm(x0_corrected):.4f}
||delta||: {np.linalg.norm(delta):.4f}
||position_next||: {np.linalg.norm(position_next):.4f}

||residual||: {np.linalg.norm(residual):.4f}
||residual_corrected||: {np.linalg.norm(residual_corrected):.4f}
||measurements||: {np.linalg.norm(measurements):.4f}

t_current: {t_current:.6f}
t_next: {t_next:.6f}
    """
    axes[2, 2].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment="center", family="monospace")

    plt.tight_layout()
    save_path = output_path / f"step_{step_idx:04d}.png"
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"[Debug] Saved plot to {save_path}")


@dataclass
class TweedieProjectionDenoiser(CondDenoiser):
    """Conditional denoiser using Tweedie's formula with projection-based data consistency.

    The algorithm follows three stages at every reverse step:

    1. **Tweedie estimation** – apply Tweedie's formula to map the current noisy state ``x_t``
       to an estimate of the clean sample ``x̂₀``.
    2. **Projection** – enforce data-consistency by solving the normal equations
       ``(AᵀΣ_y⁻¹A + λI) δ = AᵀΣ_y⁻¹ (y - A x̂₀)`` with conjugate gradients.
    3. **Re-embedding** – inject the corrected clean estimate back to the next time level
       either by keeping the latent noise path (path-consistent) or by resampling it.
    """

    cg_tol: float = 1e-5
    cg_maxiter: int = 50
    lambda_reg: float = 0.0
    path_consistent: bool = True
    min_measurement_variance: float = 1e-6
    min_noise_level: float = 1e-5
    debug_plot: bool = False  # Enable debug plotting
    debug_output_dir: str = "debug_tweedie_plots"  # Output directory for debug plots

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        del rng_key  # RNG managed via integrator state to control re-embedding noise

        integrator_state = state.integrator_state
        x_t = integrator_state.position
        step_idx = integrator_state.step

        t_current = jnp.asarray(self.integrator.timer(step_idx), dtype=x_t.dtype)
        t_next = jnp.asarray(self.integrator.timer(step_idx + 1), dtype=x_t.dtype)

        # (A) Tweedie lift: estimate the clean sample at the current time level
        tweedie_state = self.model.tweedie(SDEState(x_t, t_current), self.predictor.score)
        x0_hat = tweedie_state.position

        # Measurement noise handling (Σ_y = σ_y² I)
        sigma_y = getattr(self.forward_model, "std", 1.0)
        sigma_y = jnp.asarray(sigma_y, dtype=x_t.dtype)
        sigma_y2 = jnp.maximum(sigma_y * sigma_y, self.min_measurement_variance)
        inv_sigma_y2 = jnp.reciprocal(sigma_y2)

        # (B) Data consistency: project x̂₀ with CG on the normal equations
        def normal_operator(z):
            measured = self.forward_model.apply(z, measurement_state)
            scaled = measured * inv_sigma_y2
            corrected = self.forward_model.adjoint(scaled, measurement_state)
            return corrected + self.lambda_reg * z

        residual = jnp.asarray(measurement_state.y, dtype=x_t.dtype) - self.forward_model.apply(x0_hat, measurement_state)
        scaled_residual = residual * inv_sigma_y2
        rhs = self.forward_model.adjoint(scaled_residual, measurement_state)

        delta, _ = cg(
            normal_operator,
            rhs,
            x0=jnp.zeros_like(x0_hat),
            tol=self.cg_tol,
            maxiter=self.cg_maxiter,
        )
        x0_corrected = x0_hat + delta
        residual_corrected = jnp.asarray(measurement_state.y, dtype=x_t.dtype) - self.forward_model.apply(
            x0_corrected, measurement_state
        )

        # (C) Re-embed to the next time level
        alpha_t = self.model.signal_level(t_current)
        sigma_t = jnp.maximum(self.model.noise_level(t_current), self.min_noise_level)
        alpha_next = self.model.signal_level(t_next)
        sigma_next = self.model.noise_level(t_next)

        rng_key_noise, rng_key_next = jax.random.split(integrator_state.rng_key)

        if self.path_consistent:
            epsilon_hat = (x_t - alpha_t * x0_hat) / sigma_t
            position_next = alpha_next * x0_corrected + sigma_next * epsilon_hat
        else:
            xi = jax.random.normal(rng_key_noise, x_t.shape, dtype=x_t.dtype)
            position_next = alpha_next * x0_corrected + sigma_next * xi

        integrator_state_next = integrator_state._replace(
            position=position_next,
            rng_key=rng_key_next,
            step=step_idx + 1,
        )

        # Debug plotting callback
        if self.debug_plot:
            callback = partial(_plot_debug_callback, output_dir=self.debug_output_dir)
            jax.experimental.io_callback(
                callback,
                (),  # No return value
                step_idx,
                x_t,
                x0_hat,
                x0_corrected,
                delta,
                position_next,
                residual,
                residual_corrected,
                measurement_state.y,
                t_current,
                t_next,
            )

        return state._replace(integrator_state=integrator_state_next)
