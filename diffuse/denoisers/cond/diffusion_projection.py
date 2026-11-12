# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
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
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState


def _diffusion_projection_debug_callback(
    step_idx: int,
    x_uncond: np.ndarray,
    position_next: np.ndarray,
    delta: np.ndarray,
    residual: np.ndarray,
    residual_projected: np.ndarray,
    measurements: np.ndarray,
    y_target: np.ndarray,
    t_current: float,
    alpha_t: float,
    sigma_t: float,
    output_dir: str = "debug_diffusion_projection",
) -> None:
    """Host-side callback for plotting diffusion projection intermediates."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def get_magnitude(arr):
        if arr.ndim and arr.shape[-1] == 2:
            return np.linalg.norm(arr, axis=-1)
        return np.abs(arr)

    step_idx = int(step_idx)
    x_uncond = np.asarray(x_uncond)
    position_next = np.asarray(position_next)
    delta = np.asarray(delta)
    residual = np.asarray(residual)
    residual_projected = np.asarray(residual_projected)
    measurements = np.asarray(measurements)
    y_target = np.asarray(y_target)
    t_current = float(t_current)
    alpha_t = float(alpha_t)
    sigma_t = float(sigma_t)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Diffusion Projection Debug - Step {step_idx} (t={t_current:.4f}, alpha={alpha_t:.4f}, sigma={sigma_t:.4f})",
        fontsize=16,
    )

    im0 = axes[0, 0].imshow(get_magnitude(x_uncond), cmap="gray")
    axes[0, 0].set_title("x_uncond")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(get_magnitude(position_next), cmap="gray")
    axes[0, 1].set_title("x_projected")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    diff_correction = get_magnitude(position_next) - get_magnitude(x_uncond)
    im2 = axes[0, 2].imshow(diff_correction, cmap="seismic")
    axes[0, 2].set_title("|x_projected| - |x_uncond|")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(get_magnitude(delta), cmap="seismic")
    axes[1, 0].set_title("delta (CG correction)")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(get_magnitude(residual), cmap="hot")
    axes[1, 1].set_title("Residual (y_target - A x_uncond)")
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].axis("off")
    stats_text = f"""Statistics:

||x_uncond||: {np.linalg.norm(x_uncond):.4f}
||x_projected||: {np.linalg.norm(position_next):.4f}
||delta||: {np.linalg.norm(delta):.4f}

||residual||: {np.linalg.norm(residual):.4f}
||residual_projected||: {np.linalg.norm(residual_projected):.4f}
||measurements||: {np.linalg.norm(measurements):.4f}
||y_target||: {np.linalg.norm(y_target):.4f}

alpha_t: {alpha_t:.6f}
sigma_t: {sigma_t:.6f}
"""
    axes[1, 2].text(0.05, 0.5, stats_text, fontsize=10, verticalalignment="center", family="monospace")

    plt.tight_layout()
    save_path = output_path / f"step_{step_idx:04d}.png"
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"[Debug] Diffusion projection plot saved to {save_path}")


@dataclass
class DiffusionProjectionDenoiser(CondDenoiser):
    """Diffusion-Projection denoiser.

    A two-stage conditional denoiser that alternates between unconditional diffusion
    and measurement-based projection:

    1. **Diffusion**: Run the unconditional integrator step
    2. **Projection**: Project the result onto the measurement manifold by solving
       the normal equations AᵀA z = Aᵀ y_target using conjugate gradients

    The target measurement y_target incorporates noise-level-dependent stochasticity.
    """

    cg_tol: float = 1e-5
    cg_maxiter: int = 50
    debug_plot: bool = False
    debug_output_dir: str = "debug_diffusion_projection"

    def __post_init__(self):
        # No adaptive resampling in this simplified variant
        self.resample = False

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Single diffusion-projection step.

        1. Run the unconditional integrator step.
        2. Project the updated particle onto the measurement manifold by solving
           AᵀA z = Aᵀ y_target using conjugate gradients.
        """
        # del rng_key  # Deterministic update for each particle

        integrator_state = state.integrator_state
        step_idx = integrator_state.step
        t_current = self.integrator.timer(step_idx)

        # 1) Unconditional step
        integrator_state_uncond = self.integrator(integrator_state, self.predictor)
        x_uncond = integrator_state_uncond.position

        alpha_t = self.model.signal_level(t_current)
        sigma_t = self.model.noise_level(t_current)

        def normal_operator(z):
            """Return AᵀA z."""
            return self.forward_model.adjoint(
                self.forward_model.apply(z, measurement_state),
                measurement_state,
            )

        y_target = alpha_t * jnp.asarray(
            measurement_state.y
        )  # + sigma_t * self.forward_model.apply(jax.random.normal(rng_key, x_uncond.shape), measurement_state)
        residual = y_target - self.forward_model.apply(x_uncond, measurement_state)
        rhs = self.forward_model.adjoint(
            residual,
            measurement_state,
        )

        delta, _ = cg(
            normal_operator,
            rhs,
            x0=jnp.zeros_like(x_uncond),
            tol=self.cg_tol,
            maxiter=self.cg_maxiter,
        )

        position_next = x_uncond + delta
        residual_projected = y_target - self.forward_model.apply(position_next, measurement_state)

        integrator_state_next = integrator_state_uncond._replace(position=position_next)

        if self.debug_plot:
            callback = partial(
                _diffusion_projection_debug_callback,
                output_dir=self.debug_output_dir,
            )
            jax.experimental.io_callback(
                callback,
                (),
                step_idx,
                x_uncond,
                position_next,
                delta,
                residual,
                residual_projected,
                measurement_state.y,
                y_target,
                t_current,
                alpha_t,
                sigma_t,
            )

        return state._replace(integrator_state=integrator_state_next)
