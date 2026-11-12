#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd().parent / "triax"))

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich import print as rprint

import dm_pix
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def print_memory_stats(label=""):
    """Print current GPU memory usage with detailed breakdown."""
    for device in jax.devices():
        stats = device.memory_stats()
        if stats:
            bytes_in_use = stats.get("bytes_in_use", 0)
            peak_bytes = stats.get("peak_bytes_in_use", 0)
            bytes_limit = stats.get("bytes_limit", 0)
            num_allocs = stats.get("num_allocs", 0)
            largest_alloc = stats.get("largest_alloc_size", 0)

            console.print(f"[cyan]Memory {label}[/cyan] [{device.device_kind} {device.id}]:")
            console.print(f"  Current:      [yellow]{bytes_in_use / 1e9:.2f} GB[/yellow]")
            console.print(f"  Peak:         [yellow]{peak_bytes / 1e9:.2f} GB[/yellow]")
            if bytes_limit > 0:
                percent_used = (bytes_in_use / bytes_limit) * 100
                console.print(
                    f"  Limit:        [cyan]{bytes_limit / 1e9:.2f} GB[/cyan] ([yellow]{percent_used:.1f}%[/yellow] used)"
                )
            if num_allocs > 0:
                console.print(f"  Allocations:  [cyan]{num_allocs}[/cyan]")
            if largest_alloc > 0:
                console.print(f"  Largest:      [yellow]{largest_alloc / 1e9:.2f} GB[/yellow]")


def start_memory_profiler(output_dir="/tmp/jax_memory_profile"):
    """Start JAX memory profiling."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    # Emit a TensorBoard-compatible trace.
    jax.profiler.start_trace(output_dir, create_perfetto_trace=False)
    console.print(f"[green]Started JAX profiler[/green] → {output_dir}")
    console.print(f"  View with: [cyan]tensorboard --logdir={output_dir}[/cyan]")


def stop_memory_profiler(output_dir="/tmp/jax_memory_profile"):
    """Stop JAX profiling and dump device memory profile.

    Keeps running even if the device is already in an error state, and reports whether files were written.
    """
    event_files = []
    try:
        jax.profiler.stop_trace()
        event_files = list(Path(output_dir).glob("events.*"))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed stopping JAX trace[/red]: {exc}")

    mem_profile = Path(output_dir) / "device_memory.prof"
    try:
        jax.profiler.save_device_memory_profile(str(mem_profile))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed saving device memory profile[/red]: {exc}")
    else:
        console.print(f"[green]Device memory profile saved[/green] → [cyan]{mem_profile}[/cyan]")

    if event_files:
        console.print(f"[green]Trace events found[/green]: {[str(f) for f in event_files]}")
    else:
        console.print(f"[yellow]No trace events detected in[/yellow] [cyan]{output_dir}[/cyan]")

    console.print("[green]Stopped JAX profiler[/green]")


def checkpoint_predictor(pred: Predictor) -> Predictor:
    """Wrap predictor outputs with checkpointing to drop activations between steps."""
    pred._score_fn = jax.checkpoint(pred._score_fn)
    pred._noise_fn = jax.checkpoint(pred._noise_fn)
    pred._velocity_fn = jax.checkpoint(pred._velocity_fn)
    pred._x0_fn = jax.checkpoint(pred._x0_fn)
    return pred


from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import DiffusionProjectionDenoiser, TweedieProjectionDenoiser
from diffuse.denoisers.cond.dps import DPSDenoiser
from diffuse.denoisers.cond.fps import FPSDenoiser
from diffuse.denoisers.cond.tmp import TMPDenoiser
from diffuse.denoisers.cond.mala import AdaptiveMALADenoiser
from diffuse.denoisers.cond.stochastic_mala import StochasticMALADenoiser
from diffuse.denoisers.denoiser import Denoiser
from diffuse.examples.mri import (
    FullMask,
    RandomMask,
    RadialMask,
    SpiralMask,
    VerticalMask,
    VerticalDiscreteMask,
    load_brain_ar,
)
from diffuse.integrator.deterministic import (
    DDIMIntegrator,
    DPMpp2sIntegrator,
    EulerIntegrator,
    HeunIntegrator,
)
from diffuse.integrator.stochastic import EulerMaruyamaIntegrator
from diffuse.predictor import Predictor
from diffuse.timer.base import HeunTimer, VpTimer
from triax.utils.diagnostics import FlowMatchingDiagnostics

# Model checkpoint configuration
os.environ["SCRATCH"] = "/lustre/fsn1/projects/rech/hlp/uha64uw/"
os.environ["WORK"] = "/lustre/fswork/projects/rech/hlp/uha64uw/"

# -----------------------------------------------------------------------------
# Rich Logger Setup (following triax pattern)
# -----------------------------------------------------------------------------
console = Console()
logger = logging.getLogger("brain_ar_inference")
logger.setLevel(logging.DEBUG if "-v" in sys.argv or "--verbose" in sys.argv else logging.INFO)
logger.propagate = False

# Add Rich handler directly
logger.addHandler(
    RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=False,
        markup=True,
        show_path=False,
        show_level=False,
    )
)

# Silence noisy libraries
os.environ["ABSL_LOGGING_VERBOSITY"] = "3"
noisy_loggers = ["jax._src.xla_bridge", "orbax.checkpoint", "jax._src", "absl", "flax"]
for logger_name in noisy_loggers:
    noisy_logger = logging.getLogger(logger_name)
    noisy_logger.setLevel(logging.ERROR)
    noisy_logger.propagate = False

# -----------------------------------------------------------------------------
# Configuration defaults (mirrors the original notebook)
# -----------------------------------------------------------------------------
CHECKPOINT_NAME = "mri_flow"
CHECKPOINT_STEP = None
USE_EMA = True
PRECISION = "bfloat16"
PREDICTION_TYPE = "velocity"
VAE_CHECKPOINT_NAME = None

BRAIN_AR_PATH = Path("brain.ar")
BRAIN_RECORD_INDEX = 0

NUM_SAMPLES = 1
NUM_STEPS = 50
COND_NUM_STEPS = 50
SEED = 40
COND_SAMPLE_INDEX = 0

TIMER_EPS = 0.001
TIMER_TF = 1.0

DPS_EPSILON = 1e-3
DPS_ZETA = 5e-1

FPS_SIGMA_Y = 0.05
FPS_N_PARTICLES = 10

TMP_LAMBDA_REG = 0.1

# MALA (Adaptive MALA and Stochastic MALA) parameters
MALA_EPSILON = 1e-3  # Base step size for MALA proposal
MALA_BETA = 0.999  # Decay rate for squared-gradient accumulator
MALA_ALPHA = 0.9  # Decay rate for momentum accumulator
MALA_DELTA = 1e-8  # Numerical stability term for preconditioner
MALA_C1 = 0.55  # Exponent for diminishing adaptation in variance
MALA_C2 = 0.45  # Exponent for diminishing adaptation in momentum

# Projection-based denoiser CG parameters
# Used by DiffusionProjectionDenoiser and TweedieProjectionDenoiser
# for solving normal equations AᵀA z = Aᵀ y via conjugate gradients
PROJECTION_CG_TOL = 1e-5  # CG convergence tolerance
PROJECTION_CG_MAXITER = 80  # Maximum CG iterations

# Tweedie-specific parameters (in addition to CG params)
TWEEDIE_LAMBDA_REG = 0.5  # Regularization parameter for (AᵀA + λI)
TWEEDIE_PATH_CONSISTENT = True  # Path-consistent re-embedding (vs. resampling)

# Sample clipping/masking parameters
CLIP_SAMPLES = True  # Whether to clip generated samples
CLIP_THRESHOLD = 0.05  # Magnitude threshold (fraction of max magnitude)
CLIP_METHOD = "magnitude"  # "magnitude" or "percentile"

NUM_MASK_LINES = 4
SPIRAL_NUM_SAMPLES = 128
SPIRAL_SIGMA = 0.3
MASK_STD = 0.05
MASK_DATA_MODEL = "kneeFastMRI"

DEFAULT_MAX_RECON_SAMPLES = 4


# -----------------------------------------------------------------------------
# Experiment Configuration System (pytest-style)
# -----------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Configuration for a single reconstruction experiment."""

    config_id: str
    denoiser_name: str
    denoiser_class: type
    denoiser_params: dict[str, Any]
    integrator_name: str
    integrator_class: type
    integrator_params: dict[str, Any]
    mask_name: str
    mask_class: type
    mask_params: dict[str, Any]
    mask_design_fn: Any  # Function to create design for this mask


# Available denoisers with their default parameters
AVAILABLE_DENOISERS = [
    ("DPSDenoiser", DPSDenoiser, {"epsilon": DPS_EPSILON, "zeta": DPS_ZETA}),
    ("FPSDenoiser", FPSDenoiser, {}),
    ("TMPDenoiser", TMPDenoiser, {"lambda_reg": TMP_LAMBDA_REG}),
    (
        "DiffusionProjectionDenoiser",
        DiffusionProjectionDenoiser,
        {"cg_tol": PROJECTION_CG_TOL, "cg_maxiter": PROJECTION_CG_MAXITER},
    ),
    (
        "TweedieProjectionDenoiser",
        TweedieProjectionDenoiser,
        {
            "cg_tol": PROJECTION_CG_TOL,
            "cg_maxiter": PROJECTION_CG_MAXITER,
            "lambda_reg": TWEEDIE_LAMBDA_REG,
            "path_consistent": TWEEDIE_PATH_CONSISTENT,
        },
    ),
    (
        "AdaptiveMALADenoiser",
        AdaptiveMALADenoiser,
        {"epsilon": MALA_EPSILON, "beta": MALA_BETA, "alpha": MALA_ALPHA, "delta": MALA_DELTA, "c1": MALA_C1, "c2": MALA_C2},
    ),
    (
        "StochasticMALADenoiser",
        StochasticMALADenoiser,
        {"epsilon": MALA_EPSILON, "beta": MALA_BETA, "alpha": MALA_ALPHA, "delta": MALA_DELTA, "c1": MALA_C1, "c2": MALA_C2},
    ),
]

# Available integrators with their default parameters
AVAILABLE_INTEGRATORS = [
    ("DDIMIntegrator", DDIMIntegrator, {}),
    ("EulerIntegrator", EulerIntegrator, {}),
    ("HeunIntegrator", HeunIntegrator, {}),
    ("EulerMaruyamaIntegrator", EulerMaruyamaIntegrator, {}),
    ("DPMpp2sIntegrator", DPMpp2sIntegrator, {}),
]

# Available masks with their default parameters
AVAILABLE_MASKS = [
    ("RadialMask", RadialMask, {"num_lines": NUM_MASK_LINES, "std": MASK_STD, "data_model": MASK_DATA_MODEL}),
    ("VerticalMask", VerticalMask, {"num_lines": NUM_MASK_LINES, "std": MASK_STD, "data_model": MASK_DATA_MODEL}),
    ("VerticalDiscreteMask", VerticalDiscreteMask, {"num_lines": NUM_MASK_LINES, "std": MASK_STD, "data_model": MASK_DATA_MODEL}),
    (
        "SpiralMask",
        SpiralMask,
        {"num_samples": SPIRAL_NUM_SAMPLES, "sigma": SPIRAL_SIGMA, "std": MASK_STD, "data_model": MASK_DATA_MODEL},
    ),
    ("RandomMask", RandomMask, {"std": MASK_STD, "data_model": MASK_DATA_MODEL}),
    ("FullMask", FullMask, {"std": MASK_STD, "data_model": MASK_DATA_MODEL}),
]


def create_mask_design(mask_name: str, mask_params: dict, img_shape: tuple, seed: int = SEED):
    """Create the design parameter for a specific mask type."""
    if mask_name == "RadialMask":
        num_lines = mask_params.get("num_lines", NUM_MASK_LINES)
        angles = jnp.linspace(0, jnp.pi, num_lines, endpoint=False)
        radii = jnp.ones(num_lines) * 50.0
        return jnp.stack([angles, radii], axis=-1)
    elif mask_name == "VerticalMask":
        num_lines = mask_params.get("num_lines", NUM_MASK_LINES)
        return jnp.linspace(0.2, 0.8, num_lines)
    elif mask_name == "VerticalDiscreteMask":
        num_lines = mask_params.get("num_lines", NUM_MASK_LINES)
        width = img_shape[1]
        return jnp.array(
            [int(i * width / (num_lines + 1)) for i in range(1, num_lines + 1)],
            dtype=jnp.int32,
        )
    elif mask_name == "SpiralMask":
        return jnp.array([0.1, 0.1, 0.0], dtype=jnp.float32)
    elif mask_name == "RandomMask":
        # Random mask uses a sampled design
        from diffuse.examples.mri import RandomMask as RM

        temp_mask = RM(img_shape=img_shape, std=MASK_STD, data_model=MASK_DATA_MODEL)
        return temp_mask.sample_design(jax.random.PRNGKey(seed))
    elif mask_name == "FullMask":
        return jnp.array([])
    else:
        raise ValueError(f"Unknown mask type: {mask_name}")


def generate_experiment_configs() -> list[ExperimentConfig]:
    """Generate all possible experiment configurations (denoiser x integrator x mask)."""
    configs = []

    for denoiser_name, denoiser_class, denoiser_params in AVAILABLE_DENOISERS:
        for integrator_name, integrator_class, integrator_params in AVAILABLE_INTEGRATORS:
            for mask_name, mask_class, mask_params in AVAILABLE_MASKS:
                config_id = f"{denoiser_name}_{integrator_name}_{mask_name}"

                config = ExperimentConfig(
                    config_id=config_id,
                    denoiser_name=denoiser_name,
                    denoiser_class=denoiser_class,
                    denoiser_params=denoiser_params.copy(),
                    integrator_name=integrator_name,
                    integrator_class=integrator_class,
                    integrator_params=integrator_params.copy(),
                    mask_name=mask_name,
                    mask_class=mask_class,
                    mask_params=mask_params.copy(),
                    mask_design_fn=lambda mn=mask_name, mp=mask_params: (
                        lambda img_shape, seed=SEED: create_mask_design(mn, mp, img_shape, seed)
                    ),
                )
                configs.append(config)

    return configs


def matches_filter(config_id: str, filter_expr: str) -> bool:
    """
    Check if a config_id matches a pytest-style -k filter expression.

    Supports:
    - Substring matching: "DPS" matches any config containing "DPS"
    - AND: "DPS and Radial" matches configs containing both
    - OR: "DPS or FPS" matches configs containing either
    - NOT: "not EulerMaruyama" excludes configs containing "EulerMaruyama"
    - Parentheses: "(DPS or FPS) and Radial"
    """
    if not filter_expr:
        return True

    # Simple implementation: convert to boolean expression
    # Replace keywords with Python operators
    expr = filter_expr

    # Find all quoted strings and temporarily replace them
    quoted_strings = re.findall(r'"([^"]*)"', expr)
    for i, qs in enumerate(quoted_strings):
        expr = expr.replace(f'"{qs}"', f"__QUOTED_{i}__")

    # Find all word tokens (potential substrings to match)
    tokens = re.findall(r"\b(?!and\b|or\b|not\b)\w+\b", expr)

    # Replace each token with a membership test
    for token in tokens:
        # Check if it's a quoted string placeholder
        if token.startswith("__QUOTED_"):
            idx = int(token.replace("__QUOTED_", "").replace("__", ""))
            search_str = quoted_strings[idx]
        else:
            search_str = token

        # Replace with membership test
        expr = re.sub(r"\b" + re.escape(token) + r"\b", f'("{search_str}" in config_id)', expr)

    # Replace logical operators
    expr = re.sub(r"\band\b", " and ", expr)
    expr = re.sub(r"\bor\b", " or ", expr)
    expr = re.sub(r"\bnot\b", " not ", expr)

    # Evaluate the expression
    try:
        return eval(expr, {"config_id": config_id})
    except Exception as e:
        logger.warning("[yellow]⚠[/yellow] Failed to parse filter expression '[cyan]%s[/cyan]': %s", filter_expr, e)
        # Fallback to simple substring matching
        return filter_expr.lower() in config_id.lower()


def filter_configs(configs: list[ExperimentConfig], filter_expr: Optional[str]) -> list[ExperimentConfig]:
    """Filter experiment configurations based on pytest-style -k expression."""
    if not filter_expr:
        return configs

    filtered = [cfg for cfg in configs if matches_filter(cfg.config_id, filter_expr)]
    return filtered


def clip_samples(
    samples: jnp.ndarray,
    threshold: float = CLIP_THRESHOLD,
    method: str = CLIP_METHOD,
    reference: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Clip/mask generated samples based on magnitude threshold.

    Args:
        samples: Generated samples, shape [..., H, W, 2] (complex as [real, imag])
        threshold: Threshold value (interpretation depends on method)
        method: "magnitude" (fraction of max) or "percentile" (percentile cutoff)
        reference: Optional reference image for computing threshold

    Returns:
        Clipped samples with same shape as input
    """
    samples_np = np.asarray(samples)

    # Compute magnitude
    magnitude = np.linalg.norm(samples_np, axis=-1)  # [..., H, W]

    # Determine threshold value
    if method == "magnitude":
        # Threshold as fraction of maximum magnitude
        if reference is not None:
            ref_mag = np.linalg.norm(np.asarray(reference), axis=-1)
            max_mag = float(np.max(ref_mag))
        else:
            max_mag = float(np.max(magnitude))
        threshold_value = threshold * max_mag
    elif method == "percentile":
        # Threshold as percentile (e.g., 5th percentile)
        threshold_value = float(np.percentile(magnitude, threshold * 100))
    else:
        raise ValueError(f"Unknown clipping method: {method}")

    # Create mask: 1 where magnitude > threshold, 0 elsewhere
    mask = (magnitude > threshold_value).astype(samples_np.dtype)

    # Apply mask to both real and imaginary parts
    # Expand mask to match complex dimension: [..., H, W] -> [..., H, W, 1]
    mask_expanded = mask[..., np.newaxis]

    # Clip samples
    clipped = samples_np * mask_expanded

    return jnp.asarray(clipped)


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def finalize_figure(fig: plt.Figure, save_path: Optional[Path]) -> None:
    """Save and close a Matplotlib figure."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        logger.debug("[figure] saved → [cyan]%s[/cyan]", path)
    plt.close(fig)


def sanitize_filename(name: str) -> str:
    """Return a filesystem-friendly filename derived from the given name."""
    safe = "".join(c.lower() if c.isalnum() else "_" for c in name)
    safe = "_".join(filter(None, safe.split("_")))
    return safe or "figure"


def plot_all_brain_records(
    records: Sequence[np.ndarray],
    selected_idx: Optional[int] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Plot all brain MRI records with an optional highlight on the selected index."""
    if not records:
        logger.debug("No brain records available for plotting.")
        return

    num_cols = 4
    num_rows = (len(records) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, record in enumerate(records):
        magnitude = np.linalg.norm(np.asarray(record), axis=-1)
        axes[idx].imshow(magnitude, cmap="gray")
        title = f"Record {idx}"
        if idx == selected_idx:
            title += " [SELECTED]"
        axes[idx].set_title(title)
        axes[idx].axis("off")
        if idx == selected_idx:
            for spine in axes[idx].spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)
                spine.set_visible(True)

    for idx in range(len(records), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"All {len(records)} Brain MRI Records from brain.ar", fontsize=14)
    fig.tight_layout()
    finalize_figure(fig, save_path)


def plot_unconditional_samples(
    samples: jnp.ndarray,
    integrator,
    timer,
    num_steps: int,
    save_path: Optional[Path] = None,
) -> None:
    """Plot unconditionally generated samples and save the result."""
    samples_np = np.asarray(samples)
    num_samples = samples_np.shape[0]
    fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
    axes = np.atleast_1d(axes)

    for idx in range(num_samples):
        magnitude = np.linalg.norm(samples_np[idx], axis=-1)
        axes[idx].imshow(magnitude, cmap="gray")
        axes[idx].set_title(f"Sample {idx}")
        axes[idx].axis("off")

    fig.suptitle(
        f"Unconditional Generation: {type(integrator).__name__} + {type(timer).__name__} ({num_steps} steps)",
        fontsize=14,
    )
    fig.tight_layout()
    finalize_figure(fig, save_path)


def plot_unconditional_trajectory(
    history: Optional[jnp.ndarray],
    num_steps: int,
    integrator,
    save_path: Optional[Path] = None,
) -> None:
    """Plot the denoising trajectory for unconditional generation."""
    if history is None:
        print("Unconditional trajectory history is not available (keep_history=False).")
        return

    history_np = np.asarray(history)
    timestep_indices = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
    fig, axes = plt.subplots(1, len(timestep_indices), figsize=(3 * len(timestep_indices), 3))
    axes = np.atleast_1d(axes)

    for plot_idx, step_idx in enumerate(timestep_indices):
        step_idx = min(step_idx, history_np.shape[0] - 1)
        sample = history_np[step_idx, 0]
        magnitude = np.linalg.norm(sample, axis=-1)
        axes[plot_idx].imshow(magnitude, cmap="gray")
        axes[plot_idx].set_title(f"Step {step_idx}/{num_steps}")
        axes[plot_idx].axis("off")

    fig.suptitle(f"Denoising Trajectory (Sample 0) - {type(integrator).__name__}", fontsize=14)
    fig.tight_layout()
    finalize_figure(fig, save_path)


def plot_samples_as_lines(
    samples: jnp.ndarray,
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    """Plot horizontal and vertical magnitude profiles for the provided samples."""
    samples_np = np.asarray(samples)
    num_samples = samples_np.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    center_row = samples_np.shape[1] // 2
    for i in range(num_samples):
        magnitude = np.linalg.norm(samples_np[i], axis=-1)
        profile = magnitude[center_row, :]
        color = plt.cm.viridis(i / max(1, num_samples - 1))
        axes[0].plot(profile, color=color, alpha=0.7, label=f"Sample {i}")

    axes[0].set_title("Horizontal Profile (Center Row)")
    axes[0].set_xlabel("Column Index")
    axes[0].set_ylabel("Magnitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    center_col = samples_np.shape[2] // 2
    for i in range(num_samples):
        magnitude = np.linalg.norm(samples_np[i], axis=-1)
        profile = magnitude[:, center_col]
        color = plt.cm.viridis(i / max(1, num_samples - 1))
        axes[1].plot(profile, color=color, alpha=0.7, label=f"Sample {i}")

    axes[1].set_title("Vertical Profile (Center Column)")
    axes[1].set_xlabel("Row Index")
    axes[1].set_ylabel("Magnitude")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    finalize_figure(fig, save_path)


def plot_mask(mask: jnp.ndarray, forward_model, save_path: Optional[Path] = None) -> None:
    """Visualize the sampling mask associated with the forward model."""
    fig, ax = plt.subplots(figsize=(6, 6))
    mask_np = np.asarray(mask)
    im = ax.imshow(mask_np, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"{type(forward_model).__name__} (Sampling rate: {mask_np.mean():.1%})")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    finalize_figure(fig, save_path)


def plot_measurements(
    brain_img: jnp.ndarray,
    mask: jnp.ndarray,
    measurements: jnp.ndarray,
    zero_filled_recon: jnp.ndarray,
    forward_model,
    save_path: Optional[Path] = None,
) -> None:
    """Plot original image, sampling mask, measurements, and zero-filled reconstruction."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    orig_mag = np.linalg.norm(np.asarray(brain_img), axis=-1)
    axes[0].imshow(orig_mag, cmap="gray")
    axes[0].set_title("Original |x|")
    axes[0].axis("off")

    axes[1].imshow(np.asarray(mask), cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title(f"Mask ({np.asarray(mask).mean():.1%} sampling)")
    axes[1].axis("off")

    meas_mag = np.linalg.norm(np.asarray(measurements), axis=-1)
    axes[2].imshow(meas_mag, cmap="magma")
    axes[2].set_title("Measurements |y| (k-space)")
    axes[2].axis("off")

    zf_mag = np.linalg.norm(np.asarray(zero_filled_recon), axis=-1)
    axes[3].imshow(zf_mag, cmap="gray")
    axes[3].set_title("Zero-filled |A^T y|")
    axes[3].axis("off")

    fig.suptitle(f"Forward Model: {type(forward_model).__name__} Mask", fontsize=14)
    fig.tight_layout()
    finalize_figure(fig, save_path)


def plot_reconstruction(
    brain_img: jnp.ndarray,
    zero_filled_recon: jnp.ndarray,
    reconstructed: jnp.ndarray,
    mask: jnp.ndarray,
    forward_model,
    cond_denoiser,
    integrator,
    num_steps: int,
    error: float,
    save_path: Optional[Path] = None,
) -> None:
    """Plot reconstruction comparison among original, zero-filled, and diffusion outputs."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    orig_mag = np.linalg.norm(np.asarray(brain_img), axis=-1)
    axes[0, 0].imshow(orig_mag, cmap="gray")
    axes[0, 0].set_title("Original |x|")
    axes[0, 0].axis("off")

    zf_mag = np.linalg.norm(np.asarray(zero_filled_recon), axis=-1)
    axes[0, 1].imshow(zf_mag, cmap="gray")
    axes[0, 1].set_title("Zero-filled |A^T y|")
    axes[0, 1].axis("off")

    recon_mag = np.linalg.norm(np.asarray(reconstructed), axis=-1)
    axes[0, 2].imshow(recon_mag, cmap="gray")
    axes[0, 2].set_title(f"{type(cond_denoiser).__name__} Reconstruction\n(error: {error:.3f})")
    axes[0, 2].axis("off")

    mask_np = np.asarray(mask)
    axes[1, 0].imshow(mask_np, cmap="viridis", vmin=0, vmax=1)
    axes[1, 0].set_title(f"{type(forward_model).__name__}\n({mask_np.mean():.1%} sampling)")
    axes[1, 0].axis("off")

    zf_error = np.linalg.norm(np.asarray(zero_filled_recon - brain_img), axis=-1)
    im1 = axes[1, 1].imshow(zf_error, cmap="hot")
    axes[1, 1].set_title("Zero-filled Error |A^T y - x|")
    axes[1, 1].axis("off")

    recon_error_map = np.linalg.norm(np.asarray(reconstructed - brain_img), axis=-1)
    im2 = axes[1, 2].imshow(recon_error_map, cmap="hot")
    axes[1, 2].set_title(f"{type(cond_denoiser).__name__} Error |x̂ - x|")
    axes[1, 2].axis("off")

    fig.colorbar(im1, ax=axes[1, 1], fraction=0.046)
    fig.colorbar(im2, ax=axes[1, 2], fraction=0.046)

    fig.suptitle(
        (
            f"Reconstruction: {type(cond_denoiser).__name__} + {type(integrator).__name__}"
            f" + {type(forward_model).__name__} ({num_steps} steps)"
        ),
        fontsize=14,
    )
    fig.tight_layout()
    finalize_figure(fig, save_path)


def plot_reconstruction_trajectory(
    history_cond: Optional[jnp.ndarray],
    brain_img: jnp.ndarray,
    num_steps: int,
    cond_denoiser,
    integrator,
    save_path: Optional[Path] = None,
) -> None:
    """Plot the reconstruction trajectory showing magnitude and error evolution."""
    if history_cond is None:
        print("Conditional trajectory history is not available (keep_history=False).")
        return

    history_np = np.asarray(history_cond)
    brain_np = np.asarray(brain_img)
    timestep_indices = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
    fig, axes = plt.subplots(2, len(timestep_indices), figsize=(3 * len(timestep_indices), 6))

    last_im = None
    for plot_idx, step_idx in enumerate(timestep_indices):
        step_idx = min(step_idx, history_np.shape[0] - 1)
        sample = history_np[step_idx, 0]
        magnitude = np.linalg.norm(sample, axis=-1)
        axes[0, plot_idx].imshow(magnitude, cmap="gray")
        axes[0, plot_idx].set_title(f"Step {step_idx}/{num_steps}")
        axes[0, plot_idx].axis("off")

        error_map = np.linalg.norm(sample - brain_np, axis=-1)
        last_im = axes[1, plot_idx].imshow(error_map, cmap="hot")
        axes[1, plot_idx].set_title(f"Error (step {step_idx})")
        axes[1, plot_idx].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes[1, :], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{type(cond_denoiser).__name__} Reconstruction Trajectory - {type(integrator).__name__}",
        fontsize=14,
    )
    fig.tight_layout()
    finalize_figure(fig, save_path)


def plot_lines(
    array: jnp.ndarray,
    fractions: Optional[Sequence[float]] = None,
    title: str = "Trajectory Evolution",
    save_path: Optional[Path] = None,
) -> None:
    """Plot array slices at specific fractions across a trajectory."""
    data = np.asarray(array)
    if data.ndim < 3:
        raise ValueError("Expected array with at least 3 dimensions (time, H, W).")

    if fractions is None:
        fractions = [0.1 * i for i in range(10)]

    n = len(fractions)
    fig, axs = plt.subplots(1, n, figsize=(n * 3, 3))
    axs = np.atleast_1d(axs)
    fig.suptitle(title)

    for idx, fraction in enumerate(fractions):
        frame_index = int(fraction * (data.shape[0] - 1))
        frame_index = max(0, min(frame_index, data.shape[0] - 1))
        axs[idx].imshow(data[frame_index].squeeze(), cmap="gray")
        axs[idx].set_title(f"f={fraction:.1f}")
        axs[idx].axis("off")

    fig.tight_layout()
    finalize_figure(fig, save_path)


def plot_sample_slices(
    samples: jnp.ndarray,
    fractions: Optional[Sequence[float]] = None,
    title: str = "Generated Sample Slices",
    save_path: Optional[Path] = None,
) -> None:
    """Visualize generated sample magnitudes using plot_lines for quick inspection."""
    magnitude_grid = np.linalg.norm(np.asarray(samples), axis=-1)
    plot_lines(jnp.asarray(magnitude_grid), fractions=fractions, title=title, save_path=save_path)


def plot_conditional_method_comparison(
    method_results: dict[str, dict[str, object]],
    brain_img: jnp.ndarray,
    zero_filled_recon: jnp.ndarray,
    title: str,
    output_dir: Path,
    basename: str = "conditional_denoisers",
    sampling_rate: Optional[float] = None,
) -> None:
    """Summarize conditional denoisers with per-method magnitudes and error maps."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    brain_np = np.asarray(brain_img)
    zero_filled_np = np.asarray(zero_filled_recon)

    orig_mag = np.linalg.norm(brain_np, axis=-1)
    zero_filled_mag = np.linalg.norm(zero_filled_np, axis=-1)
    zero_filled_error = np.linalg.norm(zero_filled_np - brain_np, axis=-1)

    scale = float(orig_mag.max()) or 1.0
    brain_metric = jnp.asarray(orig_mag[..., None] / scale, dtype=jnp.float32)
    zero_metric = jnp.asarray(zero_filled_mag[..., None] / scale, dtype=jnp.float32)

    zero_psnr = float(dm_pix.psnr(brain_metric, zero_metric))
    zero_ssim = float(dm_pix.ssim(brain_metric, zero_metric))

    preferred_order = ["DiffusionProjection", "TP", "DP", "DPS"]
    ordered_names: list[str] = []
    for name in preferred_order:
        if name in method_results and name not in ordered_names:
            ordered_names.append(name)
    for name in method_results:
        if name not in ordered_names:
            ordered_names.append(name)

    label_map = {"DiffusionProjection": "DP", "TP": "TP", "DP": "DP", "DPS": "DPS"}

    method_display: list[dict[str, object]] = []
    max_error = float(np.max(zero_filled_error)) if zero_filled_error.size else 0.0
    for name in ordered_names:
        data = method_results.get(name)
        if not data:
            continue

        reconstruction = np.asarray(data.get("reconstruction"))
        if reconstruction.size == 0:
            continue

        recon_mag = np.linalg.norm(reconstruction, axis=-1)
        recon_error = np.linalg.norm(reconstruction - brain_np, axis=-1)
        if recon_error.size:
            max_error = max(max_error, float(np.max(recon_error)))

        psnr_samples = np.asarray(data.get("psnr_per_sample", np.array([])))
        ssim_samples = np.asarray(data.get("ssim_per_sample", np.array([])))
        sample_index = int(data.get("sample_index", 0))
        sample_psnr = float(psnr_samples[sample_index]) if psnr_samples.size else float("nan")
        sample_ssim = float(ssim_samples[sample_index]) if ssim_samples.size else float("nan")

        mean_psnr = data.get("mean_psnr")
        mean_ssim = data.get("mean_ssim")

        method_display.append(
            {
                "name": name,
                "label": label_map.get(name, name),
                "magnitude": recon_mag,
                "error": recon_error,
                "psnr": sample_psnr,
                "ssim": sample_ssim,
                "mean_psnr": float(mean_psnr) if mean_psnr is not None else float("nan"),
                "mean_ssim": float(mean_ssim) if mean_ssim is not None else float("nan"),
            }
        )

    if not method_display:
        print("No conditional denoisers to display.")
        return

    if max_error == 0:
        max_error = 1.0

    num_methods = len(method_display)
    fig, axes = plt.subplots(num_methods, 3, figsize=(12, 4 * num_methods))
    axes = np.atleast_2d(axes)

    column_titles = ["Ground Truth |x|", "Zero-filled |A^T y|", "Reconstruction |x̂|"]

    for row, entry in enumerate(method_display):
        axes[row, 0].imshow(orig_mag, cmap="gray")
        if row == 0:
            axes[row, 0].set_title(column_titles[0])
        axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(entry["label"], rotation=90, fontsize=12, labelpad=10)

        axes[row, 1].imshow(zero_filled_mag, cmap="gray")
        if row == 0:
            axes[row, 1].set_title(f"{column_titles[1]}\nPSNR {zero_psnr:.2f} dB\nSSIM {zero_ssim:.3f}")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(entry["magnitude"], cmap="gray")
        axes[row, 2].set_title(
            f"{column_titles[2]} ({entry['label']})\n"
            f"PSNR {entry['psnr']:.2f} dB (mean {entry['mean_psnr']:.2f} dB)\n"
            f"SSIM {entry['ssim']:.3f} (mean {entry['mean_ssim']:.3f})"
        )
        axes[row, 2].axis("off")

    # Add sampling rate to title if provided
    full_title = title
    if sampling_rate is not None:
        full_title = f"{title}\nk-space sampling: {sampling_rate * 100:.1f}%"

    fig.suptitle(full_title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    finalize_figure(fig, output_dir / f"{basename}_magnitude.png")

    fig_err, axes_err = plt.subplots(num_methods, 3, figsize=(12, 4 * num_methods))
    axes_err = np.atleast_2d(axes_err)
    column_titles_err = ["Ground Truth Error", "Zero-filled Error", "Reconstruction Error"]

    im = None
    for row, entry in enumerate(method_display):
        axes_err[row, 0].imshow(np.zeros_like(orig_mag), cmap="hot", vmin=0, vmax=max_error)
        if row == 0:
            axes_err[row, 0].set_title(column_titles_err[0])
        axes_err[row, 0].axis("off")
        axes_err[row, 0].set_ylabel(entry["label"], rotation=90, fontsize=12, labelpad=10)

        axes_err[row, 1].imshow(zero_filled_error, cmap="hot", vmin=0, vmax=max_error)
        if row == 0:
            axes_err[row, 1].set_title(column_titles_err[1])
        axes_err[row, 1].axis("off")

        im = axes_err[row, 2].imshow(entry["error"], cmap="hot", vmin=0, vmax=max_error)
        axes_err[row, 2].set_title(f"{column_titles_err[2]} ({entry['label']})")
        axes_err[row, 2].axis("off")

    fig_err.suptitle(f"{full_title} - Error Maps", fontsize=16)
    if im is not None:
        fig_err.colorbar(im, ax=axes_err[:, 2], fraction=0.046, pad=0.04)
    fig_err.tight_layout(rect=[0, 0, 1, 0.95])
    finalize_figure(fig_err, output_dir / f"{basename}_error.png")


def plot_recon(
    method_name: str,
    brain_img: jnp.ndarray,
    zero_filled_recon: jnp.ndarray,
    method_entry: dict[str, object],
    output_dir: Path,
    zero_psnr: float,
    zero_ssim: float,
    max_samples: int = DEFAULT_MAX_RECON_SAMPLES,
) -> None:
    """Plot several reconstructed samples for a single denoiser."""
    samples = method_entry.get("samples")
    if samples is None:
        print(f"[plot_recon] Skipping {method_name}: no samples available.")
        return

    samples_np = np.asarray(samples)
    if samples_np.ndim == 3:
        samples_np = samples_np[np.newaxis, ...]
    if samples_np.size == 0:
        print(f"[plot_recon] Skipping {method_name}: empty sample set.")
        return

    brain_np = np.asarray(brain_img)
    zero_np = np.asarray(zero_filled_recon)

    psnr_vals = np.asarray(method_entry.get("psnr_per_sample", np.array([])))
    ssim_vals = np.asarray(method_entry.get("ssim_per_sample", np.array([])))

    if psnr_vals.size:
        order = np.argsort(-np.nan_to_num(psnr_vals, nan=-np.inf))
    else:
        order = np.arange(samples_np.shape[0])
    selected = order[: max_samples or samples_np.shape[0]]

    if selected.size == 0:
        print(f"[plot_recon] Skipping {method_name}: unable to select samples.")
        return

    zero_mag = np.linalg.norm(zero_np, axis=-1)
    zero_error = np.linalg.norm(zero_np - brain_np, axis=-1)

    sample_errors = np.linalg.norm(samples_np - brain_np, axis=-1)
    max_error = float(
        np.max(
            np.concatenate(
                [
                    zero_error[np.newaxis, ...],
                    sample_errors[selected],
                ]
            )
        )
    )
    if max_error == 0:
        max_error = 1.0

    num_cols = 2 + selected.size
    fig, axes = plt.subplots(2, num_cols, figsize=(3 * num_cols, 6))
    axes = np.atleast_2d(axes)

    orig_mag = np.linalg.norm(brain_np, axis=-1)
    axes[0, 0].imshow(orig_mag, cmap="gray")
    axes[0, 0].set_title("Ground Truth |x|")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(np.zeros_like(orig_mag), cmap="hot", vmin=0, vmax=max_error)
    axes[1, 0].set_title("Ground Truth Error")
    axes[1, 0].axis("off")

    axes[0, 1].imshow(zero_mag, cmap="gray")
    axes[0, 1].set_title(f"Zero-filled |A^T y|\nPSNR {zero_psnr:.2f} dB\nSSIM {zero_ssim:.3f}")
    axes[0, 1].axis("off")

    axes[1, 1].imshow(zero_error, cmap="hot", vmin=0, vmax=max_error)
    axes[1, 1].set_title("Zero-filled Error")
    axes[1, 1].axis("off")

    for col_offset, sample_idx in enumerate(selected, start=2):
        sample = samples_np[int(sample_idx)]
        magnitude = np.linalg.norm(sample, axis=-1)
        error_map = sample_errors[int(sample_idx)]

        psnr_value = psnr_vals[int(sample_idx)] if psnr_vals.size else float("nan")
        ssim_value = ssim_vals[int(sample_idx)] if ssim_vals.size else float("nan")

        axes[0, col_offset].imshow(magnitude, cmap="gray")
        axes[0, col_offset].set_title(f"Sample {int(sample_idx)}\nPSNR {psnr_value:.2f} dB\nSSIM {ssim_value:.3f}")
        axes[0, col_offset].axis("off")

        im = axes[1, col_offset].imshow(error_map, cmap="hot", vmin=0, vmax=max_error)
        axes[1, col_offset].set_title(f"Error (sample {int(sample_idx)})")
        axes[1, col_offset].axis("off")

    fig.colorbar(im, ax=axes[1, 2:], fraction=0.046, pad=0.04)
    fig.suptitle(f"{method_name} Reconstructions", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{sanitize_filename(method_name)}_recon.png"
    finalize_figure(fig, filename)


def compute_sample_metrics(
    samples: jnp.ndarray,
    brain_metric: jnp.ndarray,
    metric_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSNR and SSIM metrics between samples and the reference brain image."""
    samples_jnp = jnp.asarray(samples)
    if samples_jnp.ndim == 3:
        samples_jnp = samples_jnp[jnp.newaxis, ...]

    if samples_jnp.shape[0] == 0:
        return np.array([]), np.array([])

    sample_mags = jnp.linalg.norm(samples_jnp, axis=-1)
    sample_metric = jnp.asarray(sample_mags[..., None] / metric_scale, dtype=jnp.float32)

    psnr_vals = jax.vmap(dm_pix.psnr, in_axes=(None, 0))(brain_metric, sample_metric)
    ssim_vals = jax.vmap(dm_pix.ssim, in_axes=(None, 0))(brain_metric, sample_metric)

    return np.array(psnr_vals), np.array(ssim_vals)


def prepare_output_dir(output_root: Path, run_name: Optional[str]) -> Path:
    """Create and return the output directory for the current run."""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / (run_name or f"brain_ar_run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_single_experiment(
    config: ExperimentConfig,
    brain_img: jnp.ndarray,
    integrator,
    diffuse_model,
    predictor: Predictor,
    num_steps: int,
    num_samples: int,
    seed: int,
    output_dir: Path,
    tweedie_debug_plots: bool,
    tweedie_debug_dir: str,
    diffusion_debug_plots: bool,
    diffusion_debug_dir: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run a single reconstruction experiment with the given configuration."""
    console.rule(f"[bold cyan]Running experiment: {config.config_id}[/bold cyan]")

    console.print("  [bold]Configuration:[/bold]")
    console.print(f"    Integrator: [cyan]{config.integrator_name}[/cyan]")
    console.print(f"    Denoiser: [cyan]{config.denoiser_name}[/cyan]")
    console.print(f"    Mask: [cyan]{config.mask_name}[/cyan]")

    exp_dir = output_dir / config.config_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    tweedie_debug_dir = Path(tweedie_debug_dir)
    diffusion_debug_dir = Path(diffusion_debug_dir)

    # Create mask design
    design = create_mask_design(config.mask_name, config.mask_params, brain_img.shape, seed)

    # Instantiate forward model (mask)
    forward_model = config.mask_class(img_shape=brain_img.shape, **config.mask_params)

    # Create mask
    mask = forward_model.make(design, None)
    sampling_rate = float(np.asarray(mask).mean())
    console.print(
        f"  [bold]Mask:[/bold] [cyan]{config.mask_name}[/cyan] (sampling rate: [green]{sampling_rate * 100:.2f}%[/green])"
    )

    # Show mask-specific parameters
    mask_param_str = ", ".join([f"{k}={v}" for k, v in config.mask_params.items()])
    if mask_param_str:
        console.print(f"    Mask params: [yellow]{mask_param_str}[/yellow]")

    # Save mask visualization
    plot_mask(mask, forward_model, save_path=exp_dir / "mask.png")

    # Create measurements
    measurement_state = MeasurementState(
        y=jnp.zeros_like(brain_img),
        mask_history=mask,
    )
    measurements = forward_model.apply(brain_img, measurement_state)
    measurement_state = MeasurementState(
        y=measurements,
        mask_history=mask,
    )
    zero_filled_recon = forward_model.adjoint(measurements, measurement_state)

    # Plot measurements
    plot_measurements(
        brain_img,
        mask,
        measurements,
        zero_filled_recon,
        forward_model,
        save_path=exp_dir / "measurements.png",
    )

    # Create denoiser
    denoiser_kwargs = {
        "integrator": integrator,
        "model": diffuse_model,
        "predictor": predictor,
        "forward_model": forward_model,
        "x0_shape": brain_img.shape,
        **config.denoiser_params,
    }
    diffusion_debug_output_dir: str | None = None
    tweedie_debug_output_dir: str | None = None

    if config.denoiser_class is DiffusionProjectionDenoiser and diffusion_debug_plots:
        debug_dir = diffusion_debug_dir if diffusion_debug_dir.is_absolute() else exp_dir / diffusion_debug_dir
        denoiser_kwargs["debug_plot"] = True
        diffusion_debug_output_dir = str(debug_dir)
        denoiser_kwargs["debug_output_dir"] = diffusion_debug_output_dir
        logger.info(
            "  Diffusion projection debug plotting enabled; saving figures to [cyan]%s[/cyan]",
            diffusion_debug_output_dir,
        )
    elif config.denoiser_class is TweedieProjectionDenoiser and tweedie_debug_plots:
        debug_dir = tweedie_debug_dir if tweedie_debug_dir.is_absolute() else exp_dir / tweedie_debug_dir
        denoiser_kwargs["debug_plot"] = True
        tweedie_debug_output_dir = str(debug_dir)
        denoiser_kwargs["debug_output_dir"] = tweedie_debug_output_dir
        logger.info(
            "  Tweedie debug plotting enabled; saving figures to [cyan]%s[/cyan]",
            tweedie_debug_output_dir,
        )
    denoiser = config.denoiser_class(**denoiser_kwargs)

    # Show denoiser parameters
    denoiser_params_display = dict(config.denoiser_params)
    if config.denoiser_class is DiffusionProjectionDenoiser and diffusion_debug_plots and diffusion_debug_output_dir:
        denoiser_params_display["debug_plot"] = True
        denoiser_params_display["debug_output_dir"] = diffusion_debug_output_dir
    elif config.denoiser_class is TweedieProjectionDenoiser and tweedie_debug_plots and tweedie_debug_output_dir:
        denoiser_params_display["debug_plot"] = True
        denoiser_params_display["debug_output_dir"] = tweedie_debug_output_dir
    if denoiser_params_display:
        denoiser_param_str = ", ".join([f"{k}={v}" for k, v in denoiser_params_display.items()])
        console.print(f"  [bold]Denoiser params:[/bold] [yellow]{denoiser_param_str}[/yellow]")

    # Run reconstruction
    rng_key = jax.random.PRNGKey(seed + 100)  # Changed seed for conditional denoising
    console.print(f"  Generating [bold]{num_samples}[/bold] samples with [bold]{num_steps}[/bold] steps...")

    print_memory_stats("before generation")

    # Start profiling for MALA denoisers
    profiling_enabled = args.profile and ("MALA" in config.denoiser_name)
    profile_dir: str | None = None
    if profiling_enabled:
        profile_dir = str(output_dir / "jax_profile")
        start_memory_profiler(profile_dir)

    try:
        final_state, history = denoiser.generate(
            rng_key=rng_key,
            measurement_state=measurement_state,
            n_steps=num_steps,
            n_particles=num_samples,
            keep_history=True,
        )
    finally:
        if profiling_enabled and profile_dir is not None:
            stop_memory_profiler(profile_dir)

    print_memory_stats("after generation")

    samples = final_state.integrator_state.position
    console.print(f"  [green]✓[/green] Reconstruction complete! Shape: [cyan]{samples.shape}[/cyan]")

    # Clip samples if enabled
    if CLIP_SAMPLES:
        samples_before_clip = samples
        samples = clip_samples(samples, threshold=CLIP_THRESHOLD, method=CLIP_METHOD, reference=brain_img)
        # Compute clipping statistics
        mask_ratio = float(jnp.mean(jnp.linalg.norm(samples, axis=-1) > 0))
        logger.info(
            "  [bold]Clipping:[/bold] method=[yellow]%s[/yellow], threshold=[yellow]%.3f[/yellow], kept=[green]%.1f%%[/green]",
            CLIP_METHOD,
            CLIP_THRESHOLD,
            mask_ratio * 100,
        )

    # Compute metrics
    brain_np = np.asarray(brain_img)
    brain_mag = np.linalg.norm(brain_np, axis=-1)
    metric_scale = float(brain_mag.max()) or 1.0
    brain_metric = jnp.asarray(brain_mag[..., None] / metric_scale, dtype=jnp.float32)

    psnr_per_sample, ssim_per_sample = compute_sample_metrics(samples, brain_metric, metric_scale)
    zero_psnr_vals, zero_ssim_vals = compute_sample_metrics(zero_filled_recon, brain_metric, metric_scale)

    mean_psnr = float(psnr_per_sample.mean()) if psnr_per_sample.size else float("nan")
    mean_ssim = float(ssim_per_sample.mean()) if ssim_per_sample.size else float("nan")
    zero_psnr = float(zero_psnr_vals[0]) if zero_psnr_vals.size else float("nan")
    zero_ssim = float(zero_ssim_vals[0]) if zero_ssim_vals.size else float("nan")

    console.print("  [bold]Metrics:[/bold]")
    console.print(f"    Mean PSNR: [green]{mean_psnr:.2f} dB[/green] (zero-filled: {zero_psnr:.2f} dB)")
    console.print(f"    Mean SSIM: [yellow]{mean_ssim:.3f}[/yellow] (zero-filled: {zero_ssim:.3f})")

    # Select best sample for visualization
    if psnr_per_sample.size:
        best_idx = int(np.argmax(psnr_per_sample))
    else:
        best_idx = 0
    reconstructed = samples[best_idx]

    brain_norm = float(jnp.linalg.norm(brain_img)) or 1.0
    rel_error = float(jnp.linalg.norm(reconstructed - brain_img) / brain_norm)

    # Save reconstruction plot
    plot_reconstruction(
        brain_img=brain_img,
        zero_filled_recon=zero_filled_recon,
        reconstructed=reconstructed,
        mask=mask,
        forward_model=forward_model,
        cond_denoiser=denoiser,
        integrator=integrator,
        num_steps=num_steps,
        error=rel_error,
        save_path=exp_dir / "reconstruction.png",
    )

    # Save trajectory if available
    if history is not None:
        plot_reconstruction_trajectory(
            history,
            brain_img,
            num_steps=num_steps,
            cond_denoiser=denoiser,
            integrator=integrator,
            save_path=exp_dir / "trajectory.png",
        )

        # Plot trajectory slices (magnitude evolution)
        sample_history = np.asarray(history)[:, 0]  # First sample
        magnitude_history = np.linalg.norm(sample_history, axis=-1)
        plot_lines(
            jnp.asarray(magnitude_history),
            title=f"Conditional Trajectory Evolution - {config.denoiser_name}",
            save_path=exp_dir / "conditional_evolution.png",
        )

        # Plot sample slices overview
        plot_sample_slices(
            history[:, 0],  # First sample's trajectory
            title=f"Conditional Trajectory Slices - {config.denoiser_name}",
            save_path=exp_dir / "conditional_slices.png",
        )

    # Plot final sample profiles
    plot_samples_as_lines(
        samples,
        title=f"Reconstructed Sample Profiles - {config.denoiser_name}",
        save_path=exp_dir / "sample_profiles.png",
    )

    # Save metrics to JSON
    metrics = {
        "config_id": config.config_id,
        "denoiser": config.denoiser_name,
        "integrator": config.integrator_name,
        "mask": config.mask_name,
        "sampling_rate": sampling_rate,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "zero_filled_psnr": zero_psnr,
        "zero_filled_ssim": zero_ssim,
        "relative_error": rel_error,
        "psnr_per_sample": psnr_per_sample.tolist() if psnr_per_sample.size else [],
        "ssim_per_sample": ssim_per_sample.tolist() if ssim_per_sample.size else [],
    }

    metrics_path = exp_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    console.print(f"  [green]✓[/green] Metrics saved to [cyan]{metrics_path}[/cyan]")

    return {
        "config": config,
        "samples": samples,
        "reconstruction": reconstructed,
        "psnr_per_sample": psnr_per_sample,
        "ssim_per_sample": ssim_per_sample,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "relative_error": rel_error,
        "sample_index": best_idx,
        "zero_filled_recon": zero_filled_recon,
        "zero_psnr": zero_psnr,
        "zero_ssim": zero_ssim,
        "mask": mask,
        "forward_model": forward_model,
        "sampling_rate": sampling_rate,
    }


def run_demo(
    output_dir: Path,
    filtered_configs: list[ExperimentConfig],
    max_recon_samples: int,
    num_samples: int,
    num_steps: int,
    seed: int,
    skip_unconditional: bool,
    tweedie_debug_plots: bool,
    tweedie_debug_dir: str,
    diffusion_debug_plots: bool,
    diffusion_debug_dir: str,
    args: argparse.Namespace,
) -> None:
    """Execute the full brain MRI inference pipeline for multiple configurations."""
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    tweedie_debug_dir = Path(tweedie_debug_dir)
    diffusion_debug_dir = Path(diffusion_debug_dir)

    if not filtered_configs:
        logger.error("[red]ERROR:[/red] No configurations match the filter criteria!")
        logger.info("Use [cyan]--list-configs[/cyan] to see all available configurations.")
        return

    console.print(f"Figures will be saved under [cyan]{figure_dir}[/cyan]")
    console.print(f"JAX devices: [cyan]{jax.devices()}[/cyan]")
    console.print(f"JAX version: [cyan]{jax.__version__}[/cyan]")
    console.print(f"\nWill run [bold]{len(filtered_configs)}[/bold] configuration(s):")
    for cfg in filtered_configs:
        console.print(f"  - [yellow]{cfg.config_id}[/yellow]")

    console.print(f"\nLoading model from checkpoint: [cyan]{CHECKPOINT_NAME}[/cyan]")
    diagnostics = FlowMatchingDiagnostics.from_checkpoint(
        checkpoint_name=CHECKPOINT_NAME,
        checkpoint_step=CHECKPOINT_STEP,
        use_ema=USE_EMA,
        precision=PRECISION,
        prediction_type=PREDICTION_TYPE,
        vae_checkpoint_name=VAE_CHECKPOINT_NAME,
    )

    model = diagnostics.model
    diffuse_model = diagnostics.diffuse_model
    predictor = diagnostics.predictor
    if args.checkpoint_predictor:
        predictor = checkpoint_predictor(predictor)
    config = diagnostics.config
    vae_model = diagnostics.vae_ema if VAE_CHECKPOINT_NAME else None

    console.print("[green]✓[/green] Model loaded successfully!")
    console.print(f"  Model type: [cyan]{type(model).__name__}[/cyan]")
    console.print(f"  Diffusion model: [cyan]{type(diffuse_model).__name__}[/cyan]")
    if vae_model is not None:
        console.print(f"  VAE model: [cyan]{type(vae_model).__name__}[/cyan]")

    print_memory_stats("after model load")

    # Create timer and integrator (reused across all experiments)
    timer = VpTimer(n_steps=num_steps, eps=TIMER_EPS, tf=TIMER_TF)
    # Use the integrator from first config (or default to DDIM)
    first_integrator_class = filtered_configs[0].integrator_class if filtered_configs else DDIMIntegrator
    integrator = first_integrator_class(model=diffuse_model, timer=timer, **filtered_configs[0].integrator_params)
    console.print(f"[green]✓[/green] Using [cyan]{type(integrator).__name__}[/cyan] with [cyan]{type(timer).__name__}[/cyan]")

    # Optional: Generate unconditional samples
    brain_img = None
    if not skip_unconditional:
        console.print(f"\nGenerating [bold]{num_samples}[/bold] unconditional samples...")
        denoiser = Denoiser(
            integrator=integrator,
            model=diffuse_model,
            predictor=predictor,
            x0_shape=(256, 256, 2),
        )

        rng_key = jax.random.PRNGKey(seed)
        final_state, history = denoiser.generate(
            rng_key=rng_key,
            n_steps=num_steps,
            n_particles=num_samples,
            keep_history=True,
        )
        generated_samples = final_state.integrator_state.position
        console.print(f"[green]✓[/green] Generation complete! Generated shape: [cyan]{generated_samples.shape}[/cyan]")

        plot_unconditional_samples(
            generated_samples,
            integrator=integrator,
            timer=timer,
            num_steps=num_steps,
            save_path=figure_dir / "unconditional_samples.png",
        )
        plot_unconditional_trajectory(
            history,
            num_steps=num_steps,
            integrator=integrator,
            save_path=figure_dir / "unconditional_trajectory.png",
        )

        if history is not None:
            sample_history = np.asarray(history)[:, 0]
            magnitude_history = np.linalg.norm(sample_history, axis=-1)
            plot_lines(
                jnp.asarray(magnitude_history),
                title="Unconditional Trajectory Slices (Sample 0)",
                save_path=figure_dir / "unconditional_trajectory_slices.png",
            )

        plot_sample_slices(
            generated_samples,
            title="Generated Samples Overview",
            save_path=figure_dir / "unconditional_slices.png",
        )
        plot_samples_as_lines(
            generated_samples,
            title="Generated Sample Profiles",
            save_path=figure_dir / "unconditional_profiles.png",
        )

        # Use first generated sample as target
        brain_img = jnp.asarray(generated_samples[0])
        logger.info("Using unconditional sample 0 as the target image.")
    else:
        logger.info("\nSkipping unconditional generation.")

    # Try to load brain.ar if available
    if brain_img is None:
        brain_records: list[np.ndarray] = []
        # Uncomment to load from brain.ar:
        # try:
        #     brain_records = load_brain_ar(BRAIN_AR_PATH)
        #     print(f"\nLoaded {len(brain_records)} brain.ar records from {BRAIN_AR_PATH}.")
        # except (FileNotFoundError, ImportError) as exc:
        #     print(f"\nCould not load {BRAIN_AR_PATH}: {exc}")

        if brain_records:
            plot_all_brain_records(
                brain_records,
                selected_idx=BRAIN_RECORD_INDEX,
                save_path=figure_dir / "brain_records.png",
            )
            brain_img = jnp.asarray(brain_records[BRAIN_RECORD_INDEX])
            logger.info("Using record [cyan]%d[/cyan] as the target image.", BRAIN_RECORD_INDEX)
        else:
            # Generate a dummy sample if needed
            logger.info("\nNo brain.ar data available. Generating a target sample...")
            denoiser = Denoiser(
                integrator=integrator,
                model=diffuse_model,
                predictor=predictor,
                x0_shape=(256, 256, 2),
            )
            rng_key = jax.random.PRNGKey(seed)
            final_state, _ = denoiser.generate(
                rng_key=rng_key,
                n_steps=num_steps,
                n_particles=1,
                keep_history=False,
            )
            brain_img = final_state.integrator_state.position[0]
            logger.info("Using generated sample as the target image.")

    # Run all filtered experiments
    logger.info("\nRunning conditional reconstruction experiments...")
    all_results = []

    for idx, exp_config in enumerate(filtered_configs):
        try:
            result = run_single_experiment(
                config=exp_config,
                brain_img=brain_img,
                integrator=integrator,
                diffuse_model=diffuse_model,
                predictor=predictor,
                num_steps=num_steps,
                num_samples=num_samples,
                seed=seed + idx * 100,
                output_dir=output_dir,
                tweedie_debug_plots=tweedie_debug_plots,
                tweedie_debug_dir=tweedie_debug_dir,
                diffusion_debug_plots=diffusion_debug_plots,
                diffusion_debug_dir=diffusion_debug_dir,
                args=args,
            )
            all_results.append(result)
        except Exception as e:
            logger.error("[red]❌ ERROR running[/red] [cyan]%s[/cyan]: %s", exp_config.config_id, e, exc_info=True)
            continue

    # Generate comparison plot across all experiments
    if len(all_results) > 1:
        logger.info("\nGenerating cross-experiment comparison...")
        comparison_dir = figure_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # Prepare method_results dict for comparison plot
        method_results = {}
        for result in all_results:
            config = result["config"]
            # Use denoiser name as method label
            method_label = f"{config.denoiser_name.replace('Denoiser', '')}"
            if len(all_results) > 10:
                # If many configs, use shorter names
                method_label = config.denoiser_name[:3].upper()

            method_results[method_label] = {
                "samples": result["samples"],
                "reconstruction": result["reconstruction"],
                "psnr_per_sample": result["psnr_per_sample"],
                "ssim_per_sample": result["ssim_per_sample"],
                "mean_psnr": result["mean_psnr"],
                "mean_ssim": result["mean_ssim"],
                "relative_error": result["relative_error"],
                "sample_index": result["sample_index"],
            }

        # Get zero-filled recon and sampling rate from first result
        zero_filled_recon = all_results[0]["zero_filled_recon"]
        sampling_rate = all_results[0].get("sampling_rate")

        plot_conditional_method_comparison(
            method_results=method_results,
            brain_img=brain_img,
            zero_filled_recon=zero_filled_recon,
            title=f"Comparison of {len(all_results)} Configurations",
            output_dir=comparison_dir,
            basename="all_configs",
            sampling_rate=sampling_rate,
        )

        # Plot detailed reconstructions for each method
        for result in all_results:
            config = result["config"]
            method_name = config.config_id

            method_entry = {
                "samples": result["samples"],
                "psnr_per_sample": result["psnr_per_sample"],
                "ssim_per_sample": result["ssim_per_sample"],
            }

            plot_recon(
                method_name=method_name,
                brain_img=brain_img,
                zero_filled_recon=result["zero_filled_recon"],
                method_entry=method_entry,
                output_dir=comparison_dir,
                zero_psnr=result["zero_psnr"],
                zero_ssim=result["zero_ssim"],
                max_samples=max_recon_samples,
            )

    # Print final summary
    summary_table = Table(title="[bold]EXPERIMENT SUMMARY[/bold]", show_header=True, header_style="bold magenta")
    summary_table.add_column("Config ID", style="cyan", width=60)
    summary_table.add_column("PSNR (dB)", justify="right", style="green")
    summary_table.add_column("SSIM", justify="right", style="yellow")
    summary_table.add_column("Rel Error", justify="right", style="red")

    for result in all_results:
        config = result["config"]
        summary_table.add_row(
            config.config_id, f"{result['mean_psnr']:.2f}", f"{result['mean_ssim']:.3f}", f"{result['relative_error']:.4f}"
        )

    console.print()
    console.print(summary_table)
    console.print(f"\n[green]✓[/green] All experiments complete! Results saved to [cyan]{output_dir}[/cyan]")

    # Optional: Save summary JSON
    summary_path = output_dir / "summary.json"
    summary_data = [
        {
            "config_id": r["config"].config_id,
            "denoiser": r["config"].denoiser_name,
            "integrator": r["config"].integrator_name,
            "mask": r["config"].mask_name,
            "mean_psnr": r["mean_psnr"],
            "mean_ssim": r["mean_ssim"],
            "relative_error": r["relative_error"],
        }
        for r in all_results
    ]
    summary_path.write_text(json.dumps(summary_data, indent=2))
    console.print(f"[green]✓[/green] Summary saved to [cyan]{summary_path}[/cyan]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brain MRI inference demo with pytest-style configuration filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available configurations
  %(prog)s --list-configs

  # Run all DPS denoiser experiments
  %(prog)s -k "DPSDenoiser"

  # Run specific configuration
  %(prog)s -k "DPSDenoiser_DDIMIntegrator_RadialMask"

  # Run multiple denoisers with DDIM and Radial mask
  %(prog)s -k "DDIM and Radial"

  # Exclude certain configurations
  %(prog)s -k "not EulerMaruyama"

  # Run with custom sampling parameters
  %(prog)s -k "DPS" --num-samples 32 --num-steps 100
        """,
    )

    # Generate lists for help text
    denoiser_names = [name for name, _, _ in AVAILABLE_DENOISERS]
    integrator_names = [name for name, _, _ in AVAILABLE_INTEGRATORS]
    mask_names = [name for name, _, _ in AVAILABLE_MASKS]

    # Configuration selection
    parser.add_argument(
        "-k",
        dest="filter_expr",
        type=str,
        default=None,
        help=f"""Filter configurations using pytest-style expressions (e.g., 'DPS and Radial' or 'not EulerMaruyama').
If not specified, ALL configurations will be run.

Available components to filter:
  Denoisers: {", ".join(denoiser_names)}
  Integrators: {", ".join(integrator_names)}
  Masks: {", ".join(mask_names)}

Supports: 'and', 'or', 'not', parentheses.
""",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List all available configurations and exit.",
    )

    # Output configuration
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/brain_ar"),
        help="Root directory where run artifacts will be stored (default: runs/brain_ar).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name; defaults to a timestamped value.",
    )

    # Sampling parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help=f"Number of reconstruction samples to generate (default: {NUM_SAMPLES}).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=COND_NUM_STEPS,
        help=f"Number of diffusion steps (default: {COND_NUM_STEPS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED}).",
    )

    # Experiment control
    parser.add_argument(
        "--skip-unconditional",
        action="store_true",
        help="Skip unconditional generation (faster startup).",
    )

    # Display options
    parser.add_argument(
        "--max-recon-samples",
        type=int,
        default=DEFAULT_MAX_RECON_SAMPLES,
        help=f"Maximum number of reconstructed samples to display per denoiser (default: {DEFAULT_MAX_RECON_SAMPLES}).",
    )

    # Logging options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable JAX trace + device memory profile (writes to <output-root>/<run-name>/jax_profile).",
    )
    parser.add_argument(
        "--checkpoint-predictor",
        action="store_true",
        help="Checkpoint predictor calls (remat) to trade compute for lower memory.",
    )

    # Tweedie debug options
    parser.add_argument(
        "--tweedie-debug-plots",
        action="store_true",
        help="Enable debug plotting inside TweedieProjectionDenoiser.",
    )
    parser.add_argument(
        "--tweedie-debug-dir",
        type=str,
        default="debug_tweedie_plots",
        help="Directory where Tweedie debug plots will be saved (default: debug_tweedie_plots).",
    )
    parser.add_argument(
        "--diffusion-debug-plots",
        action="store_true",
        help="Enable debug plotting inside DiffusionProjectionDenoiser.",
    )
    parser.add_argument(
        "--diffusion-debug-dir",
        type=str,
        default="debug_diffusion_projection",
        help="Directory where diffusion projection debug plots will be saved (default: debug_diffusion_projection).",
    )

    # Clipping options
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable sample clipping (keeps all generated pixels).",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=CLIP_THRESHOLD,
        help=f"Clipping threshold - fraction of max magnitude or percentile (default: {CLIP_THRESHOLD}).",
    )
    parser.add_argument(
        "--clip-method",
        type=str,
        choices=["magnitude", "percentile"],
        default=CLIP_METHOD,
        help=f"Clipping method: 'magnitude' (fraction of max) or 'percentile' (default: {CLIP_METHOD}).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Apply verbosity level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("[yellow]Debug logging enabled[/yellow]")

    # Generate all possible configurations
    all_configs = generate_experiment_configs()

    # Handle --list-configs
    if args.list_configs:
        table = Table(
            title=f"[bold cyan]Available Configurations[/bold cyan] ({len(all_configs)} total)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("#", style="cyan", width=5, justify="right")
        table.add_column("Config ID", style="yellow")

        for idx, cfg in enumerate(all_configs, 1):
            table.add_row(str(idx), cfg.config_id)

        console.print(table)
        console.print("\n[bold]Configuration format:[/bold] <Denoiser>_<Integrator>_<Mask>")
        console.print("\n[bold]Example filters:[/bold]")
        console.print("  [cyan]-k 'DPSDenoiser'[/cyan]                    # All DPS experiments")
        console.print("  [cyan]-k 'RadialMask'[/cyan]                     # All radial mask experiments")
        console.print("  [cyan]-k 'DPS and Radial'[/cyan]                 # DPS with radial mask")
        console.print("  [cyan]-k 'DPS or FPS'[/cyan]                     # All DPS or FPS experiments")
        console.print("  [cyan]-k 'not EulerMaruyama'[/cyan]              # Exclude EulerMaruyama integrator")
        console.print("  [cyan]-k '(DPS or FPS) and RadialMask'[/cyan]    # Complex expression")
        return

    # Filter configurations based on -k argument
    filtered_configs = filter_configs(all_configs, args.filter_expr)

    if not filtered_configs:
        logger.error("[red]ERROR:[/red] No configurations match the filter criteria!")
        logger.error("Filter expression: '[cyan]%s[/cyan]'", args.filter_expr)
        logger.info("\nUse [cyan]--list-configs[/cyan] to see all available configurations.")
        sys.exit(1)

    logger.info(
        "Filtered to [bold]%d[/bold] configuration(s) out of [bold]%d[/bold] total", len(filtered_configs), len(all_configs)
    )
    if args.filter_expr:
        logger.info("Filter: '[cyan]%s[/cyan]'", args.filter_expr)

    # Prepare output directory
    output_dir = prepare_output_dir(args.output_root, args.run_name)

    # Run experiments
    run_demo(
        output_dir=output_dir,
        filtered_configs=filtered_configs,
        max_recon_samples=max(1, args.max_recon_samples),
        num_samples=max(1, args.num_samples),
        num_steps=max(1, args.num_steps),
        seed=args.seed,
        skip_unconditional=args.skip_unconditional,
        tweedie_debug_plots=args.tweedie_debug_plots,
        tweedie_debug_dir=args.tweedie_debug_dir,
        diffusion_debug_plots=args.diffusion_debug_plots,
        diffusion_debug_dir=args.diffusion_debug_dir,
        args=args,
    )


if __name__ == "__main__":
    main()
