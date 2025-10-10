# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for image diffusion models using pretrained MNIST model.

This module tests unconditional and conditional image generation using a pretrained
model from HuggingFace Hub. It covers:
- Unconditional generation with various integrators
- Conditional generation (inpainting) with various denoisers
- Visual debugging with matplotlib plots

Mathematical Framework:
- Flow model (rectified flow): dx/dt = v_θ(x,t), where v is velocity field
- Unconditional: Sample from p(x) by integrating backwards from noise
- Conditional: Sample from p(x|y) using guidance or posterior sampling
"""

import jax
import jax.numpy as jnp
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import os

from diffuse.diffusion.sde import Flow
from diffuse.denoisers.denoiser import Denoiser
from diffuse.denoisers.cond import DPSDenoiser, FPSDenoiser, TMPDenoiser
from diffuse.integrator.deterministic import DDIMIntegrator, DPMpp2sIntegrator, HeunIntegrator
from diffuse.integrator.stochastic import EulerMaruyamaIntegrator
from diffuse.timer import VpTimer
from diffuse.predictor import Predictor
from diffuse.base_forward_model import MeasurementState
from diffuse.examples.mnist.forward_model import SquareMask

# Enable float64 for precise tests (optional for images)
# jax.config.update("jax_enable_x64", True)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def model_components():
    """Load pretrained MNIST model from HuggingFace Hub.

    This fixture is module-scoped to avoid reloading the model for each test.
    Returns model, predictor, and image shape.
    """
    from huggingface_hub import hf_hub_download
    from flax import nnx, serialization
    import importlib.util

    REPO_ID = "jcopo/mnist"

    # Download model weights and config
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.msgpack")
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.py")

    # Load config and initialize model
    spec = importlib.util.spec_from_file_location("model_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    model = config_module.model

    # Load weights
    with open(model_path, "rb") as f:
        state_dict = serialization.from_bytes(None, f.read())

    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(state_dict)
    model = nnx.merge(graphdef, state)
    model.eval()

    # Create Flow model and predictor
    flow = Flow()

    def network_fn(x, t):
        """Neural network velocity field v_θ(x,t)"""
        return model(x, t).output

    predictor = Predictor(model=flow, network=network_fn, prediction_type="velocity")

    img_shape = (28, 28, 1)

    return {
        "model": model,
        "flow": flow,
        "predictor": predictor,
        "img_shape": img_shape,
    }


@pytest.fixture
def test_image(model_components):
    """Load or generate a test image for conditional generation.

    Tries to load from MNIST dataset, otherwise uses a generated sample.
    """
    import einops

    mnist_path = Path(os.environ.get("DSDIR", ".")) / "MNIST" / "mnist.npz"

    if mnist_path.exists():
        data = jnp.load(str(mnist_path))
        test_images = data["X_test"]
        test_images = einops.rearrange(test_images, "b h w -> b h w 1")
        return test_images[0]
    else:
        # Generate a sample if MNIST not available
        key = jax.random.PRNGKey(42)
        flow = model_components["flow"]
        predictor = model_components["predictor"]
        img_shape = model_components["img_shape"]

        timer = VpTimer(eps=1e-3, tf=1.0, n_steps=50)
        integrator = DDIMIntegrator(model=flow, timer=timer)
        denoiser = Denoiser(integrator=integrator, model=flow, predictor=predictor, x0_shape=img_shape)

        state, _ = denoiser.generate(key, 50, 1, keep_history=False)
        return jnp.clip(state.integrator_state.position[0], 0, 1)


@pytest.fixture
def make_conditional_denoiser(model_components):
    """Factory fixture for creating conditional denoisers with proper configuration.

    Returns a function that creates a conditional denoiser with the appropriate
    hyperparameters for each denoiser type.

    Usage:
        denoiser = make_conditional_denoiser(
            denoiser_class=DPSDenoiser,
            integrator=integrator,
            forward_model=mask_model
        )
    """
    flow = model_components["flow"]
    predictor = model_components["predictor"]
    img_shape = model_components["img_shape"]

    def _make_denoiser(denoiser_class, integrator, forward_model):
        """Create a conditional denoiser with class-specific hyperparameters.

        Args:
            denoiser_class: The denoiser class to instantiate
            integrator: The integrator to use
            forward_model: The forward model for measurements

        Returns:
            Configured conditional denoiser instance
        """
        kwargs = {
            "integrator": integrator,
            "model": flow,
            "predictor": predictor,
            "forward_model": forward_model,
            "x0_shape": img_shape,
        }

        # Add class-specific hyperparameters
        if denoiser_class == DPSDenoiser:
            kwargs["zeta"] = 0.5

        return denoiser_class(**kwargs)

    return _make_denoiser


# ============================================================================
# Plotting Utilities
# ============================================================================


def plot_image_grid(images, title, n_rows=2, n_cols=8, figsize=(16, 4)):
    """Plot a grid of images.

    Args:
        images: Array of images (n_samples, H, W, 1) or (n_samples, H, W)
        title: Plot title
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        figsize: Figure size

    Returns:
        Figure object
    """
    n_show = min(len(images), n_rows * n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=14)

    axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx in range(n_show):
        img = jnp.clip(images[idx].squeeze(), 0, 1)
        axes_flat[idx].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes_flat[idx].axis("off")

    # Hide unused subplots
    for idx in range(n_show, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    return fig


def plot_denoising_trajectory(history, title, n_timesteps=6, sample_idx=0, figsize=(18, 3)):
    """Plot denoising trajectory showing noise → clean progression.

    Args:
        history: Array of images over time (n_steps, n_samples, H, W, 1)
        title: Plot title
        n_timesteps: Number of timesteps to show
        sample_idx: Which sample to visualize
        figsize: Figure size

    Returns:
        Figure object
    """
    n_steps = len(history)
    timestep_indices = jnp.linspace(0, n_steps - 1, n_timesteps, dtype=int)

    fig, axes = plt.subplots(1, n_timesteps, figsize=figsize)
    fig.suptitle(title, fontsize=14)

    # Reverse history to show noise → clean
    history = history[::-1]

    for col, t_idx in enumerate(timestep_indices):
        img = jnp.clip(history[t_idx, sample_idx].squeeze(), 0, 1)
        axes[col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[col].set_title(f"Step {t_idx}", fontsize=10)
        axes[col].axis("off")

    plt.tight_layout()
    return fig


def plot_inpainting_comparison(original, masked, reconstructions, title, figsize=(14, 3)):
    """Plot inpainting results: original, masked, and reconstructions.

    Args:
        original: Original image (H, W, 1)
        masked: Masked measurement (H, W, 1)
        reconstructions: Reconstructed images (n_samples, H, W, 1)
        title: Plot title
        figsize: Figure size

    Returns:
        Figure object
    """
    n_samples = min(len(reconstructions), 6)
    n_cols = n_samples + 2  # +2 for original and masked

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=14)

    # Original
    axes[0].imshow(original.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Masked
    axes[1].imshow(masked.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Masked")
    axes[1].axis("off")

    # Reconstructions
    for idx in range(n_samples):
        img = jnp.clip(reconstructions[idx].squeeze(), 0, 1)
        axes[idx + 2].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[idx + 2].set_title(f"Sample {idx + 1}")
        axes[idx + 2].axis("off")

    plt.tight_layout()
    return fig


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.skip(reason="Temporarily disabled")
@pytest.mark.parametrize(
    "integrator_class",
    [
        pytest.param(DDIMIntegrator, id="DDIM"),
        pytest.param(DPMpp2sIntegrator, id="DPM++"),
    ],
)
def test_unconditional_generation(model_components, integrator_class, plot_if_enabled):
    """Test unconditional MNIST generation with different integrators.

    Validates that the model can generate realistic MNIST digits from noise
    using different numerical integration schemes.

    Args:
        model_components: Fixture providing model, predictor, etc.
        integrator_class: Integrator class to test
        plot_if_enabled: Fixture for conditional plotting
    """
    flow = model_components["flow"]
    predictor = model_components["predictor"]
    img_shape = model_components["img_shape"]

    # Setup
    n_steps = 50
    n_samples = 16
    key = jax.random.PRNGKey(456)
    timer = VpTimer(eps=1e-3, tf=1.0, n_steps=n_steps)

    # Create integrator
    integrator = integrator_class(model=flow, timer=timer)

    # Create denoiser
    denoiser = Denoiser(integrator=integrator, model=flow, predictor=predictor, x0_shape=img_shape)

    # Generate samples
    state, history = denoiser.generate(key, n_steps, n_samples, keep_history=True)
    samples = state.integrator_state.position

    # Validate output shape
    assert samples.shape == (n_samples, *img_shape), f"Expected shape {(n_samples, *img_shape)}, got {samples.shape}"

    # Validate output range (should be reasonable, even if not strictly [0,1])
    assert jnp.min(samples) > -2.0, f"Minimum value {jnp.min(samples)} is too low"
    assert jnp.max(samples) < 2.0, f"Maximum value {jnp.max(samples)} is too high"

    # Get clean test ID for plotting
    integrator_name = integrator_class.__name__.replace("Integrator", "")

    # Plot results
    plot_if_enabled(lambda: plot_image_grid(samples, f"Unconditional Generation - {integrator_name}", n_rows=2, n_cols=8))
    plot_if_enabled(lambda: plot_denoising_trajectory(history, f"Denoising Trajectory - {integrator_name}", n_timesteps=6, sample_idx=0))

    print(f"✓ Unconditional generation with {integrator_name}: {samples.shape}")


@pytest.mark.skip(reason="Temporarily disabled")
@pytest.mark.parametrize(
    "denoiser_class",
    [
        pytest.param(DPSDenoiser, id="DPS"),
        pytest.param(FPSDenoiser, id="FPS"),
        pytest.param(TMPDenoiser, id="TMP"),
    ],
)
@pytest.mark.parametrize(
    "integrator_class",
    [
        pytest.param(DDIMIntegrator, id="DDIM"),
        pytest.param(DPMpp2sIntegrator, id="DPM++"),
        pytest.param(HeunIntegrator, id="Heun"),
        pytest.param(EulerMaruyamaIntegrator, id="EM"),
    ],
)
def test_conditional_inpainting(model_components, test_image, denoiser_class, integrator_class, make_conditional_denoiser, plot_if_enabled):
    """Test conditional MNIST inpainting with different denoisers and integrators.

    Validates that conditional denoisers can restore masked regions of MNIST digits
    using measurement guidance.

    Args:
        model_components: Fixture providing model, predictor, etc.
        test_image: Test image for inpainting
        denoiser_class: Conditional denoiser class to test
        integrator_class: Integrator class to test
        make_conditional_denoiser: Factory fixture for creating denoisers
        plot_if_enabled: Fixture for conditional plotting
    """
    flow = model_components["flow"]

    # Setup
    n_steps = 50
    n_samples = 8
    key = jax.random.PRNGKey(789)
    timer = VpTimer(eps=1e-3, tf=1.0, n_steps=n_steps)

    # Create mask and measurement
    mask_model = SquareMask(size=12, img_shape=test_image.shape, std=0.1)
    xi_center = jnp.array([14.0, 14.0])  # Center of 28x28 image
    mask = mask_model.make(xi_center)
    masked_image = test_image * mask
    measurement_state = MeasurementState(y=masked_image, mask_history=mask)

    # Create integrator and conditional denoiser
    integrator = integrator_class(model=flow, timer=timer)
    cond_denoiser = make_conditional_denoiser(denoiser_class, integrator, mask_model)

    # Generate conditional samples
    state, history = cond_denoiser.generate(key, measurement_state, n_steps, n_samples, keep_history=True)
    samples = state.integrator_state.position

    # Validate output shape
    img_shape = model_components["img_shape"]
    assert samples.shape == (n_samples, *img_shape), f"Expected shape {(n_samples, *img_shape)}, got {samples.shape}"

    # Get clean test IDs for plotting
    denoiser_name = denoiser_class.__name__.replace("Denoiser", "")
    integrator_name = integrator_class.__name__.replace("Integrator", "")

    # Plot results
    plot_if_enabled(lambda: plot_inpainting_comparison(test_image, masked_image, samples, f"Inpainting - {denoiser_name} + {integrator_name}"))
    plot_if_enabled(
        lambda: plot_denoising_trajectory(history, f"Inpainting Trajectory - {denoiser_name} + {integrator_name}", n_timesteps=6, sample_idx=0)
    )

    # Validate that visible region (inside mask) is preserved
    visible_region_samples = samples * mask
    visible_region_measurement = masked_image * mask
    visible_mse = jnp.mean((visible_region_samples - visible_region_measurement) ** 2)
    assert visible_mse < 0.1, f"Visible region not preserved (MSE={visible_mse:.6f})"

    # Validate that reconstructed region is different from masked region
    inv_mask = 1 - mask
    reconstructed_region = samples * inv_mask
    original_masked_region = masked_image * inv_mask
    diff = jnp.mean(jnp.abs(reconstructed_region - original_masked_region))
    assert diff > 0.01, f"Reconstruction is too similar to masked input (diff={diff})"
    # Compute reconstruction quality (MSE on masked region)
    mse = jnp.mean((samples * inv_mask - test_image * inv_mask) ** 2)
    print(f"✓ Conditional inpainting with {denoiser_name} + {integrator_name}: MSE = {mse:.6f}")


@pytest.mark.skip(reason="Temporarily disabled")
@pytest.mark.slow
def test_restore_with_zero_measured(model_components, make_conditional_denoiser, plot_if_enabled):
    """Test restoration when measurement is zero (pure generation in masked region).

    This is a challenging case where the denoiser must generate content from scratch
    in the masked region without any measurement information.

    Args:
        model_components: Fixture providing model, predictor, etc.
        make_conditional_denoiser: Factory fixture for creating denoisers
        plot_if_enabled: Fixture for conditional plotting
    """
    flow = model_components["flow"]
    img_shape = model_components["img_shape"]

    # Setup
    n_steps = 50
    n_samples = 8
    key = jax.random.PRNGKey(999)
    timer = VpTimer(eps=1e-3, tf=1.0, n_steps=n_steps)

    # Create mask and zero measurement
    mask_model = SquareMask(size=12, img_shape=img_shape, std=0.1)
    xi_center = jnp.array([14.0, 14.0])
    mask = mask_model.make(xi_center)
    zero_measurement = jnp.zeros(img_shape)
    measurement_state = MeasurementState(y=zero_measurement, mask_history=mask)

    # Create denoiser (use DPS for this test)
    integrator = DDIMIntegrator(model=flow, timer=timer)
    cond_denoiser = make_conditional_denoiser(DPSDenoiser, integrator, mask_model)

    # Generate samples
    state, history = cond_denoiser.generate(key, measurement_state, n_steps, n_samples, keep_history=True)
    samples = state.integrator_state.position

    # Validate output shape
    assert samples.shape == (n_samples, *img_shape)

    # Validate that samples have reasonable statistics
    mean_val = jnp.mean(samples)
    std_val = jnp.std(samples)
    assert 0.0 < mean_val < 1.0, f"Mean {mean_val} is outside expected range"
    assert 0.1 < std_val < 1.0, f"Std {std_val} is outside expected range"

    # Plot results
    plot_if_enabled(lambda: plot_image_grid(samples, "Restoration from Zero Measurement (DPS + DDIM)", n_rows=2, n_cols=4))
    plot_if_enabled(lambda: plot_denoising_trajectory(history, "Zero Measurement Trajectory", n_timesteps=6, sample_idx=0))

    print(f"✓ Restoration from zero measurement: mean={mean_val:.3f}, std={std_val:.3f}")


@pytest.mark.skip(reason="Temporarily disabled")
@pytest.mark.visual
def test_visual_comparison_all_methods(model_components, test_image, make_conditional_denoiser, plot_if_enabled):
    """Visual comparison of all conditional methods × integrators (like tutorial notebook).

    This test generates a comprehensive grid comparing all combinations of:
    - Conditional methods: DPS, FPS, TMP
    - Integrators: DDIM, DPM++

    To run this test specifically:
        pytest test/test_image.py::test_visual_comparison_all_methods
        pytest test/test_image.py -m visual

    This test is marked with @pytest.mark.visual and won't run by default.
    """
    flow = model_components["flow"]

    # Setup
    n_steps = 50
    n_samples = 8
    key = jax.random.PRNGKey(42)
    timer = VpTimer(eps=1e-3, tf=1.0, n_steps=n_steps)

    # Create mask and measurement
    mask_model = SquareMask(size=12, img_shape=test_image.shape, std=0.1)
    xi_center = jnp.array([14.0, 14.0])
    mask = mask_model.make(xi_center)
    masked_image = test_image * mask
    measurement_state = MeasurementState(y=masked_image, mask_history=mask)

    # Define all combinations
    methods = {
        "DPS": DPSDenoiser,
        "FPS": FPSDenoiser,
        "TMP": TMPDenoiser,
    }

    integrator_configs = {
        "DDIM": DDIMIntegrator,
        "DPM++": DPMpp2sIntegrator,
    }

    # Generate samples for all combinations
    print("\n" + "=" * 60)
    print("Running visual comparison for all method × integrator combinations")
    print("=" * 60)

    results = {}
    keys = jax.random.split(key, len(methods) * len(integrator_configs))
    key_idx = 0

    for method_name, method_class in methods.items():
        for int_name, int_class in integrator_configs.items():
            integrator = int_class(model=flow, timer=timer)
            denoiser = make_conditional_denoiser(method_class, integrator, mask_model)

            state, history = denoiser.generate(keys[key_idx], measurement_state, n_steps, n_samples, keep_history=True)

            combo_name = f"{method_name}_{int_name}"
            results[combo_name] = {
                "samples": state.integrator_state.position,
                "history": history,
                "method": method_name,
                "integrator": int_name,
            }
            print(f"  ✓ {combo_name}")
            key_idx += 1

    # Create grouped visualization
    def plot_grouped_comparison():
        n_methods = len(methods)
        n_integrators = len(integrator_configs)
        n_show = min(n_samples, 6)

        # Main comparison plot: rows = method × integrator, cols = samples
        fig, axes = plt.subplots(n_methods * n_integrators + 2, n_show, figsize=(14, 2.5 * (n_methods * n_integrators + 2)))
        fig.suptitle("Conditional Generation: All Methods × Integrators", fontsize=16, y=0.995)

        # Row 0: Original image
        for col in range(n_show):
            axes[0, col].imshow(test_image.squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[0, col].axis("off")
        axes[0, 0].text(-0.15, 0.5, "Original", transform=axes[0, 0].transAxes, fontsize=13, fontweight="bold", va="center", ha="right")

        # Row 1: Masked measurement
        for col in range(n_show):
            axes[1, col].imshow(masked_image.squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[1, col].axis("off")
        axes[1, 0].text(-0.15, 0.5, "Masked", transform=axes[1, 0].transAxes, fontsize=13, fontweight="bold", va="center", ha="right")

        # Rows 2+: Reconstructions
        row_idx = 2
        for method_name in methods.keys():
            for int_name in integrator_configs.keys():
                combo = f"{method_name}_{int_name}"
                samples = jnp.clip(results[combo]["samples"], 0, 1)

                for col in range(n_show):
                    axes[row_idx, col].imshow(samples[col].squeeze(), cmap="gray", vmin=0, vmax=1)
                    axes[row_idx, col].axis("off")

                label = f"{method_name}\n{int_name}"
                axes[row_idx, 0].text(
                    -0.15, 0.5, label, transform=axes[row_idx, 0].transAxes, fontsize=12, fontweight="bold", va="center", ha="right"
                )
                row_idx += 1

        plt.tight_layout()
        return fig

    # Trajectory comparison plot
    def plot_grouped_trajectories():
        n_methods = len(methods)
        n_integrators = len(integrator_configs)
        n_timesteps = 6
        timestep_indices = jnp.linspace(0, n_steps - 1, n_timesteps, dtype=int)
        sample_idx = 0

        fig, axes = plt.subplots(n_methods * n_integrators, n_timesteps, figsize=(18, 2.5 * n_methods * n_integrators))
        fig.suptitle("Denoising Trajectories: All Methods × Integrators", fontsize=16, y=0.995)

        row_idx = 0
        for method_name in methods.keys():
            for int_name in integrator_configs.keys():
                combo = f"{method_name}_{int_name}"
                history = jnp.clip(results[combo]["history"][::-1], 0, 1)

                for col, t_idx in enumerate(timestep_indices):
                    axes[row_idx, col].imshow(history[t_idx, sample_idx].squeeze(), cmap="gray", vmin=0, vmax=1)
                    axes[row_idx, col].axis("off")
                    if row_idx == 0:
                        axes[row_idx, col].set_title(f"Step {t_idx}", fontsize=10)

                label = f"{method_name}\n{int_name}"
                axes[row_idx, 0].text(
                    -0.08, 0.5, label, transform=axes[row_idx, 0].transAxes, fontsize=12, fontweight="bold", va="center", ha="right"
                )
                row_idx += 1

        plt.tight_layout()
        return fig

    # Reconstruction quality comparison
    def plot_reconstruction_quality():
        inv_mask = 1 - mask
        n_methods = len(methods)
        n_integrators = len(integrator_configs)
        n_show = min(n_samples, 6)

        fig, axes = plt.subplots(n_methods * n_integrators, n_show, figsize=(14, 2 * n_methods * n_integrators))
        fig.suptitle("Reconstructed Regions Only (Masked Area)", fontsize=16, y=0.995)

        row_idx = 0
        for method_name in methods.keys():
            for int_name in integrator_configs.keys():
                combo = f"{method_name}_{int_name}"
                samples = jnp.clip(results[combo]["samples"], 0, 1)

                for col in range(n_show):
                    reconstructed = samples[col] * inv_mask
                    axes[row_idx, col].imshow(reconstructed.squeeze(), cmap="gray", vmin=0, vmax=1)
                    axes[row_idx, col].axis("off")

                label = f"{method_name}\n{int_name}"
                axes[row_idx, 0].text(
                    -0.15, 0.5, label, transform=axes[row_idx, 0].transAxes, fontsize=12, fontweight="bold", va="center", ha="right"
                )
                row_idx += 1

        plt.tight_layout()
        return fig

    # Display all plots
    plot_if_enabled(plot_grouped_comparison)
    plot_if_enabled(plot_grouped_trajectories)
    plot_if_enabled(plot_reconstruction_quality)

    # Print reconstruction quality metrics
    print("\n" + "=" * 60)
    print("Reconstruction Quality (MSE on masked region):")
    print("=" * 60)
    print(f"{'Method':<10} {'Integrator':<12} {'MSE':<12}")
    print("-" * 60)

    inv_mask = 1 - mask
    for method_name in methods.keys():
        for int_name in integrator_configs.keys():
            combo = f"{method_name}_{int_name}"
            samples = results[combo]["samples"]
            mse = jnp.mean((samples * inv_mask - test_image * inv_mask) ** 2)
            print(f"{method_name:<10} {int_name:<12} {mse:.8f}")

    print("\n✓ Visual comparison complete")

    # Validate reconstruction quality
    img_shape = model_components["img_shape"]
    assert all(results[combo]["samples"].shape == (n_samples, *img_shape) for combo in results.keys()), "All samples should have correct shape"
