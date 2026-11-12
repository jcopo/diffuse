# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
import jax
import jax.numpy as jnp
import pytest
import numpy as np
import matplotlib.pyplot as plt

from diffuse.base_forward_model import MeasurementState
from diffuse.examples.mri import (
    RadialMask,
    RandomMask,
    SpiralMask,
    VerticalMask,
    VerticalDiscreteMask,
    FullMask,
    load_brain_slice,
    load_brain_ar,
)


@pytest.fixture(scope="module")
def img_shape() -> tuple[int, int, int]:
    return (8, 8, 2)


@pytest.fixture(scope="module")
def toy_image(img_shape: tuple[int, int, int]) -> jnp.ndarray:
    h, w, _ = img_shape
    grid_x = jnp.linspace(-1.0, 1.0, num=h * w).reshape(h, w)
    grid_y = jnp.linspace(1.0, 2.0, num=h * w).reshape(h, w)
    return jnp.stack([grid_x, grid_y], axis=-1).astype(jnp.float32)


# Cartesian FFT-based mask cases (radial and spiral excluded as they now use NUFFT)
MASK_CASES = (
    (
        "random",
        lambda shape: RandomMask(img_shape=shape, std=0.05, data_model="WMH"),
        lambda mask_model, rng: mask_model.sample_design(rng),
        True,
    ),
    (
        "vertical",
        lambda shape: VerticalMask(num_lines=3, img_shape=shape, std=0.05, data_model="kneeFastMRI"),
        lambda mask_model, rng: jnp.array([0.2, 0.5, 0.8], dtype=jnp.float32),
        False,
    ),
    (
        "vertical_discrete",
        lambda shape: VerticalDiscreteMask(num_lines=3, img_shape=shape, std=0.05, data_model="kneeFastMRI"),
        lambda mask_model, rng: jnp.array([0, 2, 5], dtype=jnp.int32),
        False,
    ),
)


def _plot_results(case_id, mask, image, measurement, reconstructed, plot_if_enabled):
    def _plot():
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        fig.suptitle(f"MRI mask inspection – {case_id}")

        mask_np = np.asarray(mask)
        axes[0].imshow(mask_np, cmap="viridis")
        axes[0].set_title("Mask")
        axes[0].axis("off")

        image_mag = np.linalg.norm(np.asarray(image), axis=-1)
        axes[1].imshow(image_mag, cmap="gray")
        axes[1].set_title("Input |x|")
        axes[1].axis("off")

        meas_mag = np.linalg.norm(np.asarray(measurement), axis=-1)
        axes[2].imshow(meas_mag, cmap="magma")
        axes[2].set_title("|A x|")
        axes[2].axis("off")

        recon_mag = np.linalg.norm(np.asarray(reconstructed), axis=-1)
        axes[3].imshow(recon_mag, cmap="gray")
        axes[3].set_title("|Aᵀ A x|")
        axes[3].axis("off")

        plt.tight_layout()

    plot_if_enabled(_plot)


@pytest.mark.parametrize(
    "case_id, mask_factory, design_factory, needs_rng",
    MASK_CASES,
    ids=[case[0] for case in MASK_CASES],
)
def test_mri_masks_apply_adjoint_consistency(
    case_id,
    mask_factory,
    design_factory,
    needs_rng,
    img_shape,
    toy_image,
    plot_if_enabled,
):
    mask_model = mask_factory(img_shape)
    rng = jax.random.PRNGKey(0)
    design = design_factory(mask_model, rng)

    mask = mask_model.make(design, rng if needs_rng else None)
    assert mask.shape == img_shape[:2]
    assert jnp.all((mask >= 0.0) & (mask <= 1.0))
    assert jnp.any(mask > 0.0), f"{case_id} produced an empty mask"

    measurement_state = MeasurementState(y=jnp.zeros_like(toy_image), mask_history=mask)

    measurement = mask_model.apply(toy_image, measurement_state)
    assert measurement.shape == toy_image.shape

    reconstructed = mask_model.adjoint(measurement, measurement_state)
    assert reconstructed.shape == toy_image.shape
    assert jnp.max(jnp.abs(reconstructed)) > 0.0

    restored = mask_model.restore(measurement, measurement_state)
    assert jnp.allclose(reconstructed, restored, atol=1e-6)

    _plot_results(case_id, mask, toy_image, measurement, reconstructed, plot_if_enabled)

    consistency = mask_model.apply(reconstructed, measurement_state)
    assert jnp.allclose(consistency, measurement, atol=1e-5, rtol=1e-5), f"{case_id} apply/adjoint consistency failed"


@pytest.mark.parametrize(
    "case_id, mask_factory, design_factory, needs_rng",
    MASK_CASES,
    ids=[case[0] for case in MASK_CASES],
)
def test_mri_masks_brain_data(
    case_id,
    mask_factory,
    design_factory,
    needs_rng,
    plot_if_enabled,
):
    """Test MRI masks on real brain data from brain.h5."""
    # Load real brain MRI data
    brain_img_np = load_brain_slice("brain.h5", slice_idx=8, normalize=True)
    brain_img = jnp.array(brain_img_np)
    brain_shape = brain_img.shape

    mask_model = mask_factory(brain_shape)
    rng = jax.random.PRNGKey(42)
    design = design_factory(mask_model, rng)

    mask = mask_model.make(design, rng if needs_rng else None)
    assert mask.shape == brain_shape[:2]
    assert jnp.all((mask >= 0.0) & (mask <= 1.0))
    assert jnp.any(mask > 0.0), f"{case_id} produced an empty mask"

    measurement_state = MeasurementState(y=jnp.zeros_like(brain_img), mask_history=mask)

    measurement = mask_model.apply(brain_img, measurement_state)
    assert measurement.shape == brain_img.shape

    reconstructed = mask_model.adjoint(measurement, measurement_state)
    assert reconstructed.shape == brain_img.shape
    assert jnp.max(jnp.abs(reconstructed)) > 0.0

    restored = mask_model.restore(measurement, measurement_state)
    assert jnp.allclose(reconstructed, restored, atol=1e-6)

    _plot_results(f"{case_id}_brain", mask, brain_img, measurement, reconstructed, plot_if_enabled)

    consistency = mask_model.apply(reconstructed, measurement_state)
    assert jnp.allclose(consistency, measurement, atol=1e-5, rtol=1e-5), (
        f"{case_id} apply/adjoint consistency failed on brain data"
    )


def test_full_mask_brain_reconstruction(plot_if_enabled):
    """Test that full k-space sampling perfectly reconstructs the brain image.

    With a full mask (all k-space frequencies sampled), the operation A^T * A
    should be very close to the identity, meaning we can perfectly reconstruct
    the original image. This serves as a sanity check for the forward/adjoint
    operators.
    """
    # Load real brain MRI data
    brain_img_np = load_brain_slice("brain.h5", slice_idx=8, normalize=True)
    brain_img = jnp.array(brain_img_np)
    brain_shape = brain_img.shape

    # Create full mask (all ones - no undersampling)
    mask_model = FullMask(img_shape=brain_shape, std=0.01, data_model="kneeFastMRI")
    rng = jax.random.PRNGKey(42)

    # Design parameters are ignored for full mask
    design = jnp.array([])
    mask = mask_model.make(design, None)

    # Verify mask is all ones
    assert mask.shape == brain_shape[:2]
    assert jnp.allclose(mask, 1.0), "Full mask should be all ones"

    measurement_state = MeasurementState(y=jnp.zeros_like(brain_img), mask_history=mask)

    # Apply forward operator: y = A * x
    measurement = mask_model.apply(brain_img, measurement_state)
    assert measurement.shape == brain_img.shape

    # Apply adjoint operator: x_recon = A^T * y
    reconstructed = mask_model.adjoint(measurement, measurement_state)
    assert reconstructed.shape == brain_img.shape

    # With full mask, A^T * A * x should equal x (perfect reconstruction)
    reconstruction_error = jnp.max(jnp.abs(reconstructed - brain_img))
    relative_error = reconstruction_error / jnp.max(jnp.abs(brain_img))

    print(f"\nFull mask reconstruction metrics:")
    print(f"  Max absolute error: {reconstruction_error:.2e}")
    print(f"  Relative error: {relative_error:.2e}")
    print(f"  Mean squared error: {jnp.mean((reconstructed - brain_img) ** 2):.2e}")

    # Check that reconstruction is very close to original
    assert jnp.allclose(reconstructed, brain_img, atol=1e-5, rtol=1e-5), (
        f"Full mask reconstruction failed: max error = {reconstruction_error:.2e}"
    )

    # Verify restore gives same result as adjoint
    restored = mask_model.restore(measurement, measurement_state)
    assert jnp.allclose(reconstructed, restored, atol=1e-6)

    # Plot results if enabled
    def _plot():
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle("Full k-space sampling - Perfect reconstruction check")

        mask_np = np.asarray(mask)
        axes[0].imshow(mask_np, cmap="viridis", vmin=0, vmax=1)
        axes[0].set_title("Full Mask (all ones)")
        axes[0].axis("off")

        brain_mag = np.linalg.norm(np.asarray(brain_img), axis=-1)
        axes[1].imshow(brain_mag, cmap="gray")
        axes[1].set_title("Original Brain |x|")
        axes[1].axis("off")

        recon_mag = np.linalg.norm(np.asarray(reconstructed), axis=-1)
        axes[2].imshow(recon_mag, cmap="gray")
        axes[2].set_title(f"Reconstructed |A^T A x|")
        axes[2].axis("off")

        # Show absolute error map
        error_mag = np.linalg.norm(np.asarray(reconstructed - brain_img), axis=-1)
        im = axes[3].imshow(error_mag, cmap="hot")
        axes[3].set_title(f"Error |A^T A x - x|\nMax: {reconstruction_error:.2e}")
        axes[3].axis("off")
        plt.colorbar(im, ax=axes[3], fraction=0.046)

        plt.tight_layout()

    plot_if_enabled(_plot)


def test_load_brain_ar(plot_if_enabled):
    """Test loading brain.ar file."""
    # Load all records from brain.ar
    records = load_brain_ar("brain.ar")

    print(f"\nLoaded {len(records)} records from brain.ar")
    print(f"Record shape: {records[0].shape}")
    print(f"Record dtype: {records[0].dtype}")
    print(f"Value range: [{records[0].min():.3f}, {records[0].max():.3f}]")

    # Verify format matches h5 loader
    assert len(records) > 0, "No records loaded from brain.ar"
    assert records[0].shape[-1] == 2, "Expected 2 channels [real, imag]"
    assert records[0].dtype == np.float32, "Expected float32 dtype"

    # Test with MRI forward model (using Cartesian VerticalMask)
    brain_img = jnp.array(records[0])
    mask_model = VerticalMask(
        num_lines=3,
        img_shape=brain_img.shape,
        std=0.05,
        data_model="kneeFastMRI",
    )

    design = jnp.array([0.2, 0.5, 0.8], dtype=jnp.float32)
    mask = mask_model.make(design, None)

    measurement_state = MeasurementState(y=jnp.zeros_like(brain_img), mask_history=mask)
    measurement = mask_model.apply(brain_img, measurement_state)
    reconstructed = mask_model.adjoint(measurement, measurement_state)

    assert reconstructed.shape == brain_img.shape
    print("✓ brain.ar works with MRI forward models")

    # Plot results using existing _plot_results function
    _plot_results("brain_ar_vertical", mask, brain_img, measurement, reconstructed, plot_if_enabled)


# ============================================================================
# NUFFT-specific tests
# ============================================================================

NUFFT_MASK_CASES = [
    (
        "radial_nufft",
        lambda shape: RadialMask(
            num_lines=4,
            img_shape=shape,
            std=0.01,
            data_model="kneeFastMRI",
            samples_per_line=128,
        ),
        lambda mask_model, rng: jnp.array(
            [[0.0, 100.0], [jnp.pi / 2, 100.0], [jnp.pi, 100.0], [3 * jnp.pi / 2, 100.0]], dtype=jnp.float32
        ),
    ),
    (
        "spiral_nufft",
        lambda shape: SpiralMask(
            num_samples=64,
            n_arms=2,
            img_shape=shape,
            std=0.01,
            data_model="kneeFastMRI",
        ),
        lambda mask_model, rng: jnp.array([0.5, 0.5, 0.0], dtype=jnp.float32),
    ),
]


@pytest.mark.parametrize(
    "case_id, mask_factory, design_factory",
    NUFFT_MASK_CASES,
    ids=[case[0] for case in NUFFT_MASK_CASES],
)
def test_nufft_adjoint_test(case_id, mask_factory, design_factory, img_shape):
    """Test the adjoint property: <Ax, y> = <x, A*y> for NUFFT operators.

    This is the mathematical definition of an adjoint operator and should hold
    exactly (up to numerical precision) for properly implemented NUFFT forward/adjoint.
    """
    mask_model = mask_factory(img_shape)
    rng = jax.random.PRNGKey(123)
    design = design_factory(mask_model, rng)

    # Initialize mask and measurement state
    mask = mask_model.make(design, None)
    measurement_state = MeasurementState(y=jnp.zeros(img_shape), mask_history=mask)

    # Create random test vectors
    x = jax.random.normal(rng, img_shape)
    y_key = jax.random.split(rng)[0]

    # Apply forward operator to get measurement shape
    Ax = mask_model.apply(x, measurement_state)
    y = jax.random.normal(y_key, Ax.shape)

    # Update measurement state with actual y
    measurement_state_y = MeasurementState(y=y, mask_history=mask)

    # Compute <Ax, y> and <x, A*y>
    Ax_dot_y = jnp.sum(Ax * y)  # Inner product <Ax, y>
    Aty = mask_model.adjoint(y, measurement_state_y)
    x_dot_Aty = jnp.sum(x * Aty)  # Inner product <x, A*y>

    print(f"\n{case_id} adjoint test:")
    print(f"  <Ax, y> = {Ax_dot_y:.6f}")
    print(f"  <x, A*y> = {x_dot_Aty:.6f}")
    print(f"  Relative error: {jnp.abs(Ax_dot_y - x_dot_Aty) / jnp.abs(Ax_dot_y):.2e}")

    # The adjoint test should pass with high precision
    assert jnp.allclose(Ax_dot_y, x_dot_Aty, rtol=1e-4, atol=1e-6), f"{case_id} adjoint test failed: <Ax, y> != <x, A*y>"


@pytest.mark.parametrize(
    "case_id, mask_factory, design_factory",
    NUFFT_MASK_CASES,
    ids=[case[0] for case in NUFFT_MASK_CASES],
)
def test_nufft_different_shapes(case_id, mask_factory, design_factory):
    """Test NUFFT forward models with various image sizes."""
    test_shapes = [
        (32, 32, 2),
        (64, 64, 2),
        (16, 32, 2),  # Non-square
    ]

    rng = jax.random.PRNGKey(456)

    for shape in test_shapes:
        mask_model = mask_factory(shape)
        design = design_factory(mask_model, rng)

        # Initialize
        mask = mask_model.make(design, None)
        measurement_state = MeasurementState(y=jnp.zeros(shape), mask_history=mask)

        # Create test image
        img = jax.random.normal(rng, shape)

        # Test forward and adjoint
        measurement = mask_model.apply(img, measurement_state)
        reconstructed = mask_model.adjoint(measurement, measurement_state)

        assert reconstructed.shape == shape, f"Shape mismatch for {shape}"
        assert jnp.isfinite(reconstructed).all(), f"Non-finite values for {shape}"
        assert jnp.abs(reconstructed).max() > 0, f"Zero reconstruction for {shape}"

        print(f"✓ {case_id} works with shape {shape}")


@pytest.mark.parametrize(
    "case_id, mask_factory, design_factory",
    NUFFT_MASK_CASES,
    ids=[case[0] for case in NUFFT_MASK_CASES],
)
def test_nufft_design_encoding(case_id, mask_factory, design_factory, img_shape):
    """Test that NUFFT models properly encode design parameters in mask."""
    mask_model = mask_factory(img_shape)
    rng = jax.random.PRNGKey(789)
    design = design_factory(mask_model, rng)

    # Create mask
    mask = mask_model.make(design, None)

    # Mask should have correct shape
    assert mask.shape == img_shape[:2], "Mask shape should match image spatial dimensions"

    # First row should contain encoded parameters (non-zero if parameters are non-zero)
    first_row = mask[0, :]
    if case_id == "radial_nufft":
        # For radial, angles are encoded in first row
        angles = first_row[: mask_model.num_lines]
        assert jnp.any(angles != 0), "Angles should be encoded in first row"
        print(f"\n{case_id} design encoding:")
        print(f"  Encoded angles: {angles}")
    elif case_id == "spiral_nufft":
        # For spiral, [fov, k_max, angle_offset] are encoded
        params = first_row[:3]
        assert jnp.any(params != 0), "Spiral parameters should be encoded in first row"
        print(f"\n{case_id} design encoding:")
        print(f"  Encoded params [fov, k_max, angle_offset]: {params}")


@pytest.mark.parametrize(
    "case_id, mask_factory, design_factory",
    NUFFT_MASK_CASES,
    ids=[case[0] for case in NUFFT_MASK_CASES],
)
def test_nufft_real_brain_data(case_id, mask_factory, design_factory, plot_if_enabled):
    """Test NUFFT forward models on real brain MRI data."""
    # Load real brain data
    brain_img_np = load_brain_slice("brain.h5", slice_idx=8, normalize=True)
    brain_img = jnp.array(brain_img_np)
    brain_shape = brain_img.shape

    mask_model = mask_factory(brain_shape)
    rng = jax.random.PRNGKey(999)
    design = design_factory(mask_model, rng)

    # Initialize and test
    mask = mask_model.make(design, None)
    measurement_state = MeasurementState(y=jnp.zeros_like(brain_img), mask_history=mask)

    measurement = mask_model.apply(brain_img, measurement_state)
    reconstructed = mask_model.adjoint(measurement, measurement_state)

    assert reconstructed.shape == brain_shape
    assert jnp.isfinite(reconstructed).all(), f"{case_id} produced non-finite values"
    assert jnp.abs(reconstructed).max() > 0, f"{case_id} produced zero reconstruction"

    # Test adjoint property on real data
    y_test = jax.random.normal(rng, measurement.shape)
    measurement_state_test = MeasurementState(y=y_test, mask_history=mask)

    Ax_dot_y = jnp.sum(measurement * y_test)
    Aty = mask_model.adjoint(y_test, measurement_state_test)
    x_dot_Aty = jnp.sum(brain_img * Aty)

    rel_error = jnp.abs(Ax_dot_y - x_dot_Aty) / jnp.abs(Ax_dot_y)
    print(f"\n{case_id} on brain data:")
    print(f"  Adjoint test relative error: {rel_error:.2e}")
    print(f"  Reconstruction energy: {jnp.sum(reconstructed**2):.2e}")

    assert jnp.allclose(Ax_dot_y, x_dot_Aty, rtol=1e-4, atol=1e-6), f"{case_id} adjoint test failed on brain data"

    # Plot results
    _plot_results(
        f"{case_id}_brain",
        mask,
        brain_img,
        measurement,
        reconstructed,
        plot_if_enabled,
    )
