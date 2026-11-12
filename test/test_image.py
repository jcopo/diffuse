import pytest
import jax
import jax.numpy as jnp
from diffuse.examples.mnist.forward_model import SquareMask, MaskState
import matplotlib.pyplot as plt
from diffuse.base_forward_model import MeasurementState


@pytest.fixture
def random_image():
    key = jax.random.PRNGKey(0)
    return jax.random.normal(key, (28, 28, 1))


@pytest.fixture
def square_mask():
    return SquareMask(10, (28, 28, 1))


def plot_grid(images, titles, figsize=(15, 5)):
    """Plot multiple images in a grid."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def test_mask_creation_and_properties(square_mask):
    """Test core mask creation functionality."""
    xi = jnp.array([14.0, 14.0])
    mask = square_mask.make(xi)

    # Shape and value range
    assert mask.shape == (28, 28, 1)
    assert jnp.all(mask >= 0) and jnp.all(mask <= 1)

    # Peak at center, low at edges
    center_value = mask[14, 14, 0]
    edge_value = mask[0, 0, 0]
    assert center_value > 0.9 and edge_value < 0.1


def test_apply_restore_cycle(square_mask, random_image, plot_if_enabled):
    """Test the core apply/restore functionality with visualization."""
    xi = jnp.array([12.0, 16.0])

    # Create mask and measurement state
    mask = square_mask.make(xi)
    measurement_state = MeasurementState(random_image, mask_history=mask)

    # Apply mask (simulate measurement)
    applied = square_mask.apply(random_image, measurement_state)

    # Restore (zeros out measured region)
    restored = square_mask.restore(random_image, measurement_state)

    # Test mathematical properties
    assert applied.shape == random_image.shape
    assert restored.shape == random_image.shape
    assert jnp.allclose(applied, random_image * mask)
    assert jnp.allclose(restored, random_image * (1 - mask))

    # Test that apply + restore ≈ original (soft mask property)
    combined = applied + restored
    max_diff = jnp.max(jnp.abs(combined - random_image))
    max_original = jnp.max(jnp.abs(random_image))
    assert max_diff < max_original * 0.2

    # Visualization
    plot_if_enabled(
        lambda: plot_grid(
            [random_image, mask, applied, restored, combined],
            ["Original", "Mask", "Applied", "Restored", "Combined"],
            figsize=(20, 4),
        )
    )


def test_measurement_accumulation(square_mask, random_image, plot_if_enabled):
    """Test sequential measurement accumulation with visualization."""
    key = jax.random.PRNGKey(42)
    positions = [jnp.array([8.0, 8.0]), jnp.array([20.0, 12.0]), jnp.array([14.0, 20.0])]

    # Start with empty state
    mask_state = square_mask.init(key)
    states = [mask_state]  # Track progression

    for i, xi in enumerate(positions):
        # Create measurement
        measurement = jax.random.normal(jax.random.split(key, len(positions))[i], random_image.shape)

        # Update state
        mask_state = square_mask.update_measurement(mask_state, measurement, xi)
        states.append(mask_state)

        # Verify accumulation (mask_history may be clamped at 1.0, so check coverage)
        coverage_new = jnp.sum(mask_state.mask_history > 0.1)
        coverage_old = jnp.sum(states[i].mask_history > 0.1)
        assert coverage_new >= coverage_old, f"Coverage should increase: {coverage_new} vs {coverage_old}"
        assert not jnp.allclose(mask_state.y, states[i].y)

    # Final verification
    final_coverage = jnp.sum(mask_state.mask_history > 0.1)
    assert final_coverage > 100, f"Final coverage should be substantial: {final_coverage}"

    # Visualization of progression
    plot_if_enabled(
        lambda: plot_grid(
            [states[0].mask_history, states[1].mask_history, states[2].mask_history, states[3].mask_history],
            ["Initial", "After 1st", "After 2nd", "After 3rd"],
            figsize=(16, 4),
        )
    )

    plot_if_enabled(
        lambda: plot_grid(
            [states[3].y, states[3].mask_history], ["Final Measurements", "Final Mask History"], figsize=(10, 4)
        )
    )


def test_mask_positioning_and_sizes(plot_if_enabled):
    """Test different mask sizes and positions with visualization."""
    positions = [jnp.array([7.0, 7.0]), jnp.array([14.0, 14.0]), jnp.array([21.0, 21.0])]
    sizes = [5, 10, 15]

    masks = []
    titles = []

    for size in sizes:
        for pos in positions:
            mask_obj = SquareMask(size, (28, 28, 1))
            mask = mask_obj.make(pos)
            masks.append(mask)
            titles.append(f"Size {size}, Pos {pos}")

            # Test properties
            assert mask.shape == (28, 28, 1)
            assert jnp.all(mask >= 0) and jnp.all(mask <= 1)

            # Test positioning
            y, x = int(pos[1]), int(pos[0])
            if 0 <= y < 28 and 0 <= x < 28:
                assert mask[y, x, 0] > 0.8, f"Peak should be at specified position"

    # Visualization - show a subset
    plot_if_enabled(
        lambda: plot_grid(
            masks[::2],  # Show every other mask to avoid clutter
            titles[::2],
            figsize=(18, 6),
        )
    )


def test_overlap_handling(square_mask, plot_if_enabled):
    """Test overlapping measurements with visualization."""
    key = jax.random.PRNGKey(123)
    mask_state = square_mask.init(key)

    # Two close but different positions
    xi1 = jnp.array([14.0, 14.0])
    xi2 = jnp.array([16.0, 16.0])

    # Sequential measurements
    measurement1 = jnp.ones((28, 28, 1))
    measurement2 = 2 * jnp.ones((28, 28, 1))

    state1 = square_mask.update_measurement(mask_state, measurement1, xi1)
    state2 = square_mask.update_measurement(state1, measurement2, xi2)

    # Test properties
    assert jnp.max(state2.mask_history) >= jnp.max(state1.mask_history)

    # Visualize overlap handling
    mask1 = square_mask.make(xi1)
    mask2 = square_mask.make(xi2)
    overlap = mask1 * mask2

    plot_if_enabled(
        lambda: plot_grid(
            [mask1, mask2, overlap, state1.mask_history, state2.mask_history],
            ["Mask 1", "Mask 2", "Overlap", "After 1st", "After 2nd"],
            figsize=(20, 4),
        )
    )


@pytest.mark.parametrize("mask_size", [5, 10, 15])
def test_different_sizes_functional(mask_size):
    """Quick functional test for different mask sizes."""
    mask = SquareMask(mask_size, (28, 28, 1))
    xi = jnp.array([14.0, 14.0])
    soft_mask = mask.make(xi)

    # Basic properties
    assert soft_mask.shape == (28, 28, 1)
    assert jnp.sum(soft_mask > 0.5) > 0

    # Larger masks should have more coverage
    coverage = jnp.sum(soft_mask > 0.5)
    expected_min_coverage = (mask_size // 2) ** 2
    assert coverage >= expected_min_coverage
