import pytest
import jax
import jax.numpy as jnp
from diffuse.images import SquareMask
import matplotlib.pyplot as plt


@pytest.fixture
def random_image():
    key = jax.random.PRNGKey(0)
    return jax.random.normal(key, (28, 28, 1))


@pytest.fixture
def square_mask():
    return SquareMask(10, (28, 28, 1))


def plot_compare(img1, img2):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1, cmap="gray")
    ax2.imshow(img2, cmap="gray")
    plt.show()

def test_measure_and_restore(random_image, square_mask, plot_if_enabled):
    key = jax.random.PRNGKey(1)
    xi = jax.random.uniform(key, (2,), minval=0, maxval=28)

    measured = square_mask.measure(xi, random_image)
    restored = square_mask.restore(xi, random_image, measured)

    #almost equal
    assert jnp.allclose(random_image, restored), plot_if_enabled(lambda: plot_compare(random_image, restored))


def test_restore_with_zero_measured(random_image, square_mask, plot_if_enabled):
    key = jax.random.PRNGKey(1)
    xi = jax.random.uniform(key, (2,), minval=0, maxval=28)

    random_img = jax.random.normal(key, random_image.shape)
    restored = square_mask.restore(xi, random_image, random_img)

    measured_restored = square_mask.measure(xi, restored)
    measured = square_mask.measure(xi, random_img)

    assert jnp.allclose(
        measured_restored, measured, rtol=1e-1, atol=1e-1
    ), plot_if_enabled(lambda: plot_compare(measured_restored, measured))


def test_measure_restore(random_image, square_mask, plot_if_enabled):
    key = jax.random.PRNGKey(1)
    key_x, key_y = jax.random.split(key)
    xi = jax.random.uniform(key, (2,), minval=0, maxval=28)
    x = jax.random.normal(key_x, random_image.shape)
    y = jax.random.normal(key_y, random_image.shape)

    restored = square_mask.restore(xi, x, y)
    measured = square_mask.measure(xi, y)
    measured_restored = square_mask.measure(xi, restored)

    assert jnp.allclose(measured, measured_restored, rtol=1e-1, atol=1e-1), plot_if_enabled(lambda: plot_compare(measured, measured_restored))


def test_mask_shape(square_mask):
    xi = jnp.array([14.0, 14.0])
    mask = square_mask.make(xi)
    assert mask.shape == (28, 28, 1)


def test_mask_values(square_mask):
    xi = jnp.array([14.0, 14.0])
    mask = square_mask.make(xi)
    assert jnp.all(mask >= 0) and jnp.all(mask <= 1)


def test_measure_preserves_shape(random_image, square_mask):
    xi = jnp.array([14.0, 14.0])
    measured = square_mask.measure(xi, random_image)
    assert measured.shape == random_image.shape


def test_restore_preserves_shape(random_image, square_mask):
    xi = jnp.array([14.0, 14.0])
    measured = square_mask.measure(xi, random_image)
    restored = square_mask.restore(xi, random_image, measured)
    assert restored.shape == random_image.shape
