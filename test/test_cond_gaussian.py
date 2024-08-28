import jax
import jax.numpy as jnp
from functools import partial
from diffuse.conditional import CondSDE, generate_cond_sample
from diffuse.images import measure, restore
from diffuse.sde import LinearSchedule
import matplotlib.pyplot as plt
import numpy as np

# Force JAX to use CPU
jax.config.update("jax_platform_name", "cpu")


def gaussian_kernel(size, sigma=1.0):
    """Generate a 2D Gaussian kernel."""
    x = jnp.arange(0, size, 1, float)
    y = x[:, jnp.newaxis]
    x0, y0 = size // 2, size // 2
    return jnp.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def score_network(x, t):
    """Simple score network using Gaussian kernel convolution."""
    kernel = gaussian_kernel(5, sigma=1.0)
    return jax.scipy.signal.convolve2d(x, kernel, mode="same")


class Mask:
    def __init__(self, mask_function):
        self.mask_function = mask_function

    def make(self, x):
        return self.mask_function(x)


def half_image_mask(x):
    """Create a mask that obscures the bottom half of the image."""
    h, w = x.shape
    return jnp.vstack([jnp.ones((h // 2, w)), jnp.zeros((h // 2, w))])


def main():
    # # Create a simple image
    # image_size = 64
    # x = jnp.zeros((image_size, image_size))
    # x = x.at[image_size//4:3*image_size//4, image_size//4:3*image_size//4].set(1.0)

    # Load MNIST dataset
    data = np.load("dataset/mnist.npz")
    xs = jnp.array(data["X"])
    # xs = jnp.array(data["X"])[:, 10:18, 10:18]  # Select 8x8 pixels from the center
    x = xs[0]

    # Create Mask object
    mask = Mask(half_image_mask)

    # Create LinearSchedule for beta
    beta = LinearSchedule(b_min=0.1, b_max=20.0, t0=0.0, T=1.0)

    # Create CondSDE
    cond_sde = CondSDE(mask=mask, tf=1.0, score=score_network, beta=beta)

    # Create observation by applying mask and adding noise
    y = measure(jnp.zeros_like(x), x, mask)
    noise = 0.1 * jax.random.normal(jax.random.PRNGKey(0), y.shape)
    y_noisy = y  # + noise

    # Set up PRNG key
    key = jax.random.PRNGKey(42)

    # Run generate_cond_sample
    n_steps = 100
    n_particles = 10
    x_shape = x.shape

    print("Starting conditional sampling...")
    end_state, hist = generate_cond_sample(
        y_noisy, jnp.zeros_like(x), key, n_steps, cond_sde, x_shape
    )

    print("Sampling completed.")
    print("Final state shape:", end_state[0].shape)
    print("History length:", len(hist[0]))

    # Visualize results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(x, cmap="gray")
    axs[0].set_title("Original Image")
    axs[1].imshow(y_noisy, cmap="gray")
    axs[1].set_title("Noisy Observation")
    axs[2].imshow(end_state[0][0], cmap="gray")  # Show the first particle
    axs[2].set_title("Reconstructed Image")
    plt.tight_layout()
    plt.savefig("conditional_sampling_results.png")
    print("Results saved as 'conditional_sampling_results.png'")


if __name__ == "__main__":
    main()
