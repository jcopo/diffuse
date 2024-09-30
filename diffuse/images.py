import jax
import jax.numpy as jnp
import pdb
import matplotlib.pyplot as plt
import einops
from diffuse.unet import UNet
from diffuse.score_matching import score_match_loss
from diffuse.sde import SDE, LinearSchedule
from functools import partial
import numpy as np
import optax
from tqdm import tqdm
from dataclasses import dataclass
from jaxtyping import PyTreeDef, PRNGKeyArray, Array


@dataclass
class SquareMask:
    size: int
    img_shape: tuple

    def make(self, xi: Array) -> Array:
        """Create a differentiable square mask."""
        height, width, *_ = self.img_shape
        y, x = jnp.mgrid[:height, :width]

        # Calculate distances from the center
        y_dist = jnp.abs(y - xi[1])
        x_dist = jnp.abs(x - xi[0])

        # Create a soft mask using sigmoid function
        mask_half_size = self.size // 2
        softness = 0.1  # Adjust this value to control the softness of the edges

        mask = jax.nn.sigmoid(
            (-jnp.maximum(y_dist, x_dist) + mask_half_size) / softness
        )
        # return jnp.where(mask > 0.5, 1.0, 0.0)[..., None]
        return mask[..., None]

    def measure_from_mask(self, hist_mask: Array, img: Array):
        return img * hist_mask

    def restore_from_mask(self, hist_mask: Array, img: Array, measured: Array):
        return img * hist_mask + measured

    def measure(self, xi: Array, img: Array):
        return self.measure_from_mask(self.make(xi), img)

    def restore(self, xi: Array, img: Array, measured: Array):
        inv_mask = 1 - self.make(xi)
        return self.restore_from_mask(inv_mask, img, measured)


if __name__ == "__main__":
    data = jnp.load("dataset/mnist.npz")
    xs = data["X"]
    xs = einops.rearrange(xs, "b h w -> b h w 1")

    x = xs[0]
    # x = jax.random.normal(jax.random.PRNGKey(0), x.shape)

    mask = SquareMask(10, x.shape)
    xi = jnp.array([15.0, 15.0])
    xi2 = jnp.array([20.0, 10.0])

    mask_history = mask.restore(xi2, mask.make(xi), mask.make(xi2))
    print(jnp.max(mask_history))
    plt.imshow(mask_history, cmap="gray")
    plt.scatter(xi[0], xi[1], color="red")
    plt.scatter(xi2[0], xi2[1], color="red")
    plt.show()

    # Create a figure with 2 row and 3 columns
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))

    # Plot the first image
    ax1.scatter(xi[0], xi[1], color="red")
    im1 = ax1.imshow(mask.make(xi), cmap="gray")
    ax1.set_title("Measured")
    ax1.axis("off")

    # Plot the second image
    im2 = ax2.imshow(x, cmap="gray")
    ax2.set_title("Original")
    ax2.axis("off")

    # Plot the third image
    measured = mask.measure(xi, x)
    im3 = ax3.imshow(measured, cmap="gray")
    ax3.set_title("Masked")
    ax3.axis("off")

    # plot inverse mask
    inv_mask = 1 - mask.make(xi)
    im4 = ax4.imshow(inv_mask, cmap="gray")
    ax4.set_title("Inverse Mask")
    ax4.axis("off")

    # plot restored image
    restored = mask.restore(xi, x, 0.0 * measured)
    im5 = ax5.imshow(restored, cmap="gray")
    ax5.set_title("Restored")
    ax5.axis("off")

    def norm_measure(xi: Array, img: Array, mask: SquareMask):
        return (mask.measure(xi, img) ** 2).sum()

    # plot the norm of the measure
    im6 = ax6.imshow(x, cmap="gray")
    ax6.set_title("Norm of the measure")
    ax6.axis("off")
    print(measured)

    plt.show()

    val, grad = jax.value_and_grad(norm_measure)(xi, x.squeeze(), mask)
    print(grad)
    print(val)
    # plot grad vector on top of the image
    plt.imshow(x, cmap="gray")
    plt.arrow(xi[1], xi[0], grad[1], grad[0], color="red", head_width=0.5, zorder=1)
    plt.show()
