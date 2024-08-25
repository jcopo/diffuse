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

data = jnp.load("dataset/mnist.npz")
xs = data["X"]
xs = einops.rearrange(xs, "b h w -> b h w 1")


x = xs[0]


@dataclass
class SquareMask:
    size: int
    img_shape: tuple

    def get(self, img: Array, xi: Array):
        # return part of the image that is within the square mask left side corner at xi
        x = int(jnp.clip(xi[0], 0, img.shape[0] - self.size))
        y = int(jnp.clip(xi[1], 0, img.shape[1] - self.size))
        return img[x : x + self.size, y : y + self.size]

    def make(self, xi: Array) -> Array:
        """Create a differentiable square mask."""
        height, width, *_ = self.img_shape
        y, x = jnp.mgrid[:height, :width]

        # Calculate distances from the center
        y_dist = jnp.abs(y - xi[0])
        x_dist = jnp.abs(x - xi[1])

        # Create a soft mask using sigmoid function
        mask_half_size = self.size // 2
        softness = 0.1  # Adjust this value to control the softness of the edges

        mask = jax.nn.sigmoid(
            (-jnp.maximum(y_dist, x_dist) + mask_half_size) / softness
        )
        return mask[..., None]


def measure(xi: Array, img: Array, mask: SquareMask):
    return img * mask.make(xi)


def restore(xi: Array, img: Array, mask: SquareMask, measured: Array):
    inv_mask = 1 - mask.make(xi)
    return img * inv_mask + measured


def norm_measure(xi: Array, img: Array, mask: SquareMask):
    return (measure(xi, img, mask) ** 2).sum()


if __name__ == "__main__":
    mask = SquareMask(10, x.shape)
    xi = jnp.array([10.0, 20.0])
    # Create a figure with 2 row and 3 columns
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))

    # Plot the first image
    im1 = ax1.imshow(mask.make(xi), cmap="gray")
    ax1.set_title("Measured")
    ax1.axis("off")

    # Plot the second image
    im2 = ax2.imshow(x, cmap="gray")
    ax2.set_title("Original")
    ax2.axis("off")

    # Plot the third image
    measured = measure(xi, x.squeeze(), mask)
    im3 = ax3.imshow(measured, cmap="gray")
    ax3.set_title("Masked")
    ax3.axis("off")

    # plot inverse mask
    inv_mask = 1 - mask.make(xi)
    im4 = ax4.imshow(inv_mask, cmap="gray")
    ax4.set_title("Inverse Mask")
    ax4.axis("off")

    # plot restored image
    restored = restore(xi, x.squeeze(), mask, 0.0 * measured)
    im5 = ax5.imshow(restored, cmap="gray")
    ax5.set_title("Restored")
    ax5.axis("off")

    # plot the norm of the measure
    val = norm_measure(xi, x.squeeze(), mask)
    im6 = ax6.imshow(x, cmap="gray")
    ax6.set_title("Norm of the measure")
    ax6.axis("off")

    plt.show()

    val, grad = jax.value_and_grad(norm_measure)(xi, x.squeeze(), mask)
    print(grad)
    print(val)
    # plot grad vector on top of the image
    plt.imshow(x, cmap="gray")
    plt.arrow(xi[1], xi[0], grad[1], grad[0], color="red", head_width=0.5, zorder=1)
    plt.show()
