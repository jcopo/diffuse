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
x = jax.random.normal(jax.random.PRNGKey(0), x.shape)


@dataclass
class SquareMask:
    size: int
    img_shape: tuple

    def get(self, img: Array, xi: Array):
        # Use jnp.floor instead of int for JAX compatibility
        x = jnp.floor(jnp.clip(xi[0] - self.size // 2, 0, self.img_shape[0] - self.size)).astype(jnp.int32)
        y = jnp.floor(jnp.clip(xi[1] - self.size // 2, 0, self.img_shape[1] - self.size)).astype(jnp.int32)
        return jax.lax.dynamic_slice(img, (x, y, 0), (self.size, self.size, img.shape[-1]))

def restore(xi: Array, img: Array, mask: SquareMask, measured: Array):
    x = jnp.floor(jnp.clip(xi[0] - mask.size // 2, 0, img.shape[0] - mask.size)).astype(jnp.int32)
    y = jnp.floor(jnp.clip(xi[1] - mask.size // 2, 0, img.shape[1] - mask.size)).astype(jnp.int32)
    return jax.lax.dynamic_update_slice(img, measured, (x, y, 0))

def measure(xi: Array, img: Array, mask: SquareMask):
    return mask.get(img, xi)

if __name__ == "__main__":
    mask = SquareMask(10, x.shape)
    xi = jnp.array([10.0, 20.0])
    # Create a figure with 2 row and 3 columns
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))

    # Plot the first image
    measured = measure(xi, x, mask)
    im1 = ax1.imshow(measured, cmap="gray")
    ax1.set_title("Measured")
    ax1.axis("off")

    # Plot the second image
    im2 = ax2.imshow(x, cmap="gray")
    ax2.set_title("Original")
    ax2.axis("off")

    # Plot the third image
    im3 = ax3.imshow(x, cmap="gray")
    ax3.add_patch(plt.Rectangle((xi[1] - mask.size // 2, xi[0] - mask.size // 2), 
                                mask.size, mask.size, fill=False, edgecolor='red'))
    ax3.set_title("Original with Measurement Area")
    ax3.axis("off")

    # Remove the inverse mask plot
    ax4.axis("off")

    # plot restored image
    restored = restore(xi, x, mask, 0.5 * jnp.ones_like(measured))
    im5 = ax5.imshow(restored, cmap="gray")
    ax5.set_title("Restored")
    ax5.axis("off")

    def norm_measure(xi: Array, img: Array, mask: SquareMask):
        return jnp.sum(measure(xi, img, mask) ** 2)

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
