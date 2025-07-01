import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import einops
from dataclasses import dataclass
from jaxtyping import Array, PRNGKeyArray
from diffuse.base_forward_model import MeasurementState
from diffuse.utils.plotting import sigle_plot


@dataclass
class MaskState:
    y: Array
    mask_history: Array
    xi: Array


@dataclass
class SquareMask:
    size: int
    img_shape: tuple
    std: float = 1.0

    def make(self, xi: Array) -> Array:
        """Create a differentiable square mask."""
        # assert xi is a 2D array
        assert xi.shape[0] == 2
        height, width, *_ = self.img_shape
        y, x = jnp.mgrid[:height, :width]

        # Calculate distances from the center
        y_dist = jnp.abs(y - xi[1])
        x_dist = jnp.abs(x - xi[0])

        # Create a soft mask using sigmoid function
        mask_half_size = self.size // 2
        softness = 0.1  # Adjust this value to control the softness of the edges

        mask = jax.nn.sigmoid((-jnp.maximum(y_dist, x_dist) + mask_half_size) / softness)
        # return jnp.where(mask > 0.5, 1.0, 0.0)[..., None]
        return mask[..., None]

    def apply(self, img: Array, measurement_state: MeasurementState):
        hist_mask = measurement_state.mask_history
        return img * hist_mask

    def measure(self, rng_key: PRNGKeyArray, img: Array, xi: Array):
        return img * self.make(xi)

    def restore(self, img: Array, measurement_state: MeasurementState):
        mask = measurement_state.mask_history
        inv_mask = 1 - mask
        return img * inv_mask

    def init_design(self, rng_key: PRNGKeyArray) -> Array:
        return jax.random.uniform(rng_key, (2,), minval=0, maxval=28)

    def init(self, rng_key: PRNGKeyArray) -> Array:
        xi = jax.random.uniform(rng_key, (2,), minval=0, maxval=28)
        y = jnp.zeros(self.img_shape)
        mask_history = jnp.zeros_like(self.make(xi))
        return MaskState(y=y, mask_history=mask_history, xi=xi)

    def update_measurement(self, mask_state: MaskState, new_measurement: Array, design: Array) -> MaskState:
        new_mask = self.make(design)
        # Compute the new part of the mask (i.e. the part not already measured)
        new_part_mask = new_mask * (1 - mask_state.mask_history)
        # Superpose the new measurement only on the new part
        joint_y = mask_state.y + new_measurement * new_part_mask
        # Update the mask history by adding the new part
        mask_history = mask_state.mask_history + new_part_mask
        return MaskState(y=joint_y, mask_history=mask_history, xi=design)

    # def logprob_y(self, theta: Array, y: Array, design: Array) -> Array:
    #     f_y = self.measure(design, theta)
    #     # Preserve batch dimension while flattening spatial dimensions
    #     # If input is (batch, 28, 28, 1), output will be (batch, 784)
    #     f_y_flat = einops.rearrange(f_y, "... h w c -> ... (h w c)")
    #     y_flat = einops.rearrange(y, "... h w c -> ... (h w c)")

    #     #import pdb; pdb.set_trace()
    #     return jax.scipy.stats.multivariate_normal.logpdf(
    #         y_flat, mean=f_y_flat, cov=self.sigma**2
    #     )  # returns shape (batch,)

    # def grad_logprob_y(self, theta: Array, y: Array, design: Array) -> Array:
    #     meas_x = self.measure_from_mask(design, theta)
    #     #jax.experimental.io_callback(sigle_plot, None, y)
    #     # jax.experimental.io_callback(sigle_plot, None, meas_x)
    #     return self.restore_from_mask(design, jnp.zeros_like(theta), (y - meas_x)) / self.sigma

    # def logprob_y_t(self, theta: Array, y: Array, mask: Array, alpha_t: float) -> Array:
    #     A_theta = self.measure_from_mask(mask, theta)
    #     logsprobs = jax.scipy.stats.norm.logpdf(y, A_theta, alpha_t)
    #     logsprobs = einops.reduce(logsprobs, "t ... -> t ", "sum")
    #     return logsprobs


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
