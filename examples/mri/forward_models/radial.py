from examples.mri.forward_models.base import (
    baseMask,
    PARAMS_SIZE_LINE,
    PARAMS_SIGMA_RADIAL,
)
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


@partial(jax.vmap, in_axes=(0, 0, None, None, None))
def generate_line_soft(angle, size, shape, width, sigma):
    H, W = shape

    xs = jnp.linspace(-W / 2, W / 2, W)
    ys = jnp.linspace(-H / 2, H / 2, H)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")

    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    x_rot = grid_x * cos_a + grid_y * sin_a
    y_rot = -grid_x * sin_a + grid_y * cos_a

    sigma = 0.1
    mask_x = jax.nn.sigmoid((size / 2 - jnp.abs(x_rot)) / sigma)
    mask_y = jax.nn.sigmoid((width / 2 - jnp.abs(y_rot)) / sigma)

    mask = mask_x * mask_y
    return mask


@partial(jax.vmap, in_axes=(0, 0, None))
def generate_line_hard(angle, size, shape):
    H, W = shape

    xs = jnp.linspace(-W / 2, W / 2, W)
    ys = jnp.linspace(-H / 2, H / 2, H)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")

    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    x_rot = grid_x * cos_a + grid_y * sin_a
    y_rot = -grid_x * sin_a + grid_y * cos_a

    # Use a small threshold instead of exact equality
    epsilon = 0.5  # Half pixel width
    mask = (jnp.abs(x_rot) <= size / 2) & (jnp.abs(y_rot) <= epsilon)
    return mask.astype(jnp.float32)


def generate_centered_circle(shape, radius_frac):
    H, W = shape
    # Calculate radius based on area percentage
    total_area = H * W
    circle_area = total_area * radius_frac
    radius = jnp.sqrt(circle_area / jnp.pi)

    # Create grid
    xs = jnp.linspace(-W / 2, W / 2, W)
    ys = jnp.linspace(-H / 2, H / 2, H)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")

    # Create circle mask
    mask = (grid_x**2 + grid_y**2) <= radius**2
    return mask.astype(jnp.float32)


@dataclass
class maskRadial(baseMask):
    num_lines: int
    img_shape: tuple  # (H, W, C)
    task: str
    data_model: str

    def init_design(self, key: PRNGKeyArray) -> Array:
        angles = jax.random.uniform(
            key, shape=(self.num_lines,), minval=0.0, maxval=2 * jnp.pi
        )

        size_line = jax.random.uniform(
            key, shape=(self.num_lines,), **PARAMS_SIZE_LINE[self.data_model]
        )

        return jnp.stack([angles, size_line], axis=-1)

    # def init_measurement(self, ground_truth: Array) -> MeasurementState:
    # mask = generate_centered_circle(self.img_shape[:-1], 0.01)
    # y = self.measure_from_mask(mask, ground_truth)
    # return MeasurementState(y=y, mask_history=mask)

    def make(self, xi: Array) -> Array:
        # xi = jax.nn.softplus(xi)
        angle_rad = xi[:, 0]
        size_line = xi[:, 1]

        lines_soft = generate_line_soft(
            angle_rad,
            size_line,
            self.img_shape[:-1],
            1,
            PARAMS_SIGMA_RADIAL[self.data_model],
        )
        lines_hard = generate_line_hard(angle_rad, size_line, self.img_shape[:-1])
        lines = lines_soft + jax.lax.stop_gradient(lines_hard - lines_soft)

        hist_lines = jnp.zeros(self.img_shape[:-1])
        for line in lines:
            inv_hist_lines = 1 - hist_lines
            hist_lines = inv_hist_lines * line + hist_lines

        return hist_lines
