from examples.mri.forward_models.base import baseMask
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


@partial(jax.vmap, in_axes=(0, 0, None))
def generate_line(angle_rad, size_line, img_shape):
    y, x = jnp.mgrid[: img_shape[0], : img_shape[1]]

    center_x = img_shape[1] // 2
    center_y = img_shape[0] // 2

    x = x - center_x
    y = y - center_y

    distance = jnp.abs(x * jnp.cos(angle_rad) + y * jnp.sin(angle_rad))

    sharpness = 300.0
    line_image = jax.nn.sigmoid(-sharpness * (distance - 0.5))

    y_circle, x_circle = jnp.ogrid[
        -center_y : img_shape[0] - center_y, -center_x : img_shape[1] - center_x
    ]
    circle_mask = jax.nn.sigmoid(
        -sharpness * ((x_circle * x_circle + y_circle * y_circle) - size_line**2)
    )

    line_image = line_image * circle_mask
    return line_image


@dataclass
class maskRadial(baseMask):
    num_lines: int
    img_shape: tuple # (H, W, C)
    task: str

    def init_design(self, key: PRNGKeyArray) -> Array:
        angles = jax.random.uniform(
            key, shape=(self.num_lines,), minval=0.0, maxval=2 * jnp.pi
        )
        size_line = jax.random.uniform(
            key, shape=(self.num_lines,), minval=0.0, maxval=10
        )
        return jnp.stack([angles, size_line], axis=-1)

    def make(self, xi: Array) -> Array:
        angle_rad = xi[:, 0] ** 2
        size_line = xi[:, 1] ** 2
        lines = generate_line(angle_rad, size_line, self.img_shape[:-1])

        hist_lines = jnp.zeros(self.img_shape[:-1])
        for line in lines:
            inv_hist_lines = 1 - hist_lines
            hist_lines = inv_hist_lines * line + hist_lines

        return hist_lines
