from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from examples.mri.forward_models.base import baseMask, PARAMS_SPIRAL


def generate_spiral_2D(
    num_samples=1000, k_max=1.0, FOV=1.0, angle_offset=0.0, max_angle=None, N=1
):
    theta_max = (2 * jnp.pi / N) * k_max * FOV
    if max_angle is not None:
        theta_max = min(theta_max, max_angle)
    theta = jnp.linspace(0, theta_max, num_samples)

    r = (N * theta) / (2 * jnp.pi * FOV)

    kx = jnp.zeros(num_samples * N)
    ky = jnp.zeros(num_samples * N)

    theta_shifted = theta[:, None] + (2 * jnp.pi * jnp.arange(N) / N) + angle_offset
    kx = (r[:, None] * jnp.cos(theta_shifted)).ravel()
    ky = (r[:, None] * jnp.sin(theta_shifted)).ravel()

    return kx, ky


@partial(jax.vmap, in_axes=(0, 0, None, None, None))
def gaussian_kernel(kx_i, ky_i, x, y, sigma=0.3):
    distances = ((x - kx_i) ** 2 + (y - ky_i) ** 2) / (2 * sigma**2)
    return jnp.exp(-distances)


@partial(jax.jit, static_argnums=(2,))
def grid(kx, ky, size, sigma=0.3, sharpness=10.0):
    y, x = jnp.mgrid[0 : size[0], 0 : size[1]]
    y = y.astype(float)
    x = x.astype(float)

    scale = min(size[0], size[1]) / 2 - 1

    kx_scaled = kx * scale + size[1] / 2
    ky_scaled = ky * scale + size[0] / 2

    grid = gaussian_kernel(kx_scaled, ky_scaled, x, y, sigma).sum(axis=0)

    grid = jax.nn.sigmoid(sharpness * (grid - 1))
    return grid


@dataclass
class maskSpiral(baseMask):
    img_shape: tuple
    task: str
    num_samples: int
    sigma: float
    data_model: str = "kneeFastMRI"
    max_angle: float = None
    

    def init_design(self, key: PRNGKeyArray) -> Array:
        # return jax.random.uniform(key, shape=(3,), minval=PARAMS_SPIRAL[self.data_model]['minval'], maxval=PARAMS_SPIRAL[self.data_model]['maxval'])
        return jax.random.uniform(key, shape=(3,), minval=2., maxval=4.)
    def make(self, xi: Array) -> Array:
        xi  = jax.nn.sigmoid(xi)
        fov = xi[0]
        k_max = xi[1]
        angle_offset = xi[2]

        kx, ky = generate_spiral_2D(
            self.num_samples, k_max, fov, angle_offset, self.max_angle
        )
        grid_mask_soft = grid(kx, ky, self.img_shape, self.sigma)
        grid_mask_hard = grid_mask_soft > .5
        return grid_mask_soft + jax.lax.stop_gradient(grid_mask_hard - grid_mask_soft)