from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array
from typing import Callable


def slice_fourier(mri_slice):
    return jnp.fft.fftshift(jnp.fft.fft2(mri_slice))


def slice_inverse_fourier(fourier_transform):
    return jnp.real(jnp.fft.ifft2(jnp.fft.ifftshift(fourier_transform)))


def generate_spiral_2D(N=1, num_samples=1000, k_max=1.0, FOV=1.0):
    theta_max = (2 * jnp.pi / N) * k_max * FOV
    theta = jnp.linspace(0, theta_max, num_samples)

    r = (N * theta) / (2 * jnp.pi * FOV)

    kx = jnp.zeros(num_samples * N)
    ky = jnp.zeros(num_samples * N)

    theta_shifted = theta[:, None] + (2 * jnp.pi * jnp.arange(N) / N)
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

    grid = jax.nn.sigmoid(sharpness * (grid - 0.1))
    return grid


@dataclass
class maskSpiral:
    img_shape: tuple
    num_spiral: int
    num_samples: int
    k_max: float
    sigma: float

    def make(self, sq_fov: float):
        fov = sq_fov**2
        kx, ky = generate_spiral_2D(self.num_spiral, self.num_samples, self.k_max, fov)
        return grid(kx, ky, self.img_shape, self.sigma)

    def measure_from_mask(self, hist_mask: Array, x: Array):
        fourier_x = hist_mask * slice_fourier(x[..., 0])
        zero_channel = jnp.zeros_like(fourier_x)
        return jnp.stack([fourier_x, zero_channel], axis=-1)
    
    def measure(self, sq_fov: float, x: Array):
        return self.measure_from_mask(self.make(sq_fov), x)
    
    def restore_from_mask(self, hist_mask: Array, x: Array, measured: Array):
        # On crée le masque inverse
        inv_mask = 1 - hist_mask

        # On calcule la transformée de Fourier de l'image
        fourier_x = slice_fourier(x[..., 0])

        # On masque les éléments observés
        masked_inv_fourier_x = inv_mask * fourier_x

        # On retrouve l'image originale
        img = slice_inverse_fourier(masked_inv_fourier_x + measured[..., 0])

        anomaly_map = x[..., 1]

        final = jnp.stack([img, anomaly_map], axis=-1)

        return final

    def restore(self, sq_fov: float, x: Array, measured: Array):
        return self.restore_from_mask(self.make(sq_fov), x, measured)
