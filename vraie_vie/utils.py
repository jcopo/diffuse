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


@jax.custom_vjp
def _make(w: Array, s: int, shape: tuple, key: PRNGKeyArray):
    normalized_vector = w / w.sum()
    uniform_vector = jax.random.uniform(
        key, shape=(92, 112), minval=0, maxval=1
    )  # 💀 trop laid mais problème avec static
    return jnp.where(s * normalized_vector < uniform_vector, 1, 0)


def make_fwd(w: Array, s: int, shape: tuple, key: PRNGKeyArray):
    normalized_vector = w / w.sum()
    uniform_vector = jax.random.uniform(
        key, shape=(92, 112), minval=0, maxval=1
    )  # 💀 trop laid mais problème avec static
    output = jnp.where(s * normalized_vector < uniform_vector, 1, 0)
    return output, (w, s, shape, key)


def make_bwd(_, grad_output):
    return (grad_output, None, None, None)


_make.defvjp(make_fwd, make_bwd)


@dataclass
class maskFourier:
    s: int
    img_shape: tuple

    def make(self, w: Array, key: PRNGKeyArray):
        return _make(w, self.s, self.img_shape, key)

    def measure(self, w: Array, x: Array, key: PRNGKeyArray):
        mask = self.make(w, key)
        fourier_x = mask * slice_fourier(x[..., 0])
        zero_channel = jnp.zeros_like(fourier_x)
        return jnp.stack([fourier_x, zero_channel], axis=-1)

    def restore(self, w: Array, x: Array, measured: Array, key: PRNGKeyArray):
        # On crée le masque inverse
        inv_mask = 1 - self.make(w, key)

        # On calcule la transformée de Fourier de l'image
        fourier_x = slice_fourier(x[..., 0])

        # On masque les éléments observés
        masked_inv_fourier_x = inv_mask * fourier_x

        # On retrouve l'image originale
        img = slice_inverse_fourier(masked_inv_fourier_x + measured[..., 0])

        anomaly_map = x[..., 1]

        final = jnp.stack([img, anomaly_map], axis=-1)

        return final


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
def grid(kx, ky, size, sigma=0.2, sharpness=10.0):
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

    def make(self, fov: float):
        kx, ky = generate_spiral_2D(self.num_spiral, self.num_samples, self.k_max, fov)
        return grid(kx, ky, self.img_shape)

    def measure(self, fov: float, x: Array):
        mask = self.make(fov)
        fourier_x = mask * slice_fourier(x[..., 0])
        zero_channel = jnp.zeros_like(fourier_x)
        return jnp.stack([fourier_x, zero_channel], axis=-1)

    def restore(self, fov: float, x: Array, measured: Array):
        # On crée le masque inverse
        inv_mask = 1 - self.make(fov)

        # On calcule la transformée de Fourier de l'image
        fourier_x = slice_fourier(x[..., 0])

        # On masque les éléments observés
        masked_inv_fourier_x = inv_mask * fourier_x

        # On retrouve l'image originale
        img = slice_inverse_fourier(masked_inv_fourier_x + measured[..., 0])

        anomaly_map = x[..., 1]

        final = jnp.stack([img, anomaly_map], axis=-1)

        return final
