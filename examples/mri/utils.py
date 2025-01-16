from dataclasses import dataclass
from functools import partial

import einops
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import MeasurementState
from diffuse.utils.plotting import sigle_plot

def slice_fourier(mri_slice):
    f = jnp.fft.fftshift(jnp.fft.fft2(mri_slice, norm="ortho"))
    return jnp.stack([jnp.real(f), jnp.imag(f)], axis=-1)


def slice_inverse_fourier(fourier_transform):
    fourier_transform = fourier_transform[..., 0] + 1j * fourier_transform[..., 1]
    return jnp.real(jnp.fft.ifft2(jnp.fft.ifftshift(fourier_transform), norm="ortho"))


def generate_spiral_2D(N=1, num_samples=1000, k_max=1.0, FOV=1.0, angle_offset=0.0):
    theta_max = (2 * jnp.pi / N) * k_max * FOV
    theta = jnp.linspace(0, theta_max, num_samples)

    r = (N * theta) / (2 * jnp.pi * FOV)

    kx = jnp.zeros(num_samples * N)
    ky = jnp.zeros(num_samples * N)

    theta_shifted = theta[:, None] + (2 * jnp.pi * jnp.arange(N) / N) + angle_offset
    kx = (r[:, None] * jnp.cos(theta_shifted)).ravel()
    ky = (r[:, None] * jnp.sin(theta_shifted)).ravel()

    return kx, ky


def generate_radial_2D(N=1, num_samples=1000, angle_offset: float = 0.0):
    """Generate radial k-space sampling pattern."""
    # Convert angle offset to radians
    angle_offset_rad = jnp.deg2rad(angle_offset)

    # Generate angles evenly spaced from 0 to π, excluding π
    angles = jnp.linspace(0, jnp.pi, N + 1)[:-1] + angle_offset_rad

    # Generate base radial points
    r = jnp.linspace(
        -jnp.sqrt(2), jnp.sqrt(2), num_samples
    )  # Scale by √2 to ensure reaching corners

    # Create meshgrid of r and angles
    r_mesh, theta_mesh = jnp.meshgrid(r, angles)

    # Convert to Cartesian coordinates for original angles
    kx = (r_mesh * jnp.cos(theta_mesh)).ravel()
    ky = (r_mesh * jnp.sin(theta_mesh)).ravel()

    # Add perpendicular lines
    theta_mesh_perp = theta_mesh + jnp.pi / 2
    kx_perp = (r_mesh * jnp.cos(theta_mesh_perp)).ravel()
    ky_perp = (r_mesh * jnp.sin(theta_mesh_perp)).ravel()

    # Concatenate original and perpendicular lines
    kx = jnp.concatenate([kx, kx_perp])
    ky = jnp.concatenate([ky, ky_perp])

    return kx, ky


@partial(jax.vmap, in_axes=(0, 0, None, None, None))
def gaussian_kernel(kx_i, ky_i, x, y, sigma=0.3):
    distances = ((x - kx_i) ** 2 + (y - ky_i) ** 2) / (2 * sigma**2)
    return jnp.exp(-distances)


@partial(jax.jit, static_argnums=(2,))
def grid(kx, ky, size, sigma=0.3, sharpness=400.0):
    y, x = jnp.mgrid[0 : size[0], 0 : size[1]]
    y = y.astype(float)
    x = x.astype(float)

    scale = min(size[0], size[1]) / 2 - 1

    kx_scaled = kx * scale + size[1] / 2
    ky_scaled = ky * scale + size[0] / 2

    grid = gaussian_kernel(kx_scaled, ky_scaled, x, y, sigma).sum(axis=0)

    grid = jax.nn.sigmoid(sharpness * (grid - 1))
    return grid


class baseMask:
    img_shape: tuple
    num_samples: int
    sigma: float
    sigma_prob: float = 1.0

    def measure_from_mask(self, hist_mask: Array, x: Array):
        return hist_mask[..., None] * slice_fourier(x[..., 0])

    def measure(self, xi: float, x: Array):
        return self.measure_from_mask(self.make(xi), x)

    def restore_from_mask(self, hist_mask: Array, x: Array, measured: Array):
        # On crée le masque inverse
        inv_mask = 1 - hist_mask

        # On calcule la transformée de Fourier de l'image
        fourier_x = slice_fourier(x[..., 0])

        # On masque les éléments observés
        # masked_inv_fourier_x = inv_mask * fourier_x
        masked_inv_fourier_x = jnp.einsum("ij,ijk->ijk", inv_mask, fourier_x)

        # On retrouve l'image originale
        img = slice_inverse_fourier(masked_inv_fourier_x + measured)

        anomaly_map = jnp.real(x[..., 1])

        final = jnp.stack([img, anomaly_map], axis=-1)

        return final

    def restore(self, xi: float, x: Array, measured: Array):
        return self.restore_from_mask(self.make(xi), x, measured)

    def init_measurement(self) -> MeasurementState:
        y = jnp.zeros(self.img_shape)
        mask_history = jnp.zeros_like(self.make(jnp.array([0.0, 0.0])))
        return MeasurementState(y=y, mask_history=mask_history)

    def supp_mask(self, xi: float, hist_mask: Array, new_mask: Array):
        inv_mask = 1 - self.make(xi)
        return hist_mask * inv_mask + new_mask

    def update_measurement(
        self, measurement_state: MeasurementState, new_measurement: Array, design: Array
    ) -> MeasurementState:
        hist_mask, y = measurement_state.mask_history, measurement_state.y
        joint_y = hist_mask[..., None] * y + new_measurement
        mask_history = self.supp_mask(
            design, measurement_state.mask_history, self.make(design)
        )
        return MeasurementState(y=joint_y, mask_history=mask_history)

    def logprob_y(self, theta: Array, y: Array, design: Array) -> Array:
        f_y = self.measure(design, theta)
        f_y_flat = einops.rearrange(f_y, "... h w c -> ... (h w c)")
        y_flat = einops.rearrange(y, "... h w c -> ... (h w c)")

        return jax.scipy.stats.multivariate_normal.logpdf(
            y_flat, mean=f_y_flat, cov=self.sigma_prob**2
        )

    def grad_logprob_y(self, theta: Array, y: Array, design: Array) -> Array:
        meas_x = self.measure_from_mask(design, theta)
        return (
            self.restore_from_mask(design, jnp.zeros_like(theta), (y - meas_x))
            / self.sigma_prob
        )


@dataclass
class maskSpiral(baseMask):
    num_spiral: int
    img_shape: tuple
    num_samples: int
    sigma: float

    def init_design(self, key: PRNGKeyArray) -> Array:
        return jnp.array([2, 1.0])
        return jax.random.uniform(key, shape=(3,), minval=0.0, maxval=1.0)

    def make(self, xi: float):
        fov = xi[0] ** 2
        k_max = xi[1]
        angle_offset = xi[2]

        kx, ky = generate_spiral_2D(
            self.num_spiral, self.num_samples, k_max, fov, angle_offset
        )
        return grid(kx, ky, self.img_shape, self.sigma)


@dataclass
class maskRadial(baseMask):
    num_lines: int
    img_shape: tuple
    num_samples: int
    sigma: float

    def make(self, xi: float):
        angle_offset = xi[0]

        kx, ky = generate_radial_2D(self.num_lines, self.num_samples, angle_offset)
        return grid(kx, ky, self.img_shape, self.sigma)


@partial(jax.vmap, in_axes=(0, 0))
def log_complex_normal_pdf(z: Array, mu: Array) -> Array:
    diff = z - mu

    # Log PDF = -n*log(pi) - |z-mu|^2
    n = jnp.size(z)
    log_norm_const = -n * jnp.log(jnp.pi)
    log_exp_term = -(jnp.abs(diff) ** 2)

    return jnp.real(log_norm_const + log_exp_term)
