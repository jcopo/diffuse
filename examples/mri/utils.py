from dataclasses import dataclass
from functools import partial

import einops
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import MeasurementState


def slice_fourier(mri_slice, task):
    # TODO: change when re-training of WMH, just keep first if
    if task == "quality":
        mri_slice = mri_slice[..., 0] + 1j * mri_slice[..., 1]

    else:
        mri_slice = mri_slice[..., 0]

    f = jnp.fft.fftshift(jnp.fft.fft2(mri_slice, norm="ortho", axes=[-2, -1]), axes=[-2, -1])
    return jnp.stack([jnp.real(f), jnp.imag(f)], axis=-1)


def slice_inverse_fourier(fourier_transform, task):
    fourier_transform_complex = fourier_transform[..., 0] + 1j * fourier_transform[..., 1]
    inv_fourier_slice = jnp.fft.ifft2(jnp.fft.ifftshift(fourier_transform_complex, axes=[-2, -1]), norm="ortho", axes=[-2, -1])
    if task == "quality":
        return jnp.stack([jnp.real(inv_fourier_slice), jnp.imag(inv_fourier_slice)], axis=-1)
    elif task == "anomaly":
        return jnp.real(inv_fourier_slice)

def generate_spiral_2D(
    N=1, num_samples=1000, k_max=1.0, FOV=1.0, angle_offset=0.0, max_angle=None
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

    y_circle, x_circle = jnp.ogrid[-center_y:img_shape[0]-center_y, -center_x:img_shape[1]-center_x]
    circle_mask = jax.nn.sigmoid(-sharpness * ((x_circle*x_circle + y_circle*y_circle) - size_line ** 2))
    
    line_image = line_image * circle_mask
    return line_image


class baseMask:
    img_shape: tuple
    task: str
    num_samples: int = 1000
    sigma: float = 0.3
    sigma_prob: float = 1.0

    def make(self, xi: Array) -> Array:
        pass

    def measure_from_mask(self, hist_mask: Array, x: Array) -> Array:
        return einops.einsum(hist_mask, slice_fourier(x, self.task), "h w, ... h w c -> ... h w c")

    def measure(self, xi: float, x: Array):
        return self.measure_from_mask(self.make(xi), x)

    def restore_from_mask(self, hist_mask: Array, x: Array, measured: Array):
        masked_inv_fourier_x = self.measure_from_mask(1 - hist_mask, x)

        # On retrouve l'image originale
        img = slice_inverse_fourier(masked_inv_fourier_x + measured, self.task)
        if self.task == "anomaly":
            anomaly_map = jnp.real(x[..., -1])
            final = jnp.stack([img, anomaly_map], axis=-1)
        else:
            final = img

        return final

    def restore(self, xi: Array, x: Array, measured: Array) -> Array:
        return self.restore_from_mask(self.make(xi), x, measured)

    def init_measurement(self, xi_init: Array) -> MeasurementState:
        y = jnp.zeros(self.img_shape)
        mask_history = jnp.zeros_like(self.make(xi_init))
        return MeasurementState(y=y, mask_history=mask_history)

    def supp_mask(self, xi: Array, hist_mask: Array, new_mask: Array) -> Array:
        inv_mask = 1 - self.make(xi)
        return hist_mask * inv_mask + new_mask

    def update_measurement(
        self, measurement_state: MeasurementState, new_measurement: Array, design: Array
    ) -> MeasurementState:
        y = measurement_state.y
        inv_mask = 1 - self.make(design)
        # joint_y = inv_mask[..., None] * y + new_measurement
        joint_y = jnp.einsum("ij,ijk->ijk", inv_mask, y) + new_measurement
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
    task: str
    num_samples: int
    sigma: float
    max_angle: float = None

    def init_design(self, key: PRNGKeyArray) -> Array:
        return jax.random.uniform(key, shape=(3,), minval=0.0, maxval=3.0)

    def make(self, xi: Array) -> Array:
        xi **= 2
        fov = xi[0]
        k_max = xi[1]
        angle_offset = xi[2]

        kx, ky = generate_spiral_2D(
            self.num_spiral, self.num_samples, k_max, fov, angle_offset, self.max_angle
        )
        return grid(kx, ky, self.img_shape, self.sigma)

@dataclass
class maskRadial(baseMask):
    num_lines: int
    img_shape: tuple
    task: str
    def init_design(self, key: PRNGKeyArray) -> Array:
        angles = jax.random.uniform(key, shape=(self.num_lines,), minval=0.0, maxval=2*jnp.pi)
        size_line = jax.random.uniform(key, shape=(self.num_lines,), minval=0.0, maxval=10)
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