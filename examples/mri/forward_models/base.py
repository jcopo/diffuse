import einops
import jax
import jax.numpy as jnp
from jaxtyping import Array

from diffuse.base_forward_model import MeasurementState


def slice_fourier(mri_slice):
    mri_slice = mri_slice[..., 0] + 1j * mri_slice[..., 1]
    f = jnp.fft.fftshift(
        jnp.fft.fft2(mri_slice, norm="ortho", axes=[-2, -1]), axes=[-2, -1]
    )
    return jnp.stack([jnp.real(f), jnp.imag(f)], axis=-1)


def slice_inverse_fourier(fourier_transform):
    fourier_transform_complex = (
        fourier_transform[..., 0] + 1j * fourier_transform[..., 1]
    )
    inv_fourier_slice = jnp.fft.ifft2(
        jnp.fft.ifftshift(fourier_transform_complex, axes=[-2, -1]),
        norm="ortho",
        axes=[-2, -1],
    )
    return jnp.stack(
        [jnp.real(inv_fourier_slice), jnp.imag(inv_fourier_slice)], axis=-1
    )


class baseMask:
    img_shape: tuple
    task: str
    num_samples: int = 1000
    sigma: float = 0.3
    sigma_prob: float = 1.0

    def make(self, xi: Array) -> Array:
        pass

    def measure_from_mask(self, hist_mask: Array, x: Array) -> Array:
        if self.task == "anomaly":
            f = slice_fourier(x)
            f = jnp.concatenate([f[..., :2], jnp.zeros_like(f[..., :1])], axis=-1)
        else:
            f = slice_fourier(x)
        return einops.einsum(hist_mask, f, "h w, ... h w c -> ... h w c")

    def measure(self, xi: float, x: Array):
        return self.measure_from_mask(self.make(xi), x)

    def restore_from_mask(self, hist_mask: Array, x: Array, measured: Array):
        masked_inv_fourier_x = self.measure_from_mask(1 - hist_mask, x)

        img = slice_inverse_fourier(masked_inv_fourier_x + measured)
        if self.task == "anomaly":
            anomaly_map = jnp.real(x[..., [-1]])
            final = jnp.concatenate([img, anomaly_map], axis=-1)
        else:
            final = img

        return final

    def restore(self, xi: Array, x: Array, measured: Array) -> Array:
        return self.restore_from_mask(self.make(xi), x, measured)

    def init_measurement(self, xi_init: Array) -> MeasurementState:
        y = jnp.zeros(self.img_shape)
        mask_history = jnp.zeros_like(self.make(xi_init))
        return MeasurementState(y=y, mask_history=mask_history)

    def supp_mask(self, xi: Array, hist_mask: Array) -> Array:
        new_mask = self.make(xi)
        inv_mask = 1 - self.make(xi)
        return hist_mask * inv_mask + new_mask

    def update_measurement(
        self, measurement_state: MeasurementState, new_measurement: Array, design: Array
    ) -> MeasurementState:
        y = measurement_state.y
        inv_mask = 1 - self.make(design)
        joint_y = jnp.einsum("ij,ijk->ijk", inv_mask, y) + new_measurement
        mask_history = self.supp_mask(
            design, measurement_state.mask_history
        )
        return MeasurementState(y=joint_y, mask_history=mask_history)

    def logprob_y(self, theta: Array, y: Array, design: Array) -> Array:
        f_y = self.measure(design, theta)
        f_y_flat = einops.rearrange(f_y, "... h w c -> ... (h w c)")
        y_flat = einops.rearrange(y, "... h w c -> ... (h w c)")

        return jax.scipy.stats.multivariate_normal.logpdf(
            y_flat, mean=f_y_flat, cov=self.sigma_prob**2
        )

    def logprob_y_t(self, theta: Array, y: Array, mask: Array, alpha_t: float) -> Array:
        A_theta = self.measure_from_mask(mask, theta)
        tmp = y[..., :2] - A_theta[..., :2]
        tmp = jnp.abs(tmp[..., 0] + 1j * tmp[..., 1]) ** 2
        logprob = - tmp / (self.sigma_prob * alpha_t) - mask.sum() * jnp.log(jnp.pi * self.sigma_prob * alpha_t)
        logprob = einops.einsum(logprob, mask, "t ..., ... -> t ...")
        logprob = einops.reduce(logprob, "t ... -> t", "sum")
        return logprob

    def grad_logprob_y(self, theta: Array, y: Array, design: Array) -> Array:
        meas_x = self.measure_from_mask(design, theta)
        return (
            self.restore_from_mask(design, jnp.zeros_like(theta), (y - meas_x))
            / self.sigma_prob
        ) * 2
