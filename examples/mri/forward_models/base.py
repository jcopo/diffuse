import einops
import jax
import jax.experimental
import jax.numpy as jnp
from jaxtyping import Array

from diffuse.base_forward_model import MeasurementState


######## RADIAL ########
PARAMS_SIZE_LINE = {
    "kneeFastMRI": {"minval": 20.0, "maxval": 400},
    "brainFastMRI": {"minval": 20.0, "maxval": 400},
    "WMH": {"minval": 5.0, "maxval": 120},
    "BRATS": {"minval": 0.0, "maxval": 15},
}

PARAMS_SIGMA_RADIAL = {
    "kneeFastMRI": .1,
    "brainFastMRI": .5,
    "WMH": .2,
    "BRATS": 0.8,
}

######## RANDOM ########
PARAMS_SIGMA_RDM = {
    "kneeFastMRI": .1,
    "brainFastMRI": .1,
    "WMH": .1,
    "BRATS": .1,
}

PARAMS_SPARSITY = {
    "kneeFastMRI": 0.005,
    "brainFastMRI": 0.005,
    "WMH": 0.05,
    "BRATS": 0.005,
}

######## VERTICAL ########
PARAMS_SIGMA_VERTICAL = {
    "kneeFastMRI": 10.,
    "brainFastMRI": .1,
    "WMH": .1,
    "BRATS": .1,
}
######## SPIRAL ########
PARAMS_SIGMA = {
    "kneeFastMRI": 5,
    "brainFastMRI": 5,
    "WMH": 1.5,
    "BRATS": 2,
}

PARAMS_FOV = {
    "kneeFastMRI": {'minval': 2.5, 'maxval': 3.5},
    "brainFastMRI": {'minval': 2.5, 'maxval': 3.5},
    "WMH": {'minval': 2.5, 'maxval': 3.5},
    "BRATS": {'minval': 0.5, 'maxval': 0.6},
}

PARAMS_KMAX = {
    "kneeFastMRI": {'minval': 3, 'maxval': 6},
    "brainFastMRI": {'minval': 3, 'maxval': 6},
    "WMH": {'minval': 3, 'maxval': 6},
    "BRATS": {'minval': 0.5, 'maxval': 1},
}



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
    data_model: str
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
        self, measurement_state: MeasurementState, ground_truth: Array, design: Array
    ) -> MeasurementState:
        mask_history = self.supp_mask(
            design, measurement_state.mask_history
        )
        joint_y = self.measure_from_mask(mask_history, ground_truth)
        return MeasurementState(y=joint_y, mask_history=mask_history)

    def logprob_y(self, theta: Array, y: Array, design: Array) -> Array:
        mask = self.make(design)
        return self.logprob_y_t(theta, y, mask, 1.0)

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
        diff = y - meas_x
        restored = self.restore_from_mask(design, jnp.zeros_like(theta), diff)
        norm_diff = jnp.linalg.norm(diff)
        return restored #/ norm_diff
