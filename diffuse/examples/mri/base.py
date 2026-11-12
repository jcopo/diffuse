# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import ForwardModel, MeasurementState

# ---------------------------------------------------------------------------
# Empirical parameter grids inherited from the driax MRI experiments
# ---------------------------------------------------------------------------

RADIAL_LINE_RANGE = {
    "kneeFastMRI": {"minval": 20.0, "maxval": 400.0},
    "fastMRI": {"minval": 20.0, "maxval": 400.0},
    "brainFastMRI": {"minval": 20.0, "maxval": 400.0},
    "WMH": {"minval": 5.0, "maxval": 120.0},
    "BRATS": {"minval": 0.0, "maxval": 180.0},
}

RADIAL_SIGMA = {
    "kneeFastMRI": 0.1,
    "fastMRI": 0.1,
    "brainFastMRI": 0.5,
    "WMH": 0.2,
    "BRATS": 0.8,
}

RANDOM_SIGMA = {
    "kneeFastMRI": 0.1,
    "fastMRI": 0.1,
    "brainFastMRI": 0.1,
    "WMH": 0.1,
    "BRATS": 0.1,
}

RANDOM_SPARSITY = {
    "kneeFastMRI": 0.005,
    "fastMRI": 0.005,
    "brainFastMRI": 0.005,
    "WMH": 0.05,
    "BRATS": 0.005,
}

VERTICAL_SIGMA = {
    "kneeFastMRI": 0.1,
    "fastMRI": 0.1,
    "brainFastMRI": 0.1,
    "WMH": 0.1,
    "BRATS": 0.1,
}

SPIRAL_RANGE = {
    "kneeFastMRI": {"minval": 0.0, "maxval": 1.0},
    "fastMRI": {"minval": 0.0, "maxval": 1.0},
    "brainFastMRI": {"minval": 0.0, "maxval": 1.0},
    "WMH": {"minval": 0.0, "maxval": 1.0},
    "BRATS": {"minval": 0.0, "maxval": 1.0},
}


# ---------------------------------------------------------------------------
# Fourier helpers
# ---------------------------------------------------------------------------


def slice_fourier(mri_slice: Array) -> Array:
    """Compute the centered 2-D Fourier transform of a complex-valued slice.

    The input is represented with real/imaginary components along the last axis.
    """
    if mri_slice.shape[-1] < 2:
        raise ValueError("MRI slices must provide at least two channels (real, imaginary).")
    complex_slice = mri_slice[..., 0] + 1j * mri_slice[..., 1]
    spectrum = jnp.fft.fftshift(jnp.fft.fft2(complex_slice, norm="ortho", axes=(-2, -1)), axes=(-2, -1))
    return jnp.stack([jnp.real(spectrum), jnp.imag(spectrum)], axis=-1)


def slice_inverse_fourier(kspace: Array) -> Array:
    """Inverse centered 2-D Fourier transform returning real/imag channels."""
    if kspace.shape[-1] < 2:
        raise ValueError("k-space measurements must provide at least two channels (real, imaginary).")
    complex_kspace = kspace[..., 0] + 1j * kspace[..., 1]
    image = jnp.fft.ifft2(
        jnp.fft.ifftshift(complex_kspace, axes=(-2, -1)),
        norm="ortho",
        axes=(-2, -1),
    )
    return jnp.stack([jnp.real(image), jnp.imag(image)], axis=-1)


def _prepare_mask(mask: Array) -> Array:
    """Ensure the mask is 2-D even if stored with a trailing singleton axis."""
    mask = jnp.asarray(mask)
    if mask.ndim < 2:
        raise ValueError("Masks must be at least two-dimensional (height, width).")
    if mask.ndim > 2:
        if mask.shape[-1] != 1:
            raise ValueError("Masks with more than two dimensions must have a trailing singleton axis.")
        mask = jnp.squeeze(mask, axis=-1)
    return mask


# ---------------------------------------------------------------------------
# Abstract MRI forward model
# ---------------------------------------------------------------------------


@dataclass
class MRIMaskForwardModel(ForwardModel):
    """Base class for MRI forward models using k-space masking strategies.

    Subclasses must implement :meth:`make` (build a mask from design parameters)
    and optionally :meth:`sample_design` / :meth:`projection_design` for active
    sensing scenarios.
    """

    img_shape: Tuple[int, int, int]
    std: float = 0.01
    data_model: str = "kneeFastMRI"
    task: str = "reconstruction"

    # ------------------------------------------------------------------ #
    # Mask construction API
    # ------------------------------------------------------------------ #
    def make(self, xi: Array, rng_key: Optional[PRNGKeyArray] = None) -> Array:  # pragma: no cover - abstract
        """Create a (soft) binary mask from design parameters."""
        raise NotImplementedError

    def sample_design(self, rng_key: PRNGKeyArray) -> Array:  # pragma: no cover - optional
        """Sample design parameters for the next measurement."""
        raise NotImplementedError

    def projection_design(self, xi: Array) -> Array:
        """Project design parameters to the feasible set (identity by default)."""
        return xi

    # ------------------------------------------------------------------ #
    # Measurement operators
    # ------------------------------------------------------------------ #
    def measure_from_mask(self, mask: Array, image: Array) -> Array:
        """Apply the masked Fourier operator to the provided image."""
        mask_2d = _prepare_mask(mask)
        spectrum = slice_fourier(image)
        if self.task == "anomaly":
            zeros = jnp.zeros_like(spectrum[..., :1])
            spectrum = jnp.concatenate([spectrum, zeros], axis=-1)
        return mask_2d[..., None] * spectrum

    def apply(self, img: Array, measurement_state: MeasurementState) -> Array:
        """Forward measurement operator A x."""
        return self.measure_from_mask(measurement_state.mask_history, img)

    def adjoint(self, meas: Array, measurement_state: MeasurementState) -> Array:
        """Adjoint operator Aᵀ y (masked inverse FFT)."""
        mask_2d = _prepare_mask(measurement_state.mask_history)
        masked_meas = mask_2d[..., None] * meas
        reconstructed = slice_inverse_fourier(masked_meas)
        if self.task == "anomaly":
            zeros = jnp.zeros_like(reconstructed[..., :1])
            return jnp.concatenate([reconstructed, zeros], axis=-1)
        return reconstructed

    def restore(self, meas: Array, measurement_state: MeasurementState) -> Array:
        """Alias for the adjoint in the new conditional denoiser API."""
        return self.adjoint(meas, measurement_state)

    # ------------------------------------------------------------------ #
    # Measurement-state management
    # ------------------------------------------------------------------ #
    def init_measurement(self, mask: Optional[Array] = None) -> MeasurementState:
        """Create an empty measurement state with either the provided mask or zeros."""
        if mask is None:
            mask = jnp.zeros(self.img_shape[:-1])
        mask_proj = self.projection_design(mask)
        zero_image = jnp.zeros(self.img_shape, dtype=jnp.float32)
        y = self.measure_from_mask(mask_proj, zero_image)
        return MeasurementState(y=y, mask_history=mask_proj)

    def supplement_mask(self, new_mask: Array, mask_history: Array) -> Array:
        """Accumulate a new mask into the measurement history (avoid duplicates)."""
        hist_2d = _prepare_mask(mask_history)
        new_mask_2d = _prepare_mask(new_mask)
        inv_hist = 1.0 - hist_2d
        return hist_2d + inv_hist * new_mask_2d

    def update_measurement(
        self,
        measurement_state: MeasurementState,
        ground_truth: Array,
        design: Array,
        rng_key: Optional[PRNGKeyArray] = None,
    ) -> MeasurementState:
        """Update measurement state with a fresh design applied to `ground_truth`."""
        mask = self.make(design, rng_key)
        mask_history = self.supplement_mask(mask, measurement_state.mask_history)
        y = self.measure_from_mask(mask_history, ground_truth)
        return MeasurementState(y=y, mask_history=mask_history)


__all__ = [
    "MRIMaskForwardModel",
    "MeasurementState",
    "RADIAL_LINE_RANGE",
    "RADIAL_SIGMA",
    "RANDOM_SIGMA",
    "RANDOM_SPARSITY",
    "VERTICAL_SIGMA",
    "SPIRAL_RANGE",
    "slice_fourier",
    "slice_inverse_fourier",
]
