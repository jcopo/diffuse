"""Flux-specific diffusion timer utilities."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from diffuse.timer.base import VpTimer


def _mobius_shift(sigma: float, shift: float) -> float:
    """Apply the Möbius shift used by Flux schedulers."""
    if shift == 1.0:
        return sigma
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _exponential_time_shift(t: float, mu: float, sigma: float = 1.0) -> float:
    """Exponential time shift used by diffusers for dynamic schedules."""
    exp_mu = jnp.exp(mu)
    return exp_mu / (exp_mu + (1.0 / t - 1.0) ** sigma)


def _linear_time_shift(t: float, mu: float, sigma: float = 1.0) -> float:
    """Linear time shift (alternative dynamic schedule)."""
    return mu / (mu + (1.0 / t - 1.0) ** sigma)


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Compute a resolution-dependent Möbius shift."""
    if image_seq_len <= base_seq_len:
        return base_shift
    if image_seq_len >= max_seq_len:
        return max_shift
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


@dataclass
class FluxTimer(VpTimer):
    """VpTimer variant that applies the Flux Möbius shift to each sigma value.

    The base VpTimer produces a linear schedule in sigma space. Flux applies an
    additional transform to bias sampling toward low-noise regions. When
    `use_dynamic_shift=False` a fixed Möbius shift is used (Flux-dev defaults
    to 1.15). When `use_dynamic_shift=True` we follow diffusers' exponential or
    linear time shifting driven by the resolution dependent parameter ``mu``.
    """

    shift: float = 1.15
    use_dynamic_shift: bool = False
    shift_type: str = "exponential"  # "exponential" or "linear"
    base_seq_len: int = 256
    max_seq_len: int = 4096
    base_shift: float = 0.5
    max_shift: float = 1.15
    _mu: float | None = None

    def __post_init__(self) -> None:
        if self.shift <= 0.0:
            raise ValueError("FluxTimer shift must be positive.")
        if self.shift_type not in ("exponential", "linear"):
            raise ValueError("shift_type must be 'exponential' or 'linear'")

    def set_image_seq_len(self, seq_len: int) -> None:
        """Update internal mu using the Flux dynamic schedule."""
        if self.use_dynamic_shift:
            self._mu = calculate_shift(
                seq_len,
                base_seq_len=self.base_seq_len,
                max_seq_len=self.max_seq_len,
                base_shift=self.base_shift,
                max_shift=self.max_shift,
            )
        else:
            self._mu = None

    def __call__(self, step: int) -> float:
        sigma = super().__call__(step)
        if self.use_dynamic_shift:
            if self._mu is None:
                raise RuntimeError("Dynamic shift enabled but set_image_seq_len() has not been called.")
            if self.shift_type == "exponential":
                return _exponential_time_shift(sigma, self._mu)
            return _linear_time_shift(sigma, self._mu)
        return _mobius_shift(sigma, self.shift)

    def apply_shift(self, sigmas):
        """Vectorised helper to apply the Flux Möbius shift to an array."""
        if self.use_dynamic_shift:
            if self._mu is None:
                raise RuntimeError("Dynamic shift enabled but set_image_seq_len() has not been called.")
            shift_fn = _exponential_time_shift if self.shift_type == "exponential" else _linear_time_shift
            return jnp.array([shift_fn(s, self._mu) for s in sigmas])
        return _mobius_shift(sigmas, self.shift)
