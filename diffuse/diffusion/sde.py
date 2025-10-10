# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, NamedTuple, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class SDEState(NamedTuple):
    position: Array
    t: Array


class Schedule(ABC):
    T: float

    @abstractmethod
    def __call__(self, t: Array) -> Array:
        pass

    @abstractmethod
    def integrate(self, t: Array, s: Array) -> Array:
        pass


@dataclass
class LinearSchedule:
    r"""Linear noise schedule for diffusion processes.

    Implements a linear interpolation between minimum and maximum noise levels:

    .. math::
        \beta(t) = \beta_{\min} + \frac{\beta_{\max} - \beta_{\min}}{T - t_0}(t - t_0)

    Args:
        b_min: The minimum noise value :math:`\beta_{\min}`
        b_max: The maximum noise value :math:`\beta_{\max}`
        t0: The starting time :math:`t_0`
        T: The ending time :math:`T`
    """

    b_min: float
    b_max: float
    t0: float
    T: float

    def __call__(self, t: Array) -> Array:
        r"""Evaluate the linear schedule at time t.

        Args:
            t: Time at which to evaluate the schedule

        Returns:
            Schedule value :math:`\beta(t)`
        """
        b_min, b_max, t0, T = self.b_min, self.b_max, self.t0, self.T
        return (b_max - b_min) / (T - t0) * t + (b_min * T - b_max * t0) / (T - t0)

    def integrate(self, t: Array, s: Array) -> Array:
        r"""Compute integral :math:`\int_s^t \beta(\tau) d\tau`.

        Args:
            t: Upper integration bound
            s: Lower integration bound

        Returns:
            Integral value
        """
        b_min, b_max, t0, T = self.b_min, self.b_max, self.t0, self.T
        slope = (b_max - b_min) / (T - t0)
        intercept = (b_min * T - b_max * t0) / (T - t0)
        return 0.5 * (t - s) * (slope * (t + s) + 2 * intercept)


@dataclass
class CosineSchedule(Schedule):
    r"""Cosine noise schedule for improved denoising.

    Implements the cosine schedule from Nichol & Dhariwal (2021) which provides
    better signal-to-noise ratio properties than linear schedules. The schedule is
    based on:

    .. math::
        \bar{\alpha}(t) = \frac{\cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2}{\cos\left(\frac{s}{1+s} \cdot \frac{\pi}{2}\right)^2}

    Args:
        b_min: The minimum beta value :math:`\beta_{\min}`
        b_max: The maximum beta value :math:`\beta_{\max}`
        t0: The starting time :math:`t_0`
        T: The ending time :math:`T`
        s: Offset parameter for numerical stability (default: 0.008)

    References:
        Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models.
        arXiv:2102.09672
    """

    b_min: float
    b_max: float
    t0: float
    T: float
    s: float = 0.008

    def __call__(self, t: Array) -> Array:
        r"""Evaluate the cosine schedule at time t.

        Args:
            t: Time at which to evaluate the schedule

        Returns:
            Schedule value :math:`\beta(t)` clipped to [:math:`\beta_{\min}`, :math:`\beta_{\max}`]
        """
        t_normalized = (t - self.t0) / (self.T - self.t0)

        beta_t = jnp.pi * jnp.tan(0.5 * jnp.pi * (t_normalized + self.s) / (1 + self.s)) / (self.T * (1 + self.s))
        beta_t = jnp.clip(beta_t, self.b_min, self.b_max)

        return beta_t

    def integrate(self, t: Array, s: Array) -> Array:
        r"""Compute integral :math:`\int_s^t \beta(\tau) d\tau` using :math:`\bar{\alpha}` values.

        Returns :math:`\log(\bar{\alpha}(s) / \bar{\alpha}(t))`

        Args:
            t: Upper integration bound
            s: Lower integration bound

        Returns:
            Integral value
        """
        time_scale = self.T - self.t0
        offset_scale = 1 + self.s

        t_norm = (t - self.t0) / time_scale
        s_norm = (s - self.t0) / time_scale

        f0 = jnp.cos(self.s / offset_scale * jnp.pi * 0.5) ** 2
        ft = jnp.cos((t_norm + self.s) / offset_scale * jnp.pi * 0.5) ** 2
        fs = jnp.cos((s_norm + self.s) / offset_scale * jnp.pi * 0.5) ** 2

        alpha_t = jnp.clip(ft / f0, 0.001, 0.9999)
        alpha_s = jnp.clip(fs / f0, 0.001, 0.9999)

        return jnp.log(alpha_s / alpha_t)


class DiffusionModel(ABC):
    @abstractmethod
    def noise_level(self, t: Array) -> Array:
        pass

    @abstractmethod
    def signal_level(self, t: Array) -> Array:
        pass

    @abstractmethod
    def sde_coefficients(self, t: Array) -> tuple[Array, Array]:
        """Compute SDE coefficients f(t) and g(t) for dx_t = f(t) x_t dt + g(t) dW_t."""
        pass

    def snr(self, t: Array) -> Array:
        """
        Compute Signal-to-Noise Ratio (SNR) at timestep t.

        For general interpolation x_t = α_t x_0 + σ_t ε:
        SNR(t) = α_t² / σ_t²
        """
        noise_level = self.noise_level(t)
        signal_level = self.signal_level(t)
        return (signal_level * signal_level) / (noise_level * noise_level + 1e-8)

    def score(self, state: SDEState, state_0: SDEState) -> Array:
        """
        Closed-form expression for the score function ∇ₓ log p(xₜ | x₀) of the Gaussian transition kernel.

        From docs: ∇log p_t(x_t|x_0) = -1/σ_t² (x_t - α_t x_0)
        """
        x, t = state.position, state.t
        x0, _t0 = state_0.position, state_0.t
        sigma_t = self.noise_level(t)
        signal_level_t = self.signal_level(t)

        return -(x - signal_level_t * x0) / (sigma_t * sigma_t)

    def tweedie(self, state: SDEState, score_fn: Callable) -> SDEState:
        """
        Tweedie's formula to compute E[x_0 | x_t].

        From docs: x̂_0 = 1/α_t (x_t + σ_t² ∇log p_t(x_t))
        """
        x, t = state.position, state.t
        sigma_t = self.noise_level(t)
        signal_level_t = self.signal_level(t)
        return SDEState((x + sigma_t * sigma_t * score_fn(x, t)) / signal_level_t, jnp.zeros_like(t))

    def path(self, key: PRNGKeyArray, state: SDEState, ts: Array, return_noise: bool = False) -> Union[SDEState, tuple[SDEState, Array]]:
        """
        Samples from the general interpolation: x_t = α_t x_0 + σ_t ε
        """
        x = state.position
        sigma_t = self.noise_level(ts)
        signal_level_t = self.signal_level(ts)

        noise = jax.random.normal(key, x.shape, dtype=x.dtype)
        res = signal_level_t * x + sigma_t * noise
        return (SDEState(res, ts), noise) if return_noise else SDEState(res, ts)


@dataclass
class SDE(DiffusionModel):
    r"""Variance Preserving (VP) SDE for diffusion models.

    Implements the forward SDE:

    .. math::
        dX(t) = -\frac{1}{2}\beta(t) X(t) dt + \sqrt{\beta(t)} dW(t)

    where :math:`\beta(t)` is the noise schedule and :math:`dW(t)` is the Wiener process.

    This formulation preserves the variance of the data distribution and is the
    standard choice for diffusion probabilistic models.

    Args:
        beta: Noise schedule (LinearSchedule or CosineSchedule)
    """

    beta: Schedule

    def __post_init__(self):
        self.tf = self.beta.T

    def sde_coefficients(self, t: Array) -> tuple[Array, Array]:
        r"""Compute SDE coefficients :math:`f(t)` and :math:`g(t)`.

        For the VP-SDE: :math:`dX(t) = -\frac{1}{2}\beta(t) X(t) dt + \sqrt{\beta(t)} dW(t)`

        Returns:
            Tuple of drift coefficient :math:`f(t) = -\frac{1}{2}\beta(t)` and
            diffusion coefficient :math:`g(t) = \sqrt{\beta(t)}`
        """
        beta_t = self.beta(t)
        f_t = -0.5 * beta_t
        g_t = jnp.sqrt(beta_t)
        return f_t, g_t

    def noise_level(self, t: Array) -> Array:
        r"""Compute noise level :math:`\sigma(t)` for diffusion process.

        The solution to the VP-SDE is:

        .. math::
            X(t) = \alpha(t) X_0 + \sigma(t) \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)

        where :math:`\alpha(t) = \exp\left(-\frac{1}{2}\int_0^t \beta(s) ds\right)` and
        :math:`\sigma(t) = \sqrt{1 - \alpha^2(t)}`

        Returns:
            Noise level :math:`\sigma(t)` clipped for numerical stability
        """
        alpha = jnp.exp(-self.beta.integrate(t, jnp.zeros_like(t)))
        sigma = jnp.sqrt(1 - alpha)
        sigma = jnp.clip(sigma, 0.001, 0.9999)
        return sigma

    def signal_level(self, t: Array) -> Array:
        r"""Compute signal level :math:`\alpha(t) = \exp\left(-\frac{1}{2}\int_0^t \beta(s) ds\right)`.

        Returns:
            Signal level clipped for numerical stability
        """
        alpha = jnp.sqrt(jnp.exp(-self.beta.integrate(t, jnp.zeros_like(t))))
        alpha = jnp.clip(alpha, 0.001, 0.9999)
        return alpha


@dataclass
class Flow(DiffusionModel):
    r"""Rectified Flow diffusion model with straight-line interpolation paths.

    Implements the rectified flow formulation from Liu et al. (2022) with linear schedules:

    .. math::
        \alpha(t) = 1 - t, \quad \sigma(t) = t

    This creates straight-line interpolation paths:

    .. math::
        x_t = (1-t)x_0 + t\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)

    which are more amenable to ODE-based sampling with fewer discretization steps.

    Args:
        tf: Final time for the diffusion process (default: 1.0)

    References:
        Liu, X., Gong, C., & Liu, Q. (2022). Flow straight and fast: Learning to
        generate and transfer data with rectified flow. arXiv:2209.03003
    """

    tf: float = 1.0

    def noise_level(self, t: Array) -> Array:
        r"""Compute noise level :math:`\sigma(t) = t`.

        Returns:
            Noise level clipped for numerical stability
        """
        return jnp.clip(t / self.tf, 0.001, 0.999)

    def signal_level(self, t: Array) -> Array:
        r"""Compute signal level :math:`\alpha(t) = 1 - t`.

        Returns:
            Signal level clipped for numerical stability
        """
        return jnp.clip(1 - t / self.tf, 0.001, 0.999)

    def sde_coefficients(self, t: Array) -> tuple[Array, Array]:
        r"""Compute SDE coefficients for rectified flow.

        Returns drift :math:`f(t) = -\frac{1}{1-t}` and diffusion :math:`g(t) = \sqrt{\frac{2t}{1-t}}`

        Returns:
            Tuple of drift and diffusion coefficients
        """
        t_safe = jnp.clip(t / self.tf, 0.001, 0.999)
        f_t = -1.0 / (1 - t_safe)
        g_t = jnp.sqrt(2 * t_safe / (1 - t_safe))
        return f_t, g_t


@dataclass
class EDM(DiffusionModel):
    r"""Efficient Diffusion Model (EDM) from Karras et al. (2022).

    Implements the EDM formulation with constant signal and increasing noise:

    .. math::
        \alpha(t) = 1, \quad \sigma(t) = t

    This creates the simple interpolation:

    .. math::
        x_t = x_0 + t\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)

    which simplifies the probability-flow ODE and is particularly effective
    with Heun's integration method.

    Args:
        tf: Final time for the diffusion process (default: 1.0)

    References:
        Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the
        design space of diffusion-based generative models. NeurIPS 35, 26565-26577.
    """

    tf: float = 1.0

    def noise_level(self, t: Array) -> Array:
        r"""Compute noise level :math:`\sigma(t) = t`.

        Returns:
            Noise level clipped for numerical stability
        """
        return jnp.clip(t, 0.001, 0.999)

    def signal_level(self, t: Array) -> Array:
        r"""Compute signal level :math:`\alpha(t) = 1`.

        Returns:
            Constant signal level of 1
        """
        return jnp.ones_like(t)

    def sde_coefficients(self, t: Array) -> tuple[Array, Array]:
        r"""Compute SDE coefficients for EDM.

        Returns drift :math:`f(t) = 0` and diffusion :math:`g(t) = 1`

        Returns:
            Tuple of zero drift and unit diffusion coefficients
        """
        f_t = jnp.zeros_like(t)
        g_t = jnp.ones_like(t)
        return f_t, g_t


def check_snr(model: DiffusionModel, t: Array, tolerance: float = 1e-3) -> Array:
    """
    Check if SNR at timestep t is effectively zero.

    Args:
        model: DiffusionModel instance
        t: Timestep to check
        tolerance: Tolerance for considering SNR as zero

    Returns:
        True if SNR is effectively zero
    """
    return jnp.all(model.snr(t) < tolerance)
