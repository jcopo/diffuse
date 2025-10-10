# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class Timer:
    """Base Timer class for scheduling time steps in diffusion processes.

    Args:
        n_steps: Number of discrete time steps
    """

    n_steps: int

    def __call__(self, step: int) -> float: ...


@dataclass
class VpTimer(Timer):
    """Variance Preserving Timer that implements linear interpolation between final and initial time.

    Args:
        n_steps: Number of discrete time steps
        eps: Initial time value
        tf: Final time value
    """

    eps: float
    tf: float

    def __call__(self, step: int) -> float:
        """Compute time value for given step.

        Args:
            step (int): Current step number

        Returns:
            float: Interpolated time value between tf and eps
        """
        return self.tf + step / self.n_steps * (self.eps - self.tf)


@dataclass
class HeunTimer(Timer):
    """Heun Timer implementing power-law scaling of noise levels.

    This timer discretizes noise space rather than time space, using a power-law
    relationship to schedule noise levels. It is designed to be used with sampling
    methods that are defined on noise levels (like EDM - Elucidating the Design
    Space of Diffusion-Based Generative Models) rather than time-based approaches.

    Args:
        n_steps: Number of discrete time steps
        rho: Power scaling factor (default: 7.0)
        sigma_min: Minimum noise level (default: 0.002)
        sigma_max: Maximum noise level (default: 0.002)
    """

    rho: float = 7.0
    sigma_min: float = 0.002
    sigma_max: float = 80.0

    def __call__(self, step: int) -> float:
        """Compute noise level for given step using power-law scaling.

        Args:
            step (int): Current step number

        Returns:
            float: Noise level at current step
        """
        sigma_max_rho = self.sigma_max ** (1 / self.rho)
        sigma_min_rho = self.sigma_min ** (1 / self.rho)
        return (sigma_max_rho + step / self.n_steps * (sigma_min_rho - sigma_max_rho)) ** self.rho


@dataclass
class DDIMTimer(Timer):
    """Denoising Diffusion Implicit Models (DDIM) Timer.

    Implements custom time scheduling for DDIM as described in https://arxiv.org/pdf/2206.00364.
    Uses a power-law interpolation between c_1 and c_2 with exponent j0.

    Args:
        n_steps (int): Number of discrete time steps
        n_time_training (int): Number of training timesteps
        c_1 (float, optional): Lower bound parameter. Defaults to 0.001
        c_2 (float, optional): Upper bound parameter. Defaults to 0.008
        j0 (int, optional): Power-law exponent. Defaults to 8
    """

    n_time_training: int
    c_1: float = 0.001
    c_2: float = 0.008
    j0: int = 8

    def __post_init__(self):
        def body_fun(u, i):
            alpha = self._alpha(i)
            alpha_next = self._alpha(i - 1)
            maxi = jnp.maximum(alpha_next / alpha, self.c_1)
            u_next = jnp.sqrt((u**2 + 1) / maxi - 1)
            return u_next, u_next

        indices = jnp.arange(self.n_time_training, 0, -1)
        _, self.u_list = jax.lax.scan(body_fun, 0.0, indices)

    def __call__(self, step: int) -> float:
        """Compute time value for given step using DDIM scheduling.

        Args:
            step (int): Current step number

        Returns:
            float: Time value at current step
        """
        j = jnp.floor(self.j0 + (self.n_time_training - 1 - self.j0) * step / (self.n_steps - 1) + 0.5).astype(int).item()
        return self.u_list[j]

    def _alpha(self, j: int) -> float:
        return jnp.sin(0.5 * jnp.pi * j / (self.n_time_training * (self.c_2 + 1))) ** 2


if __name__ == "__main__":
    timer = DDIMTimer(n_steps=100, n_time_training=1000, c_1=0.001, c_2=0.008, j0=8)
    print(timer(0))
    print(timer(50))
    print(timer(100))
