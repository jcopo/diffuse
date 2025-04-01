from dataclasses import dataclass


@dataclass
class Timer:
    """Base Timer class for scheduling time steps in diffusion processes.

    Args:
        n_steps (int): Number of discrete time steps.
    """

    n_steps: int

    def __call__(self, step: int) -> float: ...


@dataclass
class VpTimer(Timer):
    """Variance Preserving Timer that implements linear interpolation between final and initial time.

    Args:
        n_steps (int): Number of discrete time steps
        eps (float): Initial time value
        tf (float): Final time value
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

    This timer is often used in diffusion models to schedule noise levels
    following a power-law relationship.

    Args:
        n_steps (int): Number of discrete time steps
        rho (float, optional): Power scaling factor. Defaults to 7.0
        sigma_min (float, optional): Minimum noise level. Defaults to 0.002
        sigma_max (float, optional): Maximum noise level. Defaults to 80.0
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
        return (
            sigma_max_rho + step / self.n_steps * (sigma_min_rho - sigma_max_rho)
        ) ** self.rho


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
    c_1: float
    c_2: float
    j0: int

    def __call__(self, step: int) -> float:
        """Compute time value for given step using DDIM scheduling.

        Args:
            step (int): Current step number

        Returns:
            float: Time value at current step
        """
        t = self.c_1 + (self.c_2 - self.c_1) * (step / self.n_steps) ** self.j0
        return t


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import jax.numpy as jnp

    timer = HeunTimer(n_steps=100, sigma_max=1.0)
    ts = jnp.linspace(0, 100, 100)
    plt.plot(ts, timer(ts))
    plt.show()
