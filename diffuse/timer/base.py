from abc import ABC, abstractmethod
from dataclasses import dataclass

class Timer(ABC):
    """Abstract base class for timers"""

    @abstractmethod
    def __call__(self, step: int) -> float:
        pass


@dataclass
class VpTimer(Timer):
    n_steps: int
    eps: float
    tf: float

    def __call__(self, step: int) -> float:
        return self.tf + step / self.n_steps * (self.eps - self.tf)


@dataclass
class HeunTimer(Timer):
    n_steps: int
    rho: float = 7.0
    sigma_min: float = 0.002
    sigma_max: float = 80.0

    def __call__(self, step: int) -> float:
        sigma_max_rho = self.sigma_max ** (1/self.rho)
        sigma_min_rho = self.sigma_min ** (1/self.rho)
        return (sigma_max_rho + step / self.n_steps * (sigma_min_rho - sigma_max_rho)) ** self.rho


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import jax.numpy as jnp
    timer = HeunTimer(n_steps=100, sigma_max=1.)
    ts = jnp.linspace(0, 100, 100)
    plt.plot(ts, timer(ts))
    plt.show()
