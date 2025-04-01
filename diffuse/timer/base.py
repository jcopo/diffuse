from dataclasses import dataclass

@dataclass
class Timer:
    n_steps: int
    def __call__(self, step: int) -> float:
        ...


@dataclass
class VpTimer(Timer):
    eps: float
    tf: float

    def __call__(self, step: int) -> float:
        return self.tf + step / self.n_steps * (self.eps - self.tf)

@dataclass
class HeunTimer(Timer):
    rho: float = 7.0
    sigma_min: float = 0.002
    sigma_max: float = 80.0

    def __call__(self, step: int) -> float:
        sigma_max_rho = self.sigma_max ** (1/self.rho)
        sigma_min_rho = self.sigma_min ** (1/self.rho)
        return (sigma_max_rho + step / self.n_steps * (sigma_min_rho - sigma_max_rho)) ** self.rho


class DDIMTimer(Timer):
    def __init__(self, n_steps: int, n_time_training: int, c_1: float = 0.001, c_2: float = 0.008, j0: int = 8):
        self.n_steps = n_steps
        self.n_time_training = n_time_training
        self.c_1 = c_1
        self.c_2 = c_2
        self.j0 = j0

        # compute u_j from https://arxiv.org/pdf/2206.00364 p3

    def __call__(self, step: int) -> float:
        t = self.c_1 + (self.c_2 - self.c_1) * (step / self.n_steps) ** self.j0
        return t

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import jax.numpy as jnp
    timer = HeunTimer(n_steps=100, sigma_max=1.)
    ts = jnp.linspace(0, 100, 100)
    plt.plot(ts, timer(ts))
    plt.show()
