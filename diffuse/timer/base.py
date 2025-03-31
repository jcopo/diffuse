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

    def __call__(self, step: int) -> float:
        t = 1 + step / self.n_steps * (self.eps - 1)
        return t


if __name__ == "__main__":
    timer = VpTimer(n_steps=100, eps=0.001)
    print(timer(0))
    print(timer(1))
    print(timer(100))
