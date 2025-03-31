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
        t = self.tf + step / self.n_steps * (self.eps - self.tf)
        return t
