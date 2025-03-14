from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class TimestepScheduler(ABC):
    n_steps: int
    T: float

    @abstractmethod
    def __call__(self, i: int) -> float:
        pass

@dataclass
class LinearTimestepScheduler:
    n_steps: int
    T: float

    def __call__(self, i: int) -> float:
        return i / (self.n_steps - 1) * (self.T - 1e-3)
