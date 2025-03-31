from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable
import jax.numpy as jnp
from jax.random import PRNGKeyArray

@dataclass
class TimerState:
    """State of the timer containing current time and step information"""
    t: float  # Current time
    dt: float  # Time step size
    step_count: int  # Current step count
    total_steps: int  # Total number of steps

class BaseTimer(ABC):
    """Base class for SDE timers that control the time evolution of the system"""

    def __init__(self, T: float, n_steps: int):
        """
        Initialize the timer.

        Args:
            T (float): Total time duration
            n_steps (int): Number of steps to take
        """
        self.T = T
        self.n_steps = n_steps

    def init(self, t0: float = 0.0) -> TimerState:
        """
        Initialize the timer state.

        Args:
            t0 (float): Initial time (default: 0.0)

        Returns:
            TimerState: Initial timer state
        """
        dt = (self.T - t0) / self.n_steps
        return TimerState(t=t0, dt=dt, step_count=0, total_steps=self.n_steps)

    @abstractmethod
    def step(self, state: TimerState) -> TimerState:
        """
        Advance the timer by one step.

        Args:
            state (TimerState): Current timer state

        Returns:
            TimerState: Next timer state
        """
        pass

    def is_done(self, state: TimerState) -> bool:
        """
        Check if the timer has finished.

        Args:
            state (TimerState): Current timer state

        Returns:
            bool: True if timer has finished, False otherwise
        """
        return state.step_count >= state.total_steps
