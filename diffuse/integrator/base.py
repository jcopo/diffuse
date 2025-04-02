from dataclasses import dataclass
from functools import partial
from jaxtyping import Array, PRNGKeyArray
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from diffuse.diffusion.sde import SDE
from diffuse.timer.base import Timer

__all__ = ["Integrator", "IntegratorState", "ChurnedIntegrator"]


class IntegratorState(NamedTuple):
    """State container for numerical integrators.

    Attributes:
        position: Current state vector/tensor
        rng_key: JAX random number generator key
        step: Current integration step counter (default: 0)
    """

    position: Array
    rng_key: PRNGKeyArray
    step: int = 0


@dataclass
class Integrator:
    """Base class for numerical integrators of diffusion processes.

    Provides the basic interface for implementing various numerical integration
    schemes for both deterministic and stochastic differential equations.

    Attributes:
        sde: Stochastic Differential Equation object defining the diffusion process
        timer: Timer object managing the discretization of the time interval
    """

    sde: SDE
    timer: Timer

    def init(self, position: Array, rng_key: PRNGKeyArray) -> IntegratorState:
        """Initialize the integrator state.

        Args:
            position: Initial state vector/tensor
            rng_key: JAX random number generator key

        Returns:
            Initial IntegratorState
        """
        return IntegratorState(position, rng_key)

    def __call__(
        self, integrator_state: IntegratorState, score: Callable
    ) -> IntegratorState:
        """Perform one integration step.

        Args:
            integrator_state: Current state of the integration
            score: Score function approximating ∇ₓ log p(x|t)

        Returns:
            Updated IntegratorState
        """
        ...


@dataclass
class ChurnedIntegrator(Integrator):
    """Integrator with stochastic churning for improved sampling.

    Implements the stochastic churning mechanism that can help improve sampling
    quality by occasionally injecting controlled noise into the process.

    Attributes:
        stochastic_churn_rate: Rate of applying stochastic churning (default: 1.0)
        churn_min: Minimum time threshold for churning (default: 0.5)
        churn_max: Maximum time threshold for churning (default: 1.0)
        noise_inflation_factor: Factor to scale injected noise (default: 1.0001)
    """

    stochastic_churn_rate: float = 1.0
    churn_min: float = 0.5
    churn_max: float = 2.0
    noise_inflation_factor: float = 1.0001

    def _churn_fn(self, integrator_state: IntegratorState) -> Tuple[Array, float]:
        """Apply stochastic churning to the current state.

        Args:
            integrator_state: Current integration state

        Returns:
            Tuple of (churned_position, churned_time)
        """
        position, _, step = integrator_state
        t = self.timer(step)

        _apply_stochastic_churn = partial(
            apply_stochastic_churn,
            stochastic_churn_rate=self.stochastic_churn_rate,
            churn_min=self.churn_min,
            churn_max=self.churn_max,
            noise_inflation_factor=self.noise_inflation_factor,
            sde=self.sde,
            timer=self.timer,
        )
        position_churned, t_churned = jax.lax.cond(
            self.stochastic_churn_rate > 0,
            _apply_stochastic_churn,
            lambda _: (position, t),
            integrator_state,
        )
        return position_churned, t_churned


def next_churn_noise_level(
    t: float,
    stochastic_churn_rate: float,
    churn_min: float,
    churn_max: float,
    timer: Timer,
) -> float:
    """Compute the next noise level for stochastic churning.

    Determines the appropriate noise level based on the current time and churning
    parameters, ensuring the noise stays within specified bounds.

    Args:
        t: Current time
        stochastic_churn_rate: Rate of stochastic churning
        churn_min: Minimum time threshold for churning
        churn_max: Maximum time threshold for churning
        timer: Timer object managing time discretization

    Returns:
        Next noise level for churning
    """
    churn_rate = jnp.where(
        stochastic_churn_rate / timer.n_steps - jnp.sqrt(2) + 1 > 0,
        jnp.sqrt(2) - 1,
        stochastic_churn_rate / timer.n_steps,
    )
    churn_rate = jnp.where(t > churn_min, jnp.where(t < churn_max, churn_rate, 0), 0)
    return t * (1 + churn_rate)


def apply_stochastic_churn(
    integrator_state: IntegratorState,
    stochastic_churn_rate: float,
    churn_min: float,
    churn_max: float,
    noise_inflation_factor: float,
    sde: SDE,
    timer: Timer,
) -> Tuple[Array, float]:
    """Apply stochastic churning to the current sample.

    Implements the stochastic churning mechanism by:
    1. Computing the next noise level
    2. Adjusting the position using the noise schedule
    3. Injecting scaled random noise

    Args:
        integrator_state: Current integration state
        stochastic_churn_rate: Rate of stochastic churning
        churn_min: Minimum time threshold for churning
        churn_max: Maximum time threshold for churning
        noise_inflation_factor: Factor to scale injected noise
        sde: SDE object defining the diffusion process
        timer: Timer object managing time discretization

    Returns:
        Tuple of (churned_position, churned_time)

    Notes:
        The churning process follows:
        x_churned = sqrt(α_churned/α) * x + sqrt(1 - α_churned/α) * ε * noise_factor
        where:
        - α values are computed from the noise schedule
        - ε is standard normal noise
    """
    position, rng_key, step = integrator_state
    t = timer(step)

    t_churned = next_churn_noise_level(
        t, stochastic_churn_rate, churn_min, churn_max, timer
    )
    alpha, alpha_churned = (
        sde.alpha_beta(t)[0],
        sde.alpha_beta(t_churned)[0],
    )

    new_position = (
        jnp.sqrt(alpha_churned / alpha) * position
        + jax.random.normal(rng_key, position.shape)
        * jnp.sqrt(1 - alpha_churned / alpha)
        * noise_inflation_factor
    )

    return new_position, t_churned
