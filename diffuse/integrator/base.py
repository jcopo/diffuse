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
    position: Array
    rng_key: PRNGKeyArray
    step: int = 0


@dataclass
class Integrator:
    sde: SDE
    timer: Timer

    def init(self, position: Array, rng_key: PRNGKeyArray) -> IntegratorState:
        """Initialize integrator state with position, timestep and step size"""
        return IntegratorState(position, rng_key)

    def __call__(
        self, integrator_state: IntegratorState, score: Callable
    ) -> IntegratorState: ...


@dataclass
class ChurnedIntegrator(Integrator):
    stochastic_churn_rate: float = 1.0
    churn_min: float = 0.5
    churn_max: float = 1.0
    noise_inflation_factor: float = 1.0001

    def _churn_fn(self, integrator_state: IntegratorState) -> Tuple[Array, float]:
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
    """Compute the next churn noise level"""
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
    """Apply stochastic churn to the sample"""
    position, rng_key, step = integrator_state
    t = timer(step)

    t_churned = next_churn_noise_level(
        t, stochastic_churn_rate, churn_min, churn_max, timer
    )
    int_b, int_b_churned = (
        sde.beta.integrate(t, sde.beta.t0),
        sde.beta.integrate(t_churned, sde.beta.t0),
    )
    alpha, alpha_churned = (
        jnp.exp(-int_b),
        jnp.exp(-int_b_churned),
    )

    new_position = (
        jnp.sqrt(alpha_churned / alpha) * position
        + jax.random.normal(rng_key, position.shape)
        * jnp.sqrt(1 - alpha_churned / alpha)
        * noise_inflation_factor
    )

    return new_position, t_churned
