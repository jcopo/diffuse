from dataclasses import dataclass
from typing import Callable, Iterable
from functools import partial
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.integrator.base import IntegratorState
from diffuse.diffusion.sde import SDE, SDEState


class EulerState(IntegratorState):
    position: Array
    rng_key: PRNGKeyArray
    t: float
    dt: float


@dataclass
class Euler:
    """Euler deterministic integrator for ODEs"""

    sde: SDE

    def init(
        self, position: Array, rng_key: PRNGKeyArray, t: float, dt: float
    ) -> EulerState:
        """Initialize integrator state with position, timestep and step size"""
        return EulerState(position, rng_key, t, dt)

    def __call__(self, integrator_state: EulerState, score: Callable) -> EulerState:
        """Perform one Euler integration step: dx = drift*dt"""
        position, rng_key, t, dt = integrator_state

        drift = self.sde.reverse_drift_ode(integrator_state, score)
        dx = drift * dt
        _, rng_key_next = jax.random.split(rng_key)
        return EulerState(position + dx, rng_key_next, t + dt, dt)


def next_churn_noise_level(integrator_state: EulerState, stochastic_churn_rate: float, churn_min: float, churn_max: float, sde: SDE) -> float:
    """Compute the next churn noise level"""
    _, _, t_reverse, dt = integrator_state
    t_forward_curr = sde.tf - t_reverse
    n_steps = sde.tf / dt
    churn_rate = jnp.where(
        stochastic_churn_rate / n_steps - jnp.sqrt(2) + 1 > 0,
        jnp.sqrt(2) - 1,
        stochastic_churn_rate / n_steps,
    )
    churn_rate = jnp.where(
        t_forward_curr > churn_min, jnp.where(t_forward_curr < churn_max, churn_rate, 0), 0
    )
    return sde.tf - t_forward_curr * (1 + churn_rate)

def apply_stochastic_churn(integrator_state: EulerState, stochastic_churn_rate: float, churn_min: float, churn_max: float, noise_inflation_factor: float, sde: SDE) -> EulerState:
    """Apply stochastic churn to the sample"""
    position, rng_key, t_reverse, dt = integrator_state
    t_forward_curr = sde.tf - t_reverse
    t_forward_prev = sde.tf - next_churn_noise_level(integrator_state, stochastic_churn_rate, churn_min, churn_max, sde)
    int_b_curr, int_b_prev = (
        sde.beta.integrate(t_forward_curr, sde.beta.t0),
        sde.beta.integrate(t_forward_prev, sde.beta.t0),
    )
    beta_curr, beta_prev = (
        jnp.sqrt(1 - jnp.exp(-int_b_curr)),
        jnp.sqrt(1 - jnp.exp(-int_b_prev)),
    )
    extra_noise_stddev = (
        jnp.sqrt(beta_prev**2 - beta_curr**2) * noise_inflation_factor
    )
    new_position = (
        position + jax.random.normal(rng_key, position.shape) * extra_noise_stddev
    )
    _, rng_key_next = jax.random.split(rng_key)
    return EulerState(
        new_position, rng_key_next, sde.tf - t_forward_prev, dt
    )

@dataclass
class HeunIntegrator:
    """Heun deterministic integrator for ODEs"""
    sde: SDE
    stochastic_churn_rate: float = 0.0
    churn_min: float = 0.05 # in forward time
    churn_max: float = 1.95
    noise_inflation_factor: float = 1.0

    def init(self, position: Array, rng_key: PRNGKeyArray, t: float, dt: float) -> EulerState:
        """Initialize integrator state with position, timestep and step size"""
        return EulerState(position, rng_key, t, dt)

    def __call__(self, integrator_state: EulerState, score: Callable) -> EulerState:
        _, rng_key, t_reverse, _ = integrator_state
        _, rng_key_next = jax.random.split(rng_key)

        _apply_stochastic_churn = partial(apply_stochastic_churn, stochastic_churn_rate=self.stochastic_churn_rate,
                                          churn_min=self.churn_min,
                                          churn_max=self.churn_max,
                                          noise_inflation_factor=self.noise_inflation_factor,
                                          sde=self.sde)
        integrator_state = jax.lax.cond(
            self.stochastic_churn_rate > 0,
            _apply_stochastic_churn,
            lambda x: x,
            integrator_state,
        )
        position_churned, rng_key, t_reverse_churned, dt = integrator_state
        t_reverse_next = jnp.clip(t_reverse + dt, 0, self.sde.tf)
        drift_curr = self.sde.reverse_drift_ode(integrator_state, score)
        integrator_state_next = EulerState(position_churned + drift_curr * (t_reverse_next - t_reverse_churned), rng_key_next, t_reverse_next, dt)

        drift_next = self.sde.reverse_drift_ode(integrator_state_next, score)
        position_next = position_churned + (drift_curr + drift_next) * (t_reverse_next - t_reverse_churned) / 2


        integrator_state_next_tmp = EulerState(position_next, rng_key_next, t_reverse_next, dt)

        next_integrator_state = jax.lax.cond(
            t_reverse_next < 2-1e-2,
            lambda x, y: x,
            lambda x, y: y,
            integrator_state_next_tmp,
            integrator_state_next
        )
        return next_integrator_state


@dataclass
class DPMpp2sIntegrator:
    """DPM++-2S integrator for SDEs"""
    sde: SDE
    stochastic_churn_rate: float = 0.0
    churn_min: float = 0.05
    churn_max: float = 1.95
    noise_inflation_factor: float = 1.0

    def init(
        self, position: Array, rng_key: PRNGKeyArray, t: float, dt: float
    ) -> EulerState:
        """Initialize integrator state with position, timestep and step size"""
        return EulerState(position, rng_key, t, dt)

    def __call__(self, integrator_state: EulerState, score: Callable) -> EulerState:
        """Perform one DPM++-2S integration step"""
        position, _, t_reverse, _ = integrator_state
        t_forward = self.sde.tf - t_reverse

        _apply_stochastic_churn = partial(apply_stochastic_churn, stochastic_churn_rate=self.stochastic_churn_rate,
                                          churn_min=self.churn_min,
                                          churn_max=self.churn_max,
                                          noise_inflation_factor=self.noise_inflation_factor,
                                          sde=self.sde)
        integrator_state = jax.lax.cond(
            self.stochastic_churn_rate > 0,
            _apply_stochastic_churn,
            lambda x: x,
            integrator_state,
        )

        position_churned, rng_key, t_reverse_churned, dt = integrator_state
        t0 = self.sde.beta.t0
        t_forward_churned = self.sde.tf - t_reverse_churned

        t_forward_next = t_forward - dt
        t_backward_next = t_reverse + dt
        t_forward_next = jnp.where(t_forward_next < 1e-5, 1e-5, t_forward_next)
        t_backward_next = jnp.where(t_backward_next > self.sde.tf - 1e-5, self.sde.tf - 1e-5, t_backward_next)
        t_forward_mid = jnp.sqrt(t_forward_churned * t_forward_next)# (t_forward_curr + t_forward_next) / 2

        int_b_churned, int_b_next, int_b_mid = (
            self.sde.beta.integrate(t_forward_churned, t0),
            self.sde.beta.integrate(t_forward_next, t0),
            self.sde.beta.integrate(t_forward_mid, t0),
        )
        alpha_churned, alpha_next, alpha_mid = (
            jnp.exp(-0.5 * int_b_churned),
            jnp.exp(-0.5 * int_b_next),
            jnp.exp(-0.5 * int_b_mid),
        )
        sigma_churned, sigma_next, sigma_mid = (
            jnp.sqrt(1 - jnp.exp(-int_b_churned)),
            jnp.sqrt(1 - jnp.exp(-int_b_next)),
            jnp.sqrt(1 - jnp.exp(-int_b_mid)),
        )

        log_scale_churned, log_scale_next, log_scale_mid = (
            jnp.log(alpha_churned / sigma_churned),
            jnp.log(alpha_next / sigma_next),
            jnp.log(alpha_mid / sigma_mid),
        )

        h = log_scale_next - log_scale_churned
        r = (log_scale_mid - log_scale_churned) / h

        pred_x0_churned = self.sde.tweedie(SDEState(position_churned, t_forward_churned), score).position
        u = (
            sigma_mid / sigma_churned * position_churned
            - alpha_mid * (jnp.exp(-h * r) - 1) * pred_x0_churned
        )

        pred_x0_mid = self.sde.tweedie(SDEState(u, t_forward_mid), score).position
        D = (1 - 1 / (2 * r)) * pred_x0_churned + (
            1 / (2 * r)
        ) * pred_x0_mid

        next_position = (
            sigma_next / sigma_churned * position_churned
            - alpha_next * (jnp.exp(-h) - 1) * D
        )

        _, rng_key_next = jax.random.split(rng_key)
        next_state = EulerState(next_position, rng_key_next, t_backward_next, dt)


        next_state = jax.lax.cond(
            jnp.abs(t_forward - dt) < 1e-5,
            # lambda x, y: self.euler_step(x._replace(t=self.sde.tf - dt - 1e-5), rng_key_next, score),
            lambda x, y: self.tweedie_step(x, rng_key_next, score),
            lambda x, y: y,
            EulerState(position, rng_key_next, t_reverse, dt),
            next_state,
        )
        return next_state

    def tweedie_step(
        self, integrator_state: EulerState, rng_key_next: PRNGKeyArray, score: Callable
    ) -> EulerState:
        """Perform one Tweedie integration step"""
        position, _, t, dt = integrator_state
        # Convert EulerState to SDEState for tweedie
        sde_state = SDEState(position=position, t=self.sde.tf - t+dt)
        # Get result and convert back to EulerState
        result = self.sde.tweedie(sde_state, score)
        return EulerState(position=result.position, rng_key=rng_key_next, t=t + dt, dt=dt)

    def euler_step(
        self, integrator_state: EulerState, rng_key_next: PRNGKeyArray, score: Callable
    ) -> EulerState:
        """Perform one Euler integration step: dx = drift*dt"""
        position, _, t, dt = integrator_state
        drift = self.sde.reverse_drift_ode(integrator_state, score)
        dx = drift * dt
        return EulerState(position + dx, rng_key_next, t + dt, dt)


@dataclass
class DDIMIntegrator:
    """
    Deterministic DDIM integrator that performs reverse process steps using dt.
    The reverse process starts at t=0 (noise at forward time T) and progresses to t=T (data at forward time 0).
    """
    sde: SDE
    eta: float = 0.0  # Controls stochasticity (η=0 for deterministic DDIM)

    def init(
        self,
        position: Array,
        rng_key: PRNGKeyArray,
        t: Array,
    ) -> EulerState:
        """Initialize integrator state with position, time, and timestep"""
        return EulerState(position, rng_key, t, 0)

    def __call__(
        self,
        integrator_state: EulerState,
        score: Callable
    ) -> EulerState:
        """Perform one DDIM step from t to t+dt in the REVERSE process"""
        position, rng_key, t_iter, iteration = integrator_state
        noise_pred = self.sde.score_to_noise(score)

        t_reverse_curr, t_reverse_next = t_iter[iteration], t_iter[iteration+1]

        # Convert reverse time (t) to forward time (T - t_reverse)
        t_forward_curr = self.sde.tf - t_reverse_curr  # Current forward time
        t_forward_next = self.sde.tf - t_reverse_next      # Next forward time

        # Compute cumulative noise schedule (β) integrals for α calculation
        # Corrected integration limits: integrate from 0 to t_forward_curr (not curr_t to 0)
        int_b_curr = self.sde.beta.integrate(t_forward_curr, self.sde.beta.t0).squeeze()
        int_b_next = self.sde.beta.integrate(t_forward_next, self.sde.beta.t0).squeeze()

        # Compute α (signal rate) and β (noise variance) for current/next steps
        alpha_curr = jnp.exp(-0.5 * int_b_curr)        # α_t = exp(-0.5 ∫₀^t β(s) ds)
        alpha_next = jnp.exp(-0.5 * int_b_next)
        beta_curr = 1.0 - alpha_curr**2               # β_t = 1 - α_t² (noise variance)

        # Predict noise using the FORWARD time (t_forward_curr)
        eps = noise_pred(position, t_forward_curr)

        # Estimate x₀ from x_t and predicted noise
        pred_x0 = (position - jnp.sqrt(beta_curr) * eps) / alpha_curr

        # Compute stochasticity term (σ) for next step
        sigma_next = self.eta * jnp.sqrt(1 - alpha_next**2)

        # Sample noise if η > 0 (stochastic DDIM)
        noise = jax.random.normal(rng_key, position.shape)
        _, rng_key_next = jax.random.split(rng_key)

        # DDIM update rule (reverse process)
        position_next = (
            alpha_next * pred_x0 +  # Data estimate term
            jnp.sqrt(1 - alpha_next**2 - sigma_next**2) * eps +  # Direction adjustment
            sigma_next * noise  # Stochastic noise (if η > 0)
        )

        return EulerState(position_next, rng_key_next, t_iter, iteration+1)