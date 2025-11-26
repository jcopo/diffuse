from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from dataclasses import replace

from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.integrator.base import IntegratorState
from diffuse.timer.base import HeunTimer, Timer, VpTimer


@dataclass
class PnPDMDenoiser(CondDenoiser):
    """Plug-and-Play Diffusion Models (PnPDM).

    Alternates between Langevin dynamics for data fitting and full reverse
    diffusion for denoising. Uses exponential decay sigma annealing.

    At each annealing step:
    1. Run Langevin dynamics: minimize ||Ax-y||²/τ² + ||x-x₀||²/σ²
    2. Run full reverse diffusion from σ to eps

    Reference: "Principled Probabilistic Imaging using Diffusion Models as
    Plug-and-Play Priors" (Wu et al.)
    """

    sigma_max: float = 1.0
    sigma_min: float = 0.01
    rho: float = 0.95
    langevin_steps: int = 10
    langevin_lr: float = 0.01
    tau: float = 0.5
    diffusion_steps: int = 50
    annealing_steps: int = 50

    def langevin_sampling(
        self,
        rng_key: PRNGKeyArray,
        x0_hat: Array,
        sigma: Array,
        measurement_state: MeasurementState,
    ) -> Array:
        """Run Langevin dynamics optimizing ||Ax - y||²/τ² + ||x - x0_hat||²/σ²."""

        sigma_sq = sigma * sigma + 1e-8
        tau_sq = self.tau * self.tau + 1e-8

        def langevin_step(x, key):
            y_pred = self.forward_model.apply(x, measurement_state)
            residual = y_pred - measurement_state.y
            grad_likelihood = (
                self.forward_model.adjoint(residual, measurement_state) / tau_sq
            )
            grad_prior = (x - x0_hat) / sigma_sq
            grad_total = grad_likelihood + grad_prior

            noise = jax.random.normal(key, x.shape, dtype=x.dtype)
            x_next = (
                x
                - self.langevin_lr * grad_total
                + jnp.sqrt(2.0 * self.langevin_lr) * noise
            )
            return x_next, None

        keys = jax.random.split(rng_key, self.langevin_steps)
        x_final, _ = jax.lax.scan(langevin_step, x0_hat, keys)
        return x_final

    def reverse_diffuse(self, rng_key: PRNGKeyArray, z: Array, sigma: Array) -> Array:
        """Run full reverse diffusion from sigma level to eps."""
        mini_timer = self._create_timer_from_sigma(sigma)
        mini_integrator = replace(self.integrator, timer=mini_timer)
        mini_state = mini_integrator.init(z, rng_key)

        def ode_step(state: IntegratorState, _):
            state_next = mini_integrator(state, self.predictor)
            return state_next, None

        final_state, _ = jax.lax.scan(
            ode_step, mini_state, xs=None, length=self.diffusion_steps
        )
        return final_state.position

    def _sigma_to_t(self, sigma: Array) -> Array:
        """Convert noise level sigma to time t via binary search."""
        # Binary search to find t where noise_level(t) ≈ sigma
        t_lo = jnp.array(0.0)
        t_hi = jnp.array(1.0)

        def body_fn(carry):
            t_lo, t_hi = carry
            t_mid = (t_lo + t_hi) / 2
            noise_mid = self.model.noise_level(t_mid)
            t_lo = jnp.where(noise_mid < sigma, t_mid, t_lo)
            t_hi = jnp.where(noise_mid >= sigma, t_mid, t_hi)
            return t_lo, t_hi

        t_lo, t_hi = jax.lax.fori_loop(0, 20, lambda _, c: body_fn(c), (t_lo, t_hi))
        return (t_lo + t_hi) / 2

    def _create_timer_from_sigma(self, sigma: Array) -> Timer:
        """Create timer for reverse diffusion from sigma to eps."""
        base_timer = self.integrator.timer
        # Find t where noise_level(t) = sigma
        t_start = self._sigma_to_t(sigma)
        t_start = jnp.clip(t_start, base_timer.eps * 1.5, 1.0 - 1e-4)
        return replace(base_timer, tf=t_start, n_steps=self.diffusion_steps)

    def _add_noise(self, rng_key: PRNGKeyArray, x0: Array, sigma: Array) -> Array:
        """Add noise to clean sample to bring it to noise level sigma."""
        t = self._sigma_to_t(sigma)
        alpha_t = self.model.signal_level(t)
        sigma_t = self.model.noise_level(t)
        noise = jax.random.normal(rng_key, x0.shape, dtype=x0.dtype)
        return alpha_t * x0 + sigma_t * noise

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Single annealing step: Langevin + noise + reverse diffusion."""
        integrator_state = state.integrator_state
        x = integrator_state.position
        step_idx = integrator_state.step

        # Map step_idx to annealing schedule (allows more outer steps than annealing_steps)
        n_outer = self.integrator.timer.n_steps
        annealing_idx = (step_idx * self.annealing_steps) // n_outer
        sigma = jnp.maximum(self.sigma_min, self.sigma_max * (self.rho**annealing_idx))

        key_langevin, key_noise, key_diffuse = jax.random.split(rng_key, 3)

        # Langevin produces clean-ish sample z
        z = self.langevin_sampling(key_langevin, x, sigma, measurement_state)

        # Add noise to bring z to sigma level before reverse diffusion
        z_noisy = self._add_noise(key_noise, z, sigma)

        # Reverse diffusion from sigma level to clean
        x_next = self.reverse_diffuse(key_diffuse, z_noisy, sigma)

        integrator_state_next = IntegratorState(
            position=x_next, rng_key=key_diffuse, step=step_idx + 1
        )
        return state._replace(integrator_state=integrator_state_next)
