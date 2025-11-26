from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.integrator.base import IntegratorState
from diffuse.timer.base import HeunTimer, Timer, VpTimer


@dataclass
class DAPSDenoiser(CondDenoiser):
    """Decoupled Annealing Posterior Sampling (DAPS).

    At each annealing step:
    1. Reverse diffusion: ODE solve from σ_k to ~0 to get x̂₀
    2. Langevin dynamics: sample from p(x₀|y) ∝ exp(-||Ax-y||²/2τ² - ||x-x̂₀||²/2σ²)
    3. Forward diffusion: add noise to reach σ_{k+1}

    Args:
        langevin_steps: Number of Langevin MCMC steps per annealing level
        langevin_lr: Langevin step size
        langevin_lr_min_ratio: Final/initial step size ratio for annealing
        tau: Likelihood temperature (measurement noise scale)
        diffusion_steps: ODE steps for reverse diffusion

    Reference: Zhang et al., "Improving Diffusion Inverse Problem Solving
    with Decoupled Noise Annealing"
    """

    langevin_steps: int = 100
    langevin_lr: float = 1e-4
    langevin_lr_min_ratio: float = 0.01
    tau: float = 0.01
    diffusion_steps: int = 50

    def reverse_diffuse(self, rng_key: PRNGKeyArray, x_t: Array, sigma: Array) -> Array:
        from diffuse.integrator.deterministic import EulerIntegrator

        # setup mini-diffusion from σ to ~0
        mini_timer = self._create_timer_from_sigma(sigma)
        mini_integrator = EulerIntegrator(model=self.model, timer=mini_timer)
        mini_state = mini_integrator.init(x_t, rng_key)

        def ode_step(state: IntegratorState, _):
            return mini_integrator(state, self.predictor), None

        # ODE solve
        final_state, _ = jax.lax.scan(ode_step, mini_state, xs=None, length=self.diffusion_steps)
        return final_state.position

    def langevin_sampling(
        self,
        rng_key: PRNGKeyArray,
        x0_hat: Array,
        sigma: Array,
        ratio: Array,
        measurement_state: MeasurementState,
    ) -> Array:
        if self.langevin_steps == 0:
            return x0_hat

        # precompute constants
        tau_sq = 2.0 * self.tau * self.tau + 1e-8
        sigma_sq = sigma * sigma + 1e-8
        lr = self.langevin_lr * (1.0 + ratio * (self.langevin_lr_min_ratio - 1.0))

        # ULA step: x ← x - lr·∇U + √(2lr)·z
        def langevin_step(x, key):
            y_pred = self.forward_model.apply(x, measurement_state)
            residual = y_pred - measurement_state.y
            grad_likelihood = self.forward_model.adjoint(residual, measurement_state) / tau_sq
            grad_prior = (x - x0_hat) / sigma_sq
            noise = jax.random.normal(key, x.shape, dtype=x.dtype)
            return x - lr * (grad_likelihood + grad_prior) + jnp.sqrt(2.0 * lr) * noise, None

        keys = jax.random.split(rng_key, self.langevin_steps)
        x_final, _ = jax.lax.scan(langevin_step, x0_hat, keys)
        return x_final

    def forward_diffuse(self, rng_key: PRNGKeyArray, x0: Array, sigma_next: Array) -> Array:
        t_next = self._sigma_to_t(sigma_next)
        alpha_next = self.model.signal_level(t_next)
        sigma_t = self.model.noise_level(t_next)
        noise = jax.random.normal(rng_key, x0.shape, dtype=x0.dtype)
        return alpha_next * x0 + sigma_t * noise

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        x_t = state.integrator_state.position
        step_idx = state.integrator_state.step

        t_current = self.integrator.timer(step_idx)
        t_next = self.integrator.timer(step_idx + 1)
        sigma_current = self.model.noise_level(t_current)
        sigma_next = self.model.noise_level(t_next)
        ratio = step_idx / self.integrator.timer.n_steps

        key_diffuse, key_langevin, key_forward = jax.random.split(rng_key, 3)

        # reverse → langevin → forward
        x0_hat = self.reverse_diffuse(key_diffuse, x_t, sigma_current)
        x0_sample = self.langevin_sampling(key_langevin, x0_hat, sigma_current, ratio, measurement_state)
        x_next = self.forward_diffuse(key_forward, x0_sample, sigma_next)

        integrator_state_next = IntegratorState(position=x_next, rng_key=key_forward, step=step_idx + 1)
        return state._replace(integrator_state=integrator_state_next)

    def _sigma_to_t(self, sigma: Array) -> Array:
        # binary search to invert noise_level(t) = σ
        t_lo, t_hi = jnp.array(0.0), jnp.array(1.0)

        def body_fn(carry):
            t_lo, t_hi = carry
            t_mid = (t_lo + t_hi) / 2
            noise_mid = self.model.noise_level(t_mid)
            return jnp.where(noise_mid < sigma, t_mid, t_lo), jnp.where(noise_mid >= sigma, t_mid, t_hi)

        t_lo, t_hi = jax.lax.fori_loop(0, 20, lambda _, c: body_fn(c), (t_lo, t_hi))
        return (t_lo + t_hi) / 2

    def _create_timer_from_sigma(self, sigma: Array) -> Timer:
        base_timer = self.integrator.timer
        if isinstance(base_timer, VpTimer):
            t_start = self._sigma_to_t(sigma)
            t_start = jnp.clip(t_start, base_timer.eps * 1.5, 1.0 - 1e-4)
            return VpTimer(n_steps=self.diffusion_steps, eps=base_timer.eps, tf=t_start)
        if isinstance(base_timer, HeunTimer):
            sigma_start = jnp.maximum(sigma, base_timer.sigma_min * 1.5)
            return HeunTimer(
                n_steps=self.diffusion_steps,
                rho=base_timer.rho,
                sigma_min=base_timer.sigma_min,
                sigma_max=sigma_start,
            )
        raise NotImplementedError(f"Timer not implemented for {type(base_timer)}")
