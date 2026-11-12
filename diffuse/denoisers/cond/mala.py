# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Adaptive MALA conditional denoiser.

Implements Algorithm 2 (MALA with Online Adaptation) tailored for diffusion
inverse problems. The algorithm treats the forward diffusion generator as a
deterministic mapping and samples latent noise vectors that are consistent with
measurements via a preconditioned MALA proposal with Metropolis-Hastings
correction.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import MeasurementState
from diffuse.integrator.base import IntegratorState
from diffuse.denoisers.cond.base import CondDenoiser, CondDenoiserState
from diffuse.predictor import Predictor
from diffuse.denoisers.utils import normalize_log_weights, stratified_resampling, ess


class AdaptiveMALAState(NamedTuple):
    """State container for adaptive MALA denoiser."""

    integrator_state: IntegratorState
    log_weights: float
    latent: Array
    grad_second_moment: Array
    momentum: Array
    # Cached values from previous evaluation to avoid recomputation
    cached_log_posterior: float
    cached_gradient: Array
    cached_sample: Array


@dataclass
class AdaptiveMALADenoiser(CondDenoiser):
    r"""Conditional denoiser implementing adaptive MALA.

    The method augments the unconditional diffusion step with a
    Metropolis-adjusted Langevin proposal that adapts its geometry using
    Adam-style moment estimates. This allows efficient exploration of highly
    anisotropic posterior landscapes encountered in inverse problems.

    Args:
        epsilon: Base step size for the Langevin proposal.
        beta: Decay rate for the squared-gradient accumulator.
        alpha: Decay rate for the momentum accumulator.
        delta: Numerical stability term used in the diagonal preconditioner.
        c1: Exponent for diminishing adaptation in the variance estimate.
        c2: Exponent for diminishing adaptation in the momentum term.
        data_consistency_weight: Optional scaling for measurement loss
            (:math:`\lambda`). Defaults to :math:`1/(2\,\sigma_y^2)` when the
            forward model exposes a noise standard deviation.
    """

    epsilon: float = 1e-3
    beta: float = 0.999
    alpha: float = 0.9
    delta: float = 1e-8
    c1: float = 0.55
    c2: float = 0.45
    data_consistency_weight: Optional[float] = None

    def __post_init__(self):
        if self.data_consistency_weight is None:
            std = getattr(self.forward_model, "std", 1.0)
            self.data_consistency_weight = 0.5 / (std * std + 1e-8)
        self.resample = True
        self.ess_low = 0.01
        self.ess_high = 0.8

    def init(self, position: Array, rng_key: PRNGKeyArray) -> AdaptiveMALAState:  # type: ignore[override]
        n_particles = position.shape[0]
        base_state = super().init(position, rng_key)
        zeros = jnp.zeros_like(position)
        forward_state = self._integrate_from_noise(position, base_state.integrator_state.rng_key)
        forward_state = forward_state._replace(rng_key=base_state.integrator_state.rng_key, step=0)

        # Initialize cache with dummy measurement state (will be properly set in first step)
        # We use zeros for initial cache since we don't have measurement_state yet
        initial_log_posterior = jnp.array(0.0)
        initial_gradient = zeros
        initial_sample = forward_state.position

        return AdaptiveMALAState(
            forward_state, base_state.log_weights, position, zeros, zeros, initial_log_posterior, initial_gradient, initial_sample
        )

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: AdaptiveMALAState,
        measurement_state: MeasurementState,
    ) -> AdaptiveMALAState:  # type: ignore[override]
        integrator_state = state.integrator_state
        latent = state.latent
        rng_forward = integrator_state.rng_key

        def log_posterior_fn(x: Array):
            return self._log_posterior_with_sample(x, measurement_state, rng_forward)

        # Check if this is the first step (cache is uninitialized)
        is_first_step = integrator_state.step == 0

        # Use cached values if available, otherwise compute
        def compute_current():
            (log_post, sample), grad = jax.value_and_grad(log_posterior_fn, has_aux=True)(latent)
            return log_post, grad, sample

        def use_cached():
            return state.cached_log_posterior, state.cached_gradient, state.cached_sample

        log_pi_current, grad_current, sample_current = jax.lax.cond(is_first_step, compute_current, use_cached)

        # Update Adam-style statistics
        g_sq = grad_current * grad_current
        grad_second_moment = self.beta * state.grad_second_moment + (1.0 - self.beta) * g_sq
        momentum = self.alpha * state.momentum + (1.0 - self.alpha) * grad_current

        iteration = jnp.maximum(1.0, jnp.asarray(integrator_state.step + 1, dtype=latent.dtype))
        decay_var = jnp.power(iteration, -self.c1)
        decay_mom = jnp.power(iteration, -self.c2)

        preconditioner = 1.0 / (self.delta + decay_var * jnp.sqrt(grad_second_moment + 1e-12))
        drift = grad_current + decay_mom * momentum

        # Adaptive MALA proposal
        noise_key, accept_key = jax.random.split(rng_key)
        noise = jax.random.normal(noise_key, latent.shape)

        sqrt_preconditioner = jnp.sqrt(preconditioner)
        mean_forward = latent + 0.5 * self.epsilon * (preconditioner * drift)
        proposal = mean_forward + jnp.sqrt(self.epsilon) * sqrt_preconditioner * noise

        # Evaluate proposal statistics for MH correction (ONLY evaluate proposal, not current!)
        (log_pi_proposal, sample_proposal), grad_proposal = jax.value_and_grad(log_posterior_fn, has_aux=True)(proposal)

        iteration_next = iteration + 1.0
        decay_var_next = jnp.power(iteration_next, -self.c1)
        decay_mom_next = jnp.power(iteration_next, -self.c2)

        grad_second_moment_prop = self.beta * grad_second_moment + (1.0 - self.beta) * (grad_proposal * grad_proposal)
        momentum_prop = self.alpha * momentum + (1.0 - self.alpha) * grad_proposal
        preconditioner_prop = 1.0 / (self.delta + decay_var_next * jnp.sqrt(grad_second_moment_prop + 1e-12))
        drift_prop = grad_proposal + decay_mom_next * momentum_prop

        log_q_forward = self._log_gaussian_diag(proposal, mean_forward, self.epsilon * preconditioner)
        mean_reverse = proposal + 0.5 * self.epsilon * (preconditioner_prop * drift_prop)
        log_q_reverse = self._log_gaussian_diag(latent, mean_reverse, self.epsilon * preconditioner_prop)

        log_acceptance = log_pi_proposal + log_q_reverse - log_pi_current - log_q_forward
        accept_prob = jnp.exp(jnp.minimum(0.0, log_acceptance))
        accept = jax.random.uniform(accept_key) < accept_prob

        new_latent = jnp.where(accept, proposal, latent)
        new_sample = jnp.where(accept, sample_proposal, sample_current)
        new_grad_second_moment = jnp.where(accept, grad_second_moment_prop, grad_second_moment)
        new_momentum = jnp.where(accept, momentum_prop, momentum)

        # Cache values for next iteration: if accepted, cache proposal; if rejected, cache current
        new_cached_log_posterior = jnp.where(accept, log_pi_proposal, log_pi_current)
        new_cached_gradient = jnp.where(accept, grad_proposal, grad_current)
        new_cached_sample = new_sample  # This is already the correct one after accept/reject

        # Update log_weights for particle resampling based on measurement likelihood
        # Compute ||y - A(x)||² for the current sample
        y_pred = self.forward_model.apply(new_sample, measurement_state)
        residual = measurement_state.y - y_pred
        residual_squared = jnp.sum(residual * residual)

        # Log-likelihood: log p(y|x) = -||y - Ax||² / (2σ²)
        new_log_weights = -residual_squared / (2 * self.forward_model.std**2)

        integrator_state_next = integrator_state._replace(position=new_sample, step=integrator_state.step + 1)
        return AdaptiveMALAState(
            integrator_state_next,
            new_log_weights,
            new_latent,
            new_grad_second_moment,
            new_momentum,
            new_cached_log_posterior,
            new_cached_gradient,
            new_cached_sample,
        )

    def _log_posterior_with_sample(
        self,
        latent: Array,
        measurement_state: MeasurementState,
        rng_key: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        """Compute log posterior and generated sample for a latent noise vector."""
        generated_state = self._integrate_from_noise(latent, rng_key)
        denoised = generated_state.position
        y_pred = self.forward_model.apply(denoised, measurement_state)
        residual = measurement_state.y - y_pred

        log_likelihood = -self.data_consistency_weight * jnp.sum(residual * residual)
        log_prior = -0.5 * jnp.sum(latent * latent)
        return log_likelihood + log_prior, denoised

    def _integrate_from_noise(self, latent: Array, rng_key: PRNGKeyArray) -> IntegratorState:
        """Run the deterministic diffusion solver starting from latent noise."""

        n_steps = getattr(self.integrator.timer, "n_steps", None)
        if n_steps is None:
            raise ValueError("Adaptive MALA requires integrators with a timer exposing n_steps.")

        integrator_state = IntegratorState(latent, rng_key, 0)
        predictor = Predictor(self.model, self.predictor.velocity, "velocity")

        def step_fn(state, _):
            next_state = self.integrator(state, predictor)
            return next_state, None

        checkpointed_step = jax.checkpoint(step_fn, prevent_cse=False)

        final_state, _ = jax.lax.scan(checkpointed_step, integrator_state, jnp.arange(n_steps))
        return final_state

    def resampler(
        self,
        state_next: AdaptiveMALAState,
        measurement_state: MeasurementState,
        rng_key: PRNGKeyArray,
    ) -> AdaptiveMALAState:
        """Resample particles based on measurement likelihood.

        Similar to FPS resampler but for MALA states. Normalizes log_weights
        across particles and performs stratified resampling based on ESS.

        Args:
            state_next: Current MALA state for all particles (batched)
            measurement_state: Measurement information
            rng_key: Random key for resampling

        Returns:
            Resampled MALA state with normalized weights
        """
        from diffuse.denoisers.utils import resample_particles, normalize_log_weights

        # state_next.log_weights already contains per-particle log-likelihoods from step()
        # We just need to normalize and potentially resample
        log_weights = normalize_log_weights(state_next.log_weights)

        weights = jax.nn.softmax(log_weights, axis=0)
        ess_val = ess(log_weights)
        n_particles = state_next.integrator_state.position.shape[0]

        resample_cond = (ess_val < self.ess_high * n_particles) & (ess_val > self.ess_low * n_particles)

        rng_resample, rng_new_keys = jax.random.split(rng_key)
        indices = stratified_resampling(rng_resample, weights)
        new_keys = jax.random.split(rng_new_keys, n_particles)

        def do_resample(s: AdaptiveMALAState):
            take = lambda x: jnp.take(x, indices, axis=0)

            integrator_state_next = s.integrator_state._replace(
                position=take(s.integrator_state.position),
                rng_key=new_keys,
            )

            log_weights_reset = jnp.zeros_like(log_weights)

            return AdaptiveMALAState(
                integrator_state=integrator_state_next,
                log_weights=log_weights_reset,
                latent=take(s.latent),
                grad_second_moment=take(s.grad_second_moment),
                momentum=take(s.momentum),
                cached_log_posterior=take(s.cached_log_posterior),
                cached_gradient=take(s.cached_gradient),
                cached_sample=take(s.cached_sample),
            )

        return jax.lax.cond(resample_cond, do_resample, lambda s: s, state_next)

    @staticmethod
    def _log_gaussian_diag(value: Array, mean: Array, cov_diag: Array) -> Array:
        """Log-density of a diagonal Gaussian distribution."""

        diff = value - mean
        inv_cov = 1.0 / (cov_diag + 1e-12)
        log_det = jnp.sum(jnp.log(cov_diag + 1e-12))
        quad = jnp.sum(diff * diff * inv_cov)
        dim = diff.size
        return -0.5 * (dim * jnp.log(2 * jnp.pi) + log_det + quad)
