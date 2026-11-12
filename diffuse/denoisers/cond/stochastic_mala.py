# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Stochastic adaptive MALA denoiser sampling full noise sequences."""

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import MeasurementState
from diffuse.integrator.base import IntegratorState
from diffuse.integrator.external_noise import EulerMaruyamaExternalNoiseIntegrator
from diffuse.integrator.stochastic import EulerMaruyamaIntegrator
from diffuse.denoisers.cond.base import CondDenoiser
from diffuse.predictor import Predictor


class StochasticMALAState(NamedTuple):
    """State container for stochastic Adaptive MALA."""

    integrator_state: IntegratorState
    log_weights: float
    latent_sequence: Array
    grad_second_moment: Array
    momentum: Array


@dataclass
class StochasticMALADenoiser(CondDenoiser):
    """Adaptive MALA operating on the full stochastic noise sequence."""

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
        if not isinstance(self.integrator, EulerMaruyamaIntegrator):
            raise ValueError("StochasticMALADenoiser requires an EulerMaruyamaIntegrator instance.")
        self._external_integrator = EulerMaruyamaExternalNoiseIntegrator(self.model, self.integrator.timer)

    @property
    def _latent_length(self) -> int:
        return self.integrator.timer.n_steps + 1

    def init(self, position: Array, rng_key: PRNGKeyArray) -> StochasticMALAState:  # type: ignore[override]
        n_particles = position.shape[0]
        latent_key, rng_key_state = jax.random.split(rng_key)
        latent_sequence = jax.random.normal(latent_key, (self._latent_length, *self.x0_shape))
        sample = self._integrate_from_noise_sequence(latent_sequence)
        log_weights = -jnp.log(n_particles)
        zeros = jnp.zeros_like(latent_sequence)
        integrator_state = IntegratorState(sample, rng_key_state, 1)
        return StochasticMALAState(integrator_state, log_weights, latent_sequence, zeros, zeros)

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: StochasticMALAState,
        measurement_state: MeasurementState,
    ) -> StochasticMALAState:  # type: ignore[override]
        latent_sequence = state.latent_sequence

        def log_posterior_fn(latent_seq: Array) -> Tuple[Array, Array]:
            return self._log_posterior_with_sample(latent_seq, measurement_state)

        (log_pi_current, sample_current), grad_current = jax.value_and_grad(log_posterior_fn, has_aux=True)(latent_sequence)

        g_sq = grad_current * grad_current
        grad_second_moment = self.beta * state.grad_second_moment + (1.0 - self.beta) * g_sq
        momentum = self.alpha * state.momentum + (1.0 - self.alpha) * grad_current

        iteration = jnp.maximum(1.0, jnp.asarray(state.integrator_state.step, dtype=latent_sequence.dtype))
        decay_var = jnp.power(iteration, -self.c1)
        decay_mom = jnp.power(iteration, -self.c2)

        preconditioner = 1.0 / (self.delta + decay_var * jnp.sqrt(grad_second_moment + 1e-12))
        drift = grad_current + decay_mom * momentum

        noise_key, accept_key = jax.random.split(rng_key)
        noise = jax.random.normal(noise_key, latent_sequence.shape)

        sqrt_preconditioner = jnp.sqrt(preconditioner)
        mean_forward = latent_sequence + 0.5 * self.epsilon * (preconditioner * drift)
        proposal = mean_forward + jnp.sqrt(self.epsilon) * sqrt_preconditioner * noise

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
        log_q_reverse = self._log_gaussian_diag(latent_sequence, mean_reverse, self.epsilon * preconditioner_prop)

        log_accept = log_pi_proposal + log_q_reverse - log_pi_current - log_q_forward
        accept_prob = jnp.exp(jnp.minimum(0.0, log_accept))
        accept = jax.random.uniform(accept_key) < accept_prob

        new_latent = jnp.where(accept, proposal, latent_sequence)
        new_sample = jnp.where(accept, sample_proposal, sample_current)
        new_grad_second_moment = jnp.where(accept, grad_second_moment_prop, grad_second_moment)
        new_momentum = jnp.where(accept, momentum_prop, momentum)

        # Update log_weights for particle resampling based on measurement likelihood
        # Compute ||y - A(x)||² for the current sample
        y_pred = self.forward_model.apply(new_sample, measurement_state)
        residual = measurement_state.y - y_pred
        residual_squared = jnp.sum(residual * residual)

        # Log-likelihood: log p(y|x) = -||y - Ax||² / (2σ²)
        new_log_weights = -residual_squared / (2 * self.forward_model.std**2)

        integrator_state_next = state.integrator_state._replace(position=new_sample, step=state.integrator_state.step + 1)
        return StochasticMALAState(integrator_state_next, new_log_weights, new_latent, new_grad_second_moment, new_momentum)

    def _integrate_from_noise_sequence(self, latent_sequence: Array) -> Array:
        return self._external_integrator.integrate(latent_sequence, self.predictor)

    def _log_posterior_with_sample(self, latent_sequence: Array, measurement_state: MeasurementState) -> Tuple[Array, Array]:
        """Compute log posterior and generated sample for a latent noise sequence."""
        denoised = self._integrate_from_noise_sequence(latent_sequence)
        y_pred = self.forward_model.apply(denoised, measurement_state)
        residual = measurement_state.y - y_pred
        log_likelihood = -self.data_consistency_weight * jnp.sum(residual * residual)
        log_prior = -0.5 * jnp.sum(latent_sequence * latent_sequence)
        log_posterior = log_likelihood + log_prior
        return log_posterior, denoised

    def resampler(
        self,
        state_next: StochasticMALAState,
        measurement_state: MeasurementState,
        rng_key: PRNGKeyArray,
    ) -> StochasticMALAState:
        """Resample particles based on measurement likelihood.

        Similar to FPS resampler but for Stochastic MALA states. Normalizes log_weights
        across particles and performs stratified resampling based on ESS.

        Args:
            state_next: Current Stochastic MALA state for all particles (batched)
            measurement_state: Measurement information
            rng_key: Random key for resampling

        Returns:
            Resampled Stochastic MALA state with normalized weights
        """
        from diffuse.denoisers.utils import resample_particles, normalize_log_weights

        # state_next.log_weights already contains per-particle log-likelihoods from step()
        # We just need to normalize and potentially resample
        log_weights = normalize_log_weights(state_next.log_weights)

        # Resample particles if ESS is low
        position, log_weights_resampled = resample_particles(
            state_next.integrator_state.position, log_weights, rng_key, self.ess_low, self.ess_high
        )

        # Update integrator state with potentially resampled positions
        integrator_state_next = state_next.integrator_state._replace(position=position)

        # Return state with updated position and normalized log_weights
        return state_next._replace(integrator_state=integrator_state_next, log_weights=log_weights_resampled)

    @staticmethod
    def _log_gaussian_diag(value: Array, mean: Array, cov_diag: Array) -> Array:
        diff = value - mean
        inv_cov = 1.0 / (cov_diag + 1e-12)
        log_det = jnp.sum(jnp.log(cov_diag + 1e-12))
        quad = jnp.sum(diff * diff * inv_cov)
        dim = diff.size
        return -0.5 * (dim * jnp.log(2 * jnp.pi) + log_det + quad)
