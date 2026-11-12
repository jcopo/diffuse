# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond.base import CondDenoiser
from diffuse.integrator.base import IntegratorState
from diffuse.predictor import Predictor
from diffuse.denoisers.utils import normalize_log_weights, stratified_resampling, ess


class RandomWalkState(NamedTuple):
    """State for random walk noise sampling."""

    integrator_state: IntegratorState
    latent: Array
    log_posterior: Array
    log_weights: Array


@dataclass
class RandomWalkNoiseDenoiser(CondDenoiser):
    """Conditional denoiser using a Gaussian random-walk MH kernel in latent space.

    Proposes new latents z' = z + sigma * N(0, I) and accepts based on
    the measurement likelihood and standard normal prior.
    """

    step_size: float = 1e-3
    data_consistency_weight: Optional[float] = None

    def __post_init__(self):
        if self.data_consistency_weight is None:
            std = getattr(self.forward_model, "std", 1.0)
            self.data_consistency_weight = 0.5 / (std * std + 1e-8)
        self.resample = True
        self.ess_high = 0.8
        self.ess_low = 0.01

    def init(self, position: Array, rng_key: PRNGKeyArray) -> RandomWalkState:  # type: ignore[override]
        """Initialize latent and run a first reverse diffusion."""
        base_state = super().init(position, rng_key)
        latent = position
        integrator_state = self._integrate_from_noise(latent, base_state.integrator_state.rng_key)
        # log_posterior will be computed on first step
        return RandomWalkState(integrator_state, latent, jnp.array(jnp.nan), jnp.array(0.0))

    def _log_posterior(
        self,
        latent: Array,
        sample: Array,
        measurement_state: MeasurementState,
    ) -> Array:
        y_pred = self.forward_model.apply(sample, measurement_state)
        residual = measurement_state.y - y_pred
        log_likelihood = -self.data_consistency_weight * jnp.sum(residual * residual)
        log_prior = -0.5 * jnp.sum(latent * latent)
        return log_likelihood + log_prior

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: RandomWalkState,
        measurement_state: MeasurementState,
    ) -> RandomWalkState:  # type: ignore[override]
        """One random-walk Metropolis step in latent space."""
        rng_prop, rng_accept = jax.random.split(rng_key)

        latent = state.latent
        current_sample = state.integrator_state.position

        log_post_curr = state.log_posterior
        log_post_curr = jax.lax.cond(
            jnp.isnan(log_post_curr),
            lambda _: self._log_posterior(latent, current_sample, measurement_state),
            lambda x: x,
            log_post_curr,
        )

        # Proposal
        noise = jax.random.normal(rng_prop, shape=latent.shape)
        latent_prop = latent + self.step_size * noise
        prop_state = self._integrate_from_noise(latent_prop, rng_prop)
        log_post_prop = self._log_posterior(latent_prop, prop_state.position, measurement_state)

        log_accept = log_post_prop - log_post_curr
        accept = jax.random.uniform(rng_accept) < jnp.exp(jnp.minimum(0.0, log_accept))

        next_latent = jnp.where(accept, latent_prop, latent)
        next_state = jax.tree_util.tree_map(
            lambda a, b: jnp.where(accept, a, b),
            prop_state,
            state.integrator_state,
        )
        next_log_post = jnp.where(accept, log_post_prop, log_post_curr)

        return RandomWalkState(next_state, next_latent, next_log_post, next_log_post)

    def _integrate_from_noise(self, latent: Array, rng_key: PRNGKeyArray) -> IntegratorState:
        """Run the deterministic diffusion solver starting from latent noise."""
        n_steps = getattr(self.integrator.timer, "n_steps", None)
        if n_steps is None:
            raise ValueError("RandomWalkNoiseDenoiser requires integrators with a timer exposing n_steps.")

        integrator_state = IntegratorState(latent, rng_key, 0)
        predictor = Predictor(self.model, self.predictor.velocity, "velocity")

        def body_fn(state, _):
            next_state = self.integrator(state, predictor)
            return next_state, None

        final_state, _ = jax.lax.scan(body_fn, integrator_state, jnp.arange(n_steps))
        return final_state

    def resampler(
        self,
        state_next: RandomWalkState,
        measurement_state: MeasurementState,
        rng_key: PRNGKeyArray,
    ) -> RandomWalkState:
        """Resample particles based on log weights (posterior)."""
        log_weights = normalize_log_weights(state_next.log_weights)

        weights = jax.nn.softmax(log_weights, axis=0)
        ess_val = ess(log_weights)
        n_particles = state_next.integrator_state.position.shape[0]

        resample_cond = (ess_val < self.ess_high * n_particles) & (ess_val > self.ess_low * n_particles)

        rng_resample, rng_new_keys = jax.random.split(rng_key)
        indices = stratified_resampling(rng_resample, weights)
        new_keys = jax.random.split(rng_new_keys, n_particles)

        def do_resample(s: RandomWalkState):
            def take(x):
                return jnp.take(x, indices, axis=0)

            integrator_state_next = s.integrator_state._replace(
                position=take(s.integrator_state.position),
                rng_key=new_keys,
            )

            log_weights_reset = jnp.zeros_like(log_weights)

            return RandomWalkState(
                integrator_state=integrator_state_next,
                latent=take(s.latent),
                log_posterior=take(s.log_posterior),
                log_weights=log_weights_reset,
            )

        return jax.lax.cond(resample_cond, do_resample, lambda s: s, state_next)
