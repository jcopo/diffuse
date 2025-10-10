# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array
from typing import Tuple
import einops

from diffuse.base_forward_model import MeasurementState
from diffuse.utils.mapping import pmapper


def stratified_resampling(key, w):
    N = w.shape[0]
    u = (jnp.arange(N) + jax.random.uniform(key, (N,))) / N
    bins = jnp.cumsum(w)
    idx = jnp.digitize(u, bins)
    return idx


def ess(log_weights: Array) -> float:
    return jnp.exp(log_ess(log_weights))


def log_ess(log_weights: Array) -> float:
    return 2 * jsp.special.logsumexp(log_weights) - jsp.special.logsumexp(2 * log_weights)


def normalize_log_weights(log_weights: Array) -> Array:
    return jax.nn.log_softmax(log_weights, axis=0)


def weights_tweedie(
    state_next,
    measurement_state: MeasurementState,
    rng_key: Array,
    sde,
    score_fn,
    forward_model,
    ess_low: float = 0.2,
    ess_high: float = 0.5,
) -> Tuple[Array, Array]:
    """
    Compute weight with Tweedie's formula and resample particles.

    Args:
        state_next: The current state of the particles
        measurement_state: The measurement state containing observations
        rng_key: Random number generator key
        sde: The SDE object
        score_fn: The score function
        forward_model: The forward model
        ess_low: Lower threshold for ESS (default: 0.2)
        ess_high: Upper threshold for ESS (default: 0.5)

    Returns:
        Tuple of (position, log_weights)
    """
    forward_time = sde.tf - state_next.integrator_state.t
    state_forward = state_next.integrator_state._replace(t=forward_time)

    denoised_state = pmapper(sde.tweedie, state_forward, score=score_fn, batch_size=16)
    diff = forward_model.measure_from_mask(measurement_state.mask_history, denoised_state.position) - measurement_state.y
    abs_diff = jnp.abs(diff[..., 0] + 1j * diff[..., 1])
    log_weights = jax.scipy.stats.norm.logpdf(abs_diff, 0, forward_model.std)
    log_weights = einops.einsum(measurement_state.mask_history, log_weights, "..., b ... -> b")
    _norm = jax.scipy.special.logsumexp(log_weights, axis=0)
    log_weights = log_weights.reshape((-1,)) - _norm

    return resample_particles(state_next.integrator_state.position, log_weights, rng_key, ess_low, ess_high)


def resample_particles(position: Array, log_weights: Array, rng_key: Array, ess_low: float = 0.2, ess_high: float = 0.5) -> Tuple[Array, Array]:
    """
    Internal function to perform the actual resampling given the weights.

    Args:
        position: Current particle positions
        log_weights: Log weights of the particles
        rng_key: Random number generator key
        ess_low: Lower threshold for ESS
        ess_high: Upper threshold for ESS

    Returns:
        Tuple of (resampled_position, normalized_log_weights)
    """
    weights = jax.nn.softmax(log_weights, axis=0)
    ess_val = ess(log_weights)
    n_particles = position.shape[0]
    idx = stratified_resampling(rng_key, weights)

    return jax.lax.cond(
        (ess_val < ess_high * n_particles) & (ess_val > ess_low * n_particles),
        lambda x: (x[idx], normalize_log_weights(log_weights[idx])),
        lambda x: (x, normalize_log_weights(log_weights)),
        position,
    )
