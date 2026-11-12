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
    """Compute Effective Sample Size from log weights.

    ESS = (Σ w_i)² / Σ w_i² where w_i = exp(log_weights_i - max(log_weights))

    Args:
        log_weights: Unnormalized log weights

    Returns:
        Effective sample size (between 1 and n_particles)
    """
    weights = jnp.exp(log_weights - jnp.max(log_weights))
    weights = weights / jnp.sum(weights)
    ess = 1.0 / jnp.sum(weights**2)
    return ess


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

    position, log_weights = resample_particles(
        state_next.integrator_state.position,
        log_weights,
        rng_key,
        ess_low,
        ess_high,
    )
    return position, log_weights


def resample_particles(
    position: Array, log_weights: Array, rng_key: Array, ess_low: float = 0.2, ess_high: float = 0.5
) -> Tuple[Array, Array]:
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
    log_weights_resampled = jnp.zeros(n_particles)

    return jax.lax.cond(
        (ess_val < ess_high * n_particles) & (ess_val > ess_low * n_particles),
        lambda x: (x[idx], log_weights_resampled),
        lambda x: (x, log_weights),
        position,
    )


def resample_particles_block(
    position: Array,
    log_weights: Array,
    rng_key: Array,
    ess_low: float = 0.2,
    ess_high: float = 0.5,
) -> Tuple[Array, Array]:
    """
    Internal function to perform the actual resampling given the weights.

    Particles are first randomly permuted, then resampled within blocks of at most
    four elements so duplicates never spread across the entire population.

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
    jax.debug.print("ess_val: {}", ess_val / n_particles)
    threshold_low = ess_low * n_particles
    threshold_high = ess_high * n_particles
    resample = (ess_val <= threshold_low) & (ess_val < threshold_high)

    def resample_branch(args):
        pos, log_w, w, key = args
        group_size = 4 if n_particles >= 4 else n_particles  # keep resampling local to small blocks
        pad = (group_size - (n_particles % group_size)) % group_size

        key_perm, key_groups_seed = jax.random.split(key)
        perm = jax.random.permutation(key_perm, n_particles)  # shuffle once so groups change every resample
        inv_perm = jnp.argsort(perm)

        pos_perm = pos[perm]
        log_w_perm = log_w[perm]
        w_perm = w[perm]

        # Pad positions and weights to allow grouping while tracking valid entries
        pad_config = [(0, pad)] + [(0, 0)] * (pos.ndim - 1)
        pos_padded = jnp.pad(pos_perm, pad_config, mode="edge")
        log_w_padded = jnp.pad(log_w_perm, (0, pad), constant_values=-jnp.inf)
        weights_padded = jnp.pad(w_perm, (0, pad), constant_values=0.0)
        valid_mask = jnp.concatenate([jnp.ones(n_particles, dtype=bool), jnp.zeros(pad, dtype=bool)])

        num_groups = pos_padded.shape[0] // group_size
        keys_group = jax.random.split(key_groups_seed, num_groups)

        pos_groups = pos_padded.reshape((num_groups, group_size) + pos.shape[1:])
        log_w_groups = log_w_padded.reshape(num_groups, group_size)
        weight_groups = weights_padded.reshape(num_groups, group_size)
        valid_groups = valid_mask.reshape(num_groups, group_size)

        def resample_group(k, pos_g, w_g, logw_g, valid_g):
            valid_count = jnp.maximum(jnp.sum(valid_g), 1)
            w_g = jnp.where(valid_g, w_g, 0.0)
            total = jnp.sum(w_g)
            w_norm = jnp.where(
                total > 1e-8,
                w_g / total,
                jnp.where(valid_g, 1.0 / valid_count, 0.0),
            )

            idx = jax.random.choice(
                k,
                jnp.arange(group_size),
                shape=(group_size,),
                replace=True,  # allow duplicates but only within the small block
                p=w_norm,
            )
            pos_resampled = pos_g[idx]
            valid_shape = (group_size,) + (1,) * (pos_g.ndim - 1)
            pos_resampled = jnp.where(valid_g.reshape(valid_shape), pos_resampled, pos_g)

            uniform_log_weight = -jnp.log(jnp.asarray(n_particles, dtype=logw_g.dtype))
            new_logw = jnp.where(
                valid_g,
                jnp.full_like(logw_g, uniform_log_weight),
                logw_g,
            )
            return pos_resampled, new_logw

        pos_groups_resampled, logw_groups_resampled = jax.vmap(resample_group)(
            keys_group,
            pos_groups,
            weight_groups,
            log_w_groups,
            valid_groups,
        )

        pos_resampled = pos_groups_resampled.reshape((-1,) + pos.shape[1:])[:n_particles]
        logw_resampled = logw_groups_resampled.reshape(-1)[:n_particles]

        pos_out = pos_resampled[inv_perm]
        logw_out = logw_resampled[inv_perm]

        return pos_out, logw_out

    def keep_branch(args):
        pos, log_w, _, _ = args
        return pos, normalize_log_weights(log_w)

    return jax.lax.cond(
        resample,
        resample_branch,
        keep_branch,
        (position, log_weights, weights, rng_key),
    )
