from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Tuple
import pdb

import einops
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from blackjax.smc.resampling import stratified
from jax.tree_util import register_pytree_node_class
import matplotlib.pyplot as plt
from jaxtyping import Array, PRNGKeyArray, PyTreeDef

from diffuse.sde import SDE, SDEState, euler_maryama_step
from diffuse.conditional import (
    CondSDE,
    CondState,
    cond_reverse_diffusion,
    cond_reverse_drift,
)


def ess(log_weights: Array) -> float:
    return jnp.exp(log_ess(log_weights))


def log_ess(log_weights: Array) -> float:
    """Compute the effective sample size.

    Parameters
    ----------
    log_weights: 1D Array
        log-weights of the sample

    Returns
    -------
    log_ess: float
        The logarithm of the effective sample size

    """
    return 2 * jsp.special.logsumexp(log_weights) - jsp.special.logsumexp(
        2 * log_weights
    )


def filter_step(
    particles: Array,
    log_Z: float,
    key: PRNGKeyArray,
    u: SDEState,
    u_next: SDEState,
    xi: Array,
    cond_sde: CondSDE,
) -> Tuple[Array, float]:
    # u is time reversed y: u(t) = y(T - t)
    dt = u_next.t - u.t
    # update particles with SDE
    n_particles = particles.shape[0]
    key_sde, key_weights = jax.random.split(key)
    keys = jax.random.split(key_sde, n_particles)
    particles_next = jax.vmap(
        cond_sde.cond_reverse_step, in_axes=(CondState(0, None, None, None), None, 0)
    )(CondState(particles, u.position, xi, u.t), dt, keys).x

    # weights current particles according to likelihood of observation and normalize
    cond_state = CondState(particles, u.position, xi, u.t)
    log_weights = jax.vmap(
        cond_sde.logpdf, in_axes=(None, CondState(0, None, None, None), None)
    )(u_next.position, cond_state, dt)
    _norm = jax.scipy.special.logsumexp(log_weights, axis=0)
    log_weights = log_weights - _norm

    # resample particles according to weights
    # maybe resample based on ESS crit ?
    idx = stratified(key_weights, jnp.exp(log_weights), n_particles)
    ess_val = ess(log_weights)
    # particles_next = jax.lax.cond(ess_val < 0.2 * n_particles, lambda x: x[idx], lambda x: x, particles_next)
    # particles_next = particles_next[idx]

    log_Z = log_Z - jnp.log(n_particles) + _norm

    return particles_next, log_Z


def generate_cond_sample(
    y: Array,
    xi: Array,
    key: PRNGKeyArray,
    cond_sde: CondSDE,
    x_shape: Tuple,
    n_ts: int,
    n_particles: int,
):
    ts = jnp.linspace(0.0, cond_sde.tf, n_ts)
    key_y, key_x = jax.random.split(key)

    # generate path for y
    y = einops.repeat(y, "... -> ts ... ", ts=n_ts)
    state = SDEState(y, jnp.zeros((n_ts, 1)))
    keys = jax.random.split(key_y, n_ts)
    ys = jax.vmap(cond_sde.path, in_axes=(0, 0, 0))(keys, state, ts)

    # u time reversal of y
    us = SDEState(ys.position[::-1], ys.t)

    u_0Tm = jax.tree.map(lambda x: x[:-1], us)
    u_1T = jax.tree.map(lambda x: x[1:], us)

    x_T = jax.random.normal(key_x, (n_particles, *x_shape))

    # scan pcmc over x0 for n_steps
    keys = jax.random.split(key, n_ts - 1)

    def step(state, itr):
        x_p, log_Z_p = state
        key, u, u_next = itr
        n_state = filter_step(x_p, log_Z_p, key, u, u_next, xi, cond_sde)
        return n_state, n_state

    end_state, hist = jax.lax.scan(step, (x_T, 0.0), (keys, u_0Tm, u_1T))

    positions, log_zs = hist
    positions = jnp.concatenate([x_T[None], positions])
    log_zs = jnp.concatenate([jnp.zeros((1,)), log_zs])

    return end_state, (positions, log_zs)
