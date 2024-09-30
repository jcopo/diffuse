from functools import partial
from typing import Tuple
import pdb

import jax
import jax.experimental
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, PRNGKeyArray
import einops

from diffuse.conditional import CondSDE
from diffuse.sde import SDEState, euler_maryama_step, euler_maryama_step_array
from diffuse.filter import stratified


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


def logprob_y(theta, y, design, cond_sde):
    """
    log p(y | theta, design)
    """
    f_y = cond_sde.mask.measure(design, theta)
    return jax.scipy.stats.norm.logpdf(y, f_y, 1.0)


def calculate_drift_y(cond_sde: CondSDE, sde_state: SDEState, design: Array, y: Array):
    r"""
    Term
    \beta(t)  \nabla_\thetab \log p(\yb^t|\thetab'_t, \xib)
    to add for conditional diffusioon
    """
    x, t = sde_state
    beta_t = cond_sde.beta(cond_sde.tf - t)
    meas_x = cond_sde.mask.measure(design, x)
    alpha_t = jnp.exp(cond_sde.beta.integrate(0.0, t))
    drift_y = (
        beta_t
        * cond_sde.mask.restore(design, jnp.zeros_like(x), (y - meas_x))
        / alpha_t
    )
    return drift_y


def calculate_past_contribution_score(
    cond_sde: CondSDE, sde_state: SDEState, mask_history: Array, y: Array
):
    r"""
    Term
    \beta(t)  \nabla_\thetab \log p(\yb^t|\thetab'_t, \xib)
    to add for conditional diffusioon

    Only difference from calculate_drift_y is that we use the mask_history to keep track
    of the past measurements and take them into account in the drift.
    """
    x, t = sde_state
    beta_t = cond_sde.beta(cond_sde.tf - t)
    meas_x = cond_sde.mask.measure_from_mask(mask_history, x)
    alpha_t = jnp.exp(cond_sde.beta.integrate(0.0, t))
    drift_y = (
        beta_t
        * cond_sde.mask.restore_from_mask(mask_history, jnp.zeros_like(x), (y - meas_x))
        / alpha_t
    )
    return drift_y


def calculate_drift_expt_post(
    cond_sde: CondSDE, sde_state: SDEState, design: Array, y: Array
):
    r"""
    Term
    \beta(t)  \sum_{n=1}^N \nu_n \nabla_\thetab \log p(\yb_n^t|\thetab'_t, \xib)
    to add for conditional diffusioon
    """
    # pdb.set_trace()
    drifts = jax.vmap(calculate_drift_y, in_axes=(None, None, None, 0))(
        cond_sde, sde_state, design, y
    )
    # drifts = calculate_drift_y(cond_sde, t, xi, x, y)
    drift_y = drifts.mean(axis=0)
    return drift_y


def particle_step(
    sde_state, rng_key, drift_y, cond_sde, dt, logpdf
) -> Tuple[Array, Array]:
    """
    Particle step for the conditional diffusion.
    """

    drift_x = cond_sde.reverse_drift(sde_state)
    diffusion = cond_sde.reverse_diffusion(sde_state)
    sde_state = euler_maryama_step_array(
        sde_state, dt, rng_key, drift_x + drift_y, diffusion
    )
    #weights = jax.vmap(logpdf, in_axes=(SDEState(0, None),))(sde_state)
    weights = logpdf(sde_state, drift_x)

    _norm = jax.scipy.special.logsumexp(weights, axis=0)
    log_weights = weights - _norm
    weights = jnp.exp(log_weights)

    ess_val = ess(log_weights)
    n_particles = sde_state.position.shape[0]
    idx = stratified(rng_key, weights, n_particles)

    #return sde_state.position, weights
    return jax.lax.cond(
        (ess_val < 0.6 * n_particles) & (ess_val > 0.2 * n_particles),
        #(ess_val > 0.2 * n_particles),
        lambda x: (x[idx], weights[idx]),
        lambda x: (x, weights),
        sde_state.position,
    )


def logpdf_change_y(
    x_sde_state: SDEState,
    drift_x: Array,
    y_next: Array,
    design: Array,
    cond_sde: CondSDE,
    dt,
):
    r"""
    log p(y_new | y_old, x_old)
    with y_{k-1} | y_{k}, x_k ~ N(.| y_k + rev_drift*dt, sqrt(dt)*rev_diff)
    """
    x, t = x_sde_state
    alpha = jnp.sqrt(jnp.exp(cond_sde.beta.integrate(0.0, t)))
    cov = cond_sde.reverse_diffusion(x_sde_state) * jnp.sqrt(dt) + alpha

    # mean = cond_sde.mask.measure(design, x + drift_x * dt)
    mean = cond_sde.mask.measure_from_mask(design, x + drift_x * dt)
    logsprobs = jax.scipy.stats.norm.logpdf(y_next, mean, cov)
    logsprobs = cond_sde.mask.measure_from_mask(design, logsprobs)
    #jax.experimental.io_callback(plot_lines, None, logsprobs)
    #jax.experimental.io_callback(sigle_plot, None, y_next)
    #logsprobs = jax.vmap(cond_sde.mask.measure, in_axes=(None, 0))(design, logsprobs)
    logsprobs = einops.reduce(logsprobs, "t ... -> t ", "sum")
    return logsprobs


def logpdf_change_expected(
    x_sde_state: SDEState,
    drift_x: Array,
    y_next: Array,
    design: Array,
    cond_sde: CondSDE,
    dt,
):
    r"""
    \sum log p(y_n | y_old, x_old) / N
    with y_{k-1} | y_{k}, x_k ~ N(.| y_k + rev_drift*dt, sqrt(dt)*rev_diff)
    """
    logpdf = partial(logpdf_change_y, design=design, cond_sde=cond_sde, dt=dt)
    logliks = jax.vmap(logpdf, in_axes=(None, None, 0))(x_sde_state, drift_x, y_next)
    return logliks.mean(axis=0)


def generate_cond_sampleV2(
    y: Array,
    mask_history: Array,
    key: PRNGKeyArray,
    cond_sde: CondSDE,
    x_shape: Tuple,
    n_ts: int,
    n_particles: int,
):
    """
    Generate a conditional sample from the conditional diffusion.
    """
    ts = jnp.linspace(0.0, cond_sde.tf, n_ts)
    key_y, key_x = jax.random.split(key)

    def update_joint(sde_state, ys, ys_next, key):
        _, t = sde_state
        dt = ys_next.t - ys.t
        drift_past = calculate_past_contribution_score(
            cond_sde, sde_state, mask_history, ys.position
        )
        logpdf = partial(
            logpdf_change_y,
            y_next=ys_next.position,
            design=mask_history,
            cond_sde=cond_sde,
            dt=dt,
        )
        positions, weights = particle_step(sde_state, key, drift_past, cond_sde, dt, logpdf)
        return SDEState(positions, t + dt), weights

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
    state_x = SDEState(x_T, 0.0)
    weights = jnp.zeros((n_particles,))
    # scan pcmc over x0 for n_steps
    keys = jax.random.split(key, n_ts - 1)

    def step(state, itr):
        state_xt, weights = state
        key, u, u_next = itr
        keys = jax.random.split(key, n_ts)
        n_state = update_joint(state_xt, u, u_next, key)
        # n_state = jax.vmap(update_joint, in_axes=(SDEState(None, 0), 0, 0, 0))(state_xt, u, u_next, keys)
        return n_state, n_state

    end_state, hist = jax.lax.scan(step, (state_x, weights), (keys, u_0Tm, u_1T))

    (positions, _), weights = hist
    positions = jnp.concatenate([x_T[None], positions])
    weights = jnp.concatenate([jnp.zeros((1, n_particles)), weights])

    return end_state, (positions, weights)
