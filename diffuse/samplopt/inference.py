import pdb
from functools import partial
from typing import Callable, Tuple

import einops
import jax
import jax.experimental
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
from blackjax.smc.resampling import stratified
from jaxtyping import Array, PRNGKeyArray
from matplotlib.colors import LogNorm

from diffuse.diffusion.sde import SDEState
from diffuse.utils.plotting import plot_lines


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


def log_density_multivariate_complex_gaussian(x, mean, sq_std):
    diff = x - mean

    quad_form = diff.conj() * diff / sq_std

    log_density = -jnp.log(jnp.pi) - jnp.log(sq_std) - quad_form

    return jnp.real(log_density)


def print_img(x, y, logprob_val, title):
    plt.subplot(1, 2, 1)
    plt.imshow(jnp.abs(x[..., 0]), cmap="gray", norm=LogNorm())
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(jnp.abs(y[..., 0]), cmap="gray", norm=LogNorm())
    plt.colorbar()
    plt.title(f"{title}: logprob: {logprob_val:.2f} hihi")
    plt.show()


def logprob_y(theta, y, design, cond_sde):
    f_y = cond_sde.mask.measure(design, theta)

    log_density = log_density_multivariate_complex_gaussian(y, f_y, 1)
    return log_density


def logpdf_change_theta(
    x_sde_state_next: SDEState, #\theta_{k-1}
    x_sde_state: SDEState, #\theta_{k}
    rev_drift_x: Array,
    y_next: Array,
    design: Array,
    cond_sde: CondSDE,
    dt,
):
    r"""
    log p(theta_{k-1} | theta_{k}, xi_{k-1})
    with theta_{k-1} | theta_{k}, xi_{k-1} ~ N(.| theta_{k} + drift_x*dt, sqrt(dt)*rev_diff)
    """
    NOISE_SCALE = 1.0

    x, t = x_sde_state
    diffusion_unc_theta = cond_sde.reverse_diffusion(x_sde_state)
    next_theta = x_sde_state_next.position

    # define mean, cov of gaussian p(A^T_xi y_{t-1} | y_{t}, \theta_{t-1}, \xi)
    mu_y = cond_sde.mask.restore_from_mask( # A^T_xi A_xi \theta_{t-1}
        design,
        jnp.zeros_like(x),
        cond_sde.mask.measure_from_mask(design, next_theta),
    )
    restored_y = cond_sde.mask.restore_from_mask(design, jnp.zeros_like(x), y_next) # A^T_xi y_{t-1}

    sigma_y = jnp.exp(cond_sde.beta.integrate(0.0, t)) * NOISE_SCALE

    # define mean, cov of gaussian p(theta_{t-1} | theta_{t})
    mu_theta = x + rev_drift_x * dt
    sigma_theta = diffusion_unc_theta * jnp.sqrt(dt)

    # combine gaussians
    sigma = (1 / sigma_y + 1 / sigma_theta) ** -1
    mean = (restored_y / sigma_y + mu_theta / sigma_theta) * sigma

    #jax.experimental.io_callback(plot_lines, None, next_theta[..., 0], t)
    #jax.experimental.io_callback(plot_lines, None, mean[..., 0], t)
    #logprobs = jax.scipy.stats.multivariate_normal.logpdf(next_theta, mean, sigma)
    logprobs = jax.scipy.stats.multivariate_normal.logpdf(restored_y, mu_y, sigma_y)
    logprobs = einops.reduce(logprobs, "t ... -> t ", "sum")

    return logprobs


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
    rev_diff = cond_sde.reverse_diffusion(x_sde_state) + jnp.zeros_like(x[0])
    cov_diff = cond_sde.mask.measure_from_mask(design, rev_diff) * jnp.sqrt(dt)

    cov_diff = cov_diff[..., 0]
    cov_diff_flat, unravel = jax.flatten_util.ravel_pytree(cov_diff)
    y_next = y_next[..., 0]
    # cov = jnp.einsum('ijk,ljk->ilk', cov_diff, cov_diff) + alpha
    cov = cov_diff_flat**2 + alpha
    mean = cond_sde.mask.measure_from_mask(design, x + drift_x * dt)
    mean = mean[..., 0]

    mean = jax.lax.collapse(mean, 1, -1)

    logsprobs = log_density_multivariate_complex_gaussian(y_next, mean, cov)
    # jax.debug.print('{}', idx)

    # logsprobs = cond_sde.mask.measure_from_mask(design, logsprobs)
    pdb.set_trace()
    logsprobs = logsprobs[..., 0] * design[None]
    logsprobs = einops.reduce(logsprobs, "t ... -> t ", "sum")

    id_max = jnp.argmax(logsprobs)
    id_min = jnp.argmin(logsprobs)
    # jax.experimental.io_callback(print_img, None, mean[id_max], y_next, logsprobs[id_max], 1)
    # jax.experimental.io_callback(print_img, None, mean[id_min], y_next, logsprobs[id_min], -1)
    return logsprobs


def _logprob_y(theta, y, design, cond_sde):
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
    sde_state: SDEState,
    rng_key: PRNGKeyArray,
    drift_y: Array,
    cond_sde: CondSDE,
    dt: float,
    logpdf: Callable[[SDEState, Array], Array]
) -> Tuple[Array, Array]:
    """
    Particle step for the conditional diffusion.
    """

    drift_x = cond_sde.reverse_drift(sde_state)
    diffusion = cond_sde.reverse_diffusion(sde_state)
    sde_state_next = euler_maryama_step_array(
        sde_state, dt, rng_key, drift_x + drift_y, diffusion
    )
    # weights = jax.vmap(logpdf, in_axes=(SDEState(0, None),))(sde_state)
    key_pdf, key_resample = jax.random.split(rng_key)
    weights = logpdf(sde_state_next, sde_state, drift_x)
    # jax.debug.print('{}', weights)

    _norm = jax.scipy.special.logsumexp(weights, axis=0)
    log_weights = weights - _norm
    weights = jnp.exp(log_weights)

    ess_val = ess(log_weights)
    n_particles = sde_state.position.shape[0]
    idx = stratified(key_resample, weights, n_particles)

    def print_img(x, id):
        plt.imshow(x[id, ..., 0], cmap="gray")
        plt.show()

    # jax.debug.print('{}', idx)
    # jax.experimental.io_callback(print_img, None, sde_state.position, idx[0])
    # return sde_state.position, weights
    return jax.lax.cond(
        (ess_val < 0.6 * n_particles) & (ess_val > 0.2 * n_particles),
        # (ess_val > 0.2 * n_particles),
        lambda x: (x[idx], weights[idx]),
        lambda x: (x, weights),
        sde_state_next.position,
    )


def _logpdf_change_y(
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
    # jax.experimental.io_callback(plot_lines, None, logsprobs)
    # jax.experimental.io_callback(sigle_plot, None, y_next)
    # logsprobs = jax.vmap(cond_sde.mask.measure, in_axes=(None, 0))(design, logsprobs)
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


def generate_cond_sample(
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
            # logpdf_change_y,
            logpdf_change_theta,
            y_next=ys_next.position,
            design=mask_history,
            cond_sde=cond_sde,
            dt=dt,
        )
        positions, weights = particle_step(
            sde_state, key, drift_past, cond_sde, dt, logpdf
        )
        return SDEState(positions, t + dt), weights

    # generate path for y
    y = einops.repeat(y, "... -> ts ... ", ts=n_ts)
    state = SDEState(y, jnp.zeros((n_ts, 1)))
    keys = jax.random.split(key_y, n_ts)
    ys = jax.vmap(cond_sde.path_cond, in_axes=(None, 0, 0, 0))(
        mask_history, keys, state, ts
    )

    restored_y = jax.vmap(cond_sde.mask.restore_from_mask, in_axes=(None, 0, 0))(
        mask_history, jnp.zeros_like(ys.position), ys.position
    )
    # plot_lines(restored_y[..., 0])
    # plot_lines(jnp.real(ys.position[..., 0]))
    # plot_lines(jnp.imag(ys.position[..., 0]))

    # u time reversal of y
    us = SDEState(ys.position[::-1], ys.t)

    u_0Tm = jax.tree.map(lambda x: x[:-1], us)
    u_1T = jax.tree.map(lambda x: x[1:], us)

    x_T = jax.random.normal(key_x, (n_particles, *x_shape))
    state_x = SDEState(x_T, 0.0)
    weights = jnp.zeros((n_particles,))
    # weights = jnp.zeros((n_particles,), dtype=jnp.complex64)
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
