from functools import partial
from typing import NamedTuple

import jax
import jax.experimental
import jax.numpy as jnp

import optax
from jaxtyping import Array, PRNGKeyArray
from optax import GradientTransformation

from diffuse.conditional import CondSDE
from diffuse.sde import SDEState
from diffuse.inference import (
    calculate_past_contribution_score,
    calculate_drift_expt_post,
    logpdf_change_expected,
    logpdf_change_y,
    particle_step,
    logprob_y,
)


class ImplicitState(NamedTuple):
    thetas: Array
    weights: Array
    cntrst_thetas: Array
    weights_c: Array
    design: Array
    opt_state: optax.OptState


def information_gain(theta: Array, cntrst_theta: Array, design: Array, cond_sde):
    r"""
    Information gain estimator
    Estimator \sum_i log p(y_i | theta_i, design) - \sum_j w_{ij} log p(y_i | theta_j, design)
    """
    # sample y from p(y, theta_)
    y_ref = cond_sde.mask.measure(design, theta)
    logprob_ref = logprob_y(theta, y_ref, design, cond_sde)
    logprob_target = jax.vmap(logprob_y, in_axes=(None, 0, None, None))(
        cntrst_theta, y_ref, design, cond_sde
    )
    # logprob_target = jax.scipy.special.logsumexp(logprob_target, )
    logprob_means = jnp.mean(logprob_target, axis=0, keepdims=True)
    logprob_means = jnp.mean(logprob_target, axis=0, keepdims=True)
    log_weights = jax.lax.stop_gradient(logprob_target - logprob_means)
    #_norm = jax.scipy.special.logsumexp(log_weights, keepdims=True)
    _norm = jax.scipy.special.logsumexp(log_weights, axis=1, keepdims=True)
    log_weights = log_weights - _norm
    log_weights = log_weights - _norm

    weighted_logprobs = jnp.mean(log_weights + logprob_target, axis=1)
    weighted_logprobs = jnp.mean(log_weights + logprob_target, axis=1)

    return (logprob_ref - weighted_logprobs).mean(), y_ref
    #return (logprob_ref - logprob_means).mean(), y_ref



def calculate_drift_y(cond_sde:CondSDE, sde_state:SDEState, design:Array, y:Array):
    r"""
    Term
    \beta(t)  \nabla_\thetab \log p(\yb^t|\thetab'_t, \xib)
    to add for conditional diffusioon
    """
    x, t = sde_state
    beta_t = cond_sde.beta(cond_sde.tf - t)
    meas_x = cond_sde.mask.measure(design, x)
    alpha_t = jnp.exp(cond_sde.beta.integrate(0., t))
    drift_y = beta_t * cond_sde.mask.restore(design, jnp.zeros_like(x), (y - meas_x)) / alpha_t
    return drift_y


def calculate_drift_expt_post(cond_sde:CondSDE, sde_state:SDEState, design:Array, y:Array):
    r"""
    Term
    \beta(t)  \sum_{n=1}^N \nu_n \nabla_\thetab \log p(\yb_n^t|\thetab'_t, \xib)
    to add for conditional diffusioon
    """
    #pdb.set_trace()
    drifts = jax.vmap(calculate_drift_y, in_axes=(None, None, None, 0))(cond_sde, sde_state, design, y)
    #drifts = calculate_drift_y(cond_sde, t, xi, x, y)
    drift_y = drifts.mean(axis=0)
    return drift_y


def particle_step(sde_state, rng_key, drift_y, cond_sde, dt, y, ys_next):
    def reverse_drift(state):
        return cond_sde.reverse_drift(state) + drift_y

    sde_state = euler_maryama_step(
        sde_state, dt, rng_key, reverse_drift, cond_sde.reverse_diffusion
    )
    lgpdf = partial(logpdf_change_y, y=y, y_next=ys_next, drift_y=drift_y, cond_sde=cond_sde, dt=dt)
    weights = jax.vmap(lgpdf, in_axes=(SDEState(0, None),))(sde_state)
    _norm = jax.scipy.special.logsumexp(weights, axis=0)
    weights = jnp.exp(weights - _norm)
    idx = stratified(rng_key, weights, sde_state.position.shape[0])
    position = sde_state.position[idx]
    return position, weights[idx]


def logpdf_change_y(x_sde_state:SDEState, y, y_next, drift_y:Array, cond_sde:CondSDE, dt):
    r"""
    log p(y_new | y_old, x_old)
    with y_{k-1} | y_{k}, x_k ~ N(.| y_k + rev_drift*dt, sqrt(dt)*rev_diff)
    """
    cov = cond_sde.reverse_diffusion(x_sde_state) * jnp.sqrt(dt)
    mean = y + (cond_sde.reverse_drift(x_sde_state) + drift_y) * dt
    return jax.scipy.stats.multivariate_normal.logpdf(y_next, mean, cov).sum()


def impl_step(state:ImplicitState, rng_key: PRNGKeyArray, past_y:Array, cond_sde:CondSDE, optx_opt:GradientTransformation, ts:Array, dt:float):

    thetas, cntrst_thetas, design, opt_state = state
    sde_state = SDEState(thetas, ts)
    cntrst_sde_state = SDEState(cntrst_thetas, ts)

    # update joint distribution
    #   1 - conditional sde on joint -> samples theta
    #   2 - get values y from samples of joint
    key_theta, key_cntrst = jax.random.split(rng_key)

    def step_joint(sde_state, ys, ys_next, key):
        positions, weights = update_joint(
            sde_state, ys, ys_next, key, cond_sde, mask_history, design, dt
        )
        return positions, weights

    keys_time = jax.random.split(key_theta, ts.shape[0] - 1)
    sde_state = jax.tree_map(lambda x: x[1:], sde_state)
    position, weights = jax.vmap(step_joint)(
        sde_state, past_y.position[:-1], past_y.position[1:], keys_time
    )
    thetas = jnp.concatenate([thetas[:2], position[:-1]])

    # get ys
    ys = jax.vmap(cond_sde.mask.measure, in_axes=(None, 0))(design, thetas)
    # keys = jax.random.split(key_y, ts.shape[0])
    # ys = jax.vmap(cond_sde.path, in_axes=(0, 0, 0))(keys, state, ts)

    # update expected posterior
    #   1 - use y values to update post
    #   2 - Get samples contrastive theta
    def step_expected_posterior(cntrst_sde_state, ys, ys_next, y_measured, key):
        _, t = cntrst_sde_state
        positions, weights = update_expected_posterior(
            cntrst_sde_state,
            ys,
            ys_next,
            y_measured,
            key,
            cond_sde,
            mask_history,
            design,
            dt,
        )
        return positions, weights

    keys_time_c = jax.random.split(key_cntrst, ts.shape[0] - 1)
    cntrst_sde_state = jax.tree_map(lambda x: x[1:], cntrst_sde_state)
    # pdb.set_trace()
    position, weights_c = jax.vmap(step_expected_posterior)(
        cntrst_sde_state, ys[:-1], ys[1:], past_y.position[1:], keys_time_c
    )
    cntrst_thetas = jnp.concatenate([cntrst_thetas[:2], position[:-1]])

    # get EIG gradient estimator
    #  1 - evaluate score_f on thetas and contrastives_theta
    #  2 - update design parameters with optax
    design, opt_state, ys = calculate_and_apply_gradient(
        thetas[-1], cntrst_thetas[-1], design, cond_sde, optx_opt, opt_state
    )

    return ImplicitState(thetas, weights, cntrst_thetas, weights_c, design, opt_state)


def impl_one_step(
    state: ImplicitState,
    rng_key: PRNGKeyArray,
    past_y: SDEState,
    past_y_next: SDEState,
    mask_history: Array,
    cond_sde: CondSDE,
    optx_opt: GradientTransformation,
):
    """
    Implicit step with one step update
    Must use same optimization steps as time steps
    """
    dt = past_y_next.t - past_y.t
    thetas, weights, cntrst_thetas, weights_c, design, opt_state = state
    sde_state = SDEState(thetas, past_y.t)
    cntrst_sde_state = SDEState(cntrst_thetas, past_y.t)

    # update joint distribution
    #   1 - conditional sde on joint -> samples theta
    #   2 - get values y from samples of joint
    key_joint, key_cntrst = jax.random.split(rng_key, 2)

    def step_joint(sde_state, ys, ys_next, key):
        _, t = sde_state
        positions, weights = update_joint(
            sde_state, ys, ys_next, key, cond_sde, mask_history, design, dt
        )

        return (SDEState(positions, t + dt), weights), (positions, weights)

    ((thetas, weights), _) = step_joint(
        sde_state, past_y.position, past_y_next.position, key_joint
    )

    # get ys
    ys = cond_sde.mask.measure(design, thetas.position)
    ys_next = cond_sde.path(rng_key, SDEState(ys, past_y.t), past_y_next.t).position

    # update expected posterior
    #   1 - use y values to update post
    #   2 - Get samples contrastive theta
    def step_expected_posterior(cntrst_sde_state, ys, ys_next, y_measured, key):
        _, t = cntrst_sde_state
        positions, weights = update_expected_posterior(
            cntrst_sde_state,
            ys,
            ys_next,
            y_measured,
            key,
            cond_sde,
            mask_history,
            design,
            dt,
        )

        return (SDEState(positions, t + dt), weights), (positions, weights)

    ((cntrst_thetas, weights_c), _) = step_expected_posterior(
        cntrst_sde_state, ys, ys_next, past_y.position, key_cntrst
    )

    # get EIG gradient estimator
    #  1 - evaluate score_f on thetas and contrastives_theta
    #  2 - update design parameters with optax

    def step(state, itr):
        design, opt_state = state
        design, opt_state, _ = calculate_and_apply_gradient(
            thetas.position,
            cntrst_thetas.position,
            design,
            cond_sde,
            optx_opt,
            opt_state,
        )
        design = optax.projections.projection_box(design, 0.0, 28.0)
        return (design, opt_state), None

    design, opt_state = jax.lax.cond(
        past_y.t > 1.4,
        lambda _: jax.lax.scan(step, (design, opt_state), jnp.arange(100))[0],
        lambda _: calculate_and_apply_gradient(
            thetas.position,
            cntrst_thetas.position,
            design,
            cond_sde,
            optx_opt,
            opt_state,
        )[:2],
        None,
    )
    # design, opt_state, _ = calculate_and_apply_gradient( thetas.position, cntrst_thetas.position, design, cond_sde, optx_opt, opt_state)

    return ImplicitState(
        thetas.position, weights, cntrst_thetas.position, weights_c, design, opt_state
    )


def impl_full_scan(
    state: ImplicitState,
    rng_key: PRNGKeyArray,
    past_y: Array,
    mask_history: Array,
    cond_sde: CondSDE,
    optx_opt: GradientTransformation,
    ts: Array,
    dt: float,
):
    """
    Implicit step with full scan
    """

    thetas, cntrst_thetas, design, opt_state = state
    n_particles = thetas.shape[0]
    n_cntrst_particles = cntrst_thetas.shape[0]
    sde_state = SDEState(thetas, 0.0)
    cntrst_sde_state = SDEState(cntrst_thetas, 0.0)

    # update joint distribution
    #   1 - conditional sde on joint -> samples theta
    #   2 - get values y from samples of joint
    key_theta, key_cntrst, key_y = jax.random.split(rng_key, 3)

    def step_joint(state, itr):
        sde_state, weights = state
        _, t = sde_state
        ys, ys_next, key = itr
        positions, weights = update_joint(
            sde_state, ys, ys_next, key, cond_sde, mask_history, dt
        )

        return (SDEState(positions, t + dt), weights), (positions, weights)

    keys_time = jax.random.split(key_theta, ts.shape[0] - 1)
    # position, weights = jax.vmap(update_joint)(sde_state, past_y.position[:-1], past_y.position[1:], keys_time)

    ((thetas, _), weights), hist = jax.lax.scan(
        step_joint,
        (sde_state, jnp.zeros((n_particles,))),
        (past_y.position[:-1], past_y.position[1:], keys_time),
    )

    # get ys
    # ys output should be (n_t, n_particles, ...)
    # and time noise -> measurement
    thetas_hist, _ = hist
    thetas_hist = jnp.concatenate([thetas[None], thetas_hist])

    ys = jax.vmap(cond_sde.mask.measure, in_axes=(None, 0))(design, thetas_hist)
    # ys = cond_sde.mask.measure(design, thetas)
    # keys = jax.random.split(key_y, ts.shape[0])
    # ys = jax.vmap(cond_sde.path, in_axes=(0, 0, 0))(keys, state, ts)

    # update expected posterior
    #   1 - use y values to update post
    #   2 - Get samples contrastive theta
    def step_expected_posterior(state, itr):
        cntrst_sde_state, weights = state
        _, t = cntrst_sde_state
        ys, ys_next, y_measured, key = itr
        positions, weights = update_expected_posterior(
            cntrst_sde_state,
            ys,
            ys_next,
            y_measured,
            key,
            cond_sde,
            mask_history,
            design,
            dt,
        )

        return (SDEState(positions, t + dt), weights), (positions, weights)

    keys_time_c = jax.random.split(key_cntrst, ts.shape[0] - 1)

    ((cntrst_thetas, _), weights), hist = jax.lax.scan(
        step_expected_posterior,
        (cntrst_sde_state, jnp.zeros((n_cntrst_particles,))),
        (ys[:-1], ys[1:], past_y.position[1:], keys_time_c),
    )

    # get EIG gradient estimator
    #  1 - evaluate score_f on thetas and contrastives_theta
    #  2 - update design parameters with optax
    design, opt_state, ys = calculate_and_apply_gradient(
        thetas, cntrst_thetas, design, cond_sde, optx_opt, opt_state
    )

    return ImplicitState(thetas, cntrst_thetas, design, opt_state)
