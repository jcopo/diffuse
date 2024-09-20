from functools import partial
from typing import NamedTuple, Tuple
import pdb

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray
from optax import GradientTransformation
import einops

from diffuse.conditional import CondSDE
from diffuse.sde import SDEState, euler_maryama_step
from diffuse.filter import stratified


class ImplicitState(NamedTuple):
    thetas: Array
    cntrst_thetas: Array
    design: Array
    opt_state: optax.OptState


def logprob_y(theta, y, design, cond_sde):
    """
    log p(y | theta, design)
    """
    f_y = cond_sde.mask.measure(design, theta)
    return jax.scipy.stats.norm.logpdf(y, f_y, 1.)


def information_gain(
    theta: Array,
    cntrst_theta: Array,
    design: Array,
    cond_sde
):
    r"""
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
    log_weights = jax.lax.stop_gradient(logprob_target - logprob_means)
    #_norm = jax.scipy.special.logsumexp(log_weights, keepdims=True)
    _norm = jax.scipy.special.logsumexp(log_weights, axis=1, keepdims=True)
    log_weights = log_weights - _norm

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


def calculate_past_contribution_score(cond_sde:CondSDE, sde_state:SDEState, mask_history:Array, y:Array):
    r"""
    Term
    \beta(t)  \nabla_\thetab \log p(\yb^t|\thetab'_t, \xib)
    to add for conditional diffusioon
    """
    x, t = sde_state
    beta_t = cond_sde.beta(cond_sde.tf - t)
    meas_x = cond_sde.mask.measure_from_mask(mask_history, x)
    alpha_t = jnp.exp(cond_sde.beta.integrate(0., t))
    drift_y = beta_t * cond_sde.mask.restore_from_mask(mask_history, jnp.zeros_like(x), (y - meas_x)) / alpha_t
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


def generate_cond_sampleV2(
    y: Array,
    mask_history:Array,
    key: PRNGKeyArray,
    cond_sde: CondSDE,
    x_shape: Tuple,
    n_ts:int,
    n_particles:int
):
    ts = jnp.linspace(0.0, cond_sde.tf, n_ts)
    key_y, key_x = jax.random.split(key)

    def update_joint(sde_state, ys, ys_next, key):
        dt = ys_next.t - ys.t
        drift_past = calculate_past_contribution_score(cond_sde, sde_state, mask_history, ys)
        positions = particle_step(sde_state, key, drift_past, cond_sde, dt, ys, ys_next)
        return positions

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
    state_x = SDEState(x_T, jnp.zeros((n_ts, 1)))
    weights = jnp.zeros((n_particles,))
    # scan pcmc over x0 for n_steps
    keys = jax.random.split(key, n_ts - 1)

    def step(state, itr):
        state_xt, weights = state
        key, u, u_next = itr
        n_state = update_joint(state_xt, u, u_next, key)
        return n_state, (n_state, weights)

    end_state, hist = jax.lax.scan(step, (state_x, weights), (keys, u_0Tm, u_1T))

    positions, log_zs = hist
    positions = jnp.concatenate([x_T[None], positions])
    log_zs = jnp.concatenate([jnp.zeros((1,)), log_zs])


    return end_state, (positions, log_zs)


def impl_step(state:ImplicitState, rng_key: PRNGKeyArray, past_y:Array, mask_history:Array, cond_sde:CondSDE, optx_opt:GradientTransformation, ts:Array, dt:float):

    thetas, cntrst_thetas, design, opt_state = state
    sde_state = SDEState(thetas, ts)
    cntrst_sde_state = SDEState(cntrst_thetas, ts)

    # update joint distribution
    #   1 - conditional sde on joint -> samples theta
    #   2 - get values y from samples of joint
    key_theta, key_cntrst = jax.random.split(rng_key)

    def update_joint(sde_state, ys, ys_next, key):
        drift_past = calculate_past_contribution_score(cond_sde, sde_state, mask_history, ys)
        positions = particle_step(sde_state, key, drift_past, cond_sde, dt, ys, ys_next)
        return positions

    keys_time = jax.random.split(key_theta, ts.shape[0]-1)
    sde_state = jax.tree_map(lambda x: x[1:], sde_state)
    position, weights = jax.vmap(update_joint)(sde_state, past_y.position[:-1], past_y.position[1:], keys_time)
    thetas.at[1:].set(position)
    #pdb.set_trace()

    # get ys
    ys = jax.vmap(cond_sde.mask.measure, in_axes=(None, 0))(design, thetas)
    #keys = jax.random.split(key_y, ts.shape[0])
    #ys = jax.vmap(cond_sde.path, in_axes=(0, 0, 0))(keys, state, ts)

    # update expected posterior
    #   1 - use y values to update post
    #   2 - Get samples contrastive theta
    def update_expected_posterior(cntrst_sde_state, ys, ys_next, key):
        drift_past = calculate_past_contribution_score(cond_sde, cntrst_sde_state, mask_history, ys)
        drift_y = calculate_drift_expt_post(cond_sde, cntrst_sde_state, design, ys) + drift_past
        positions = particle_step(cntrst_sde_state, key, drift_y, cond_sde, dt, ys, ys_next)
        return positions

    keys_time_c = jax.random.split(key_cntrst, ts.shape[0]-1)
    cntrst_sde_state = jax.tree_map(lambda x: x[1:], cntrst_sde_state)
    #pdb.set_trace()
    position, weights = jax.vmap(update_expected_posterior)(cntrst_sde_state, ys[:-1], ys[1:], keys_time_c)
    #d_cntrst_thetas = jax.vmap(update_expected_posterior)(ts, cntrst_thetas, ys, keys_time_c)
    cntrst_thetas.at[1:].set(position)

    # get EIG gradient estimator
    #  1 - evaluate score_f on thetas and contrastives_theta
    #  2 - update design parameters with optax
    grad_xi_score = jax.grad(information_gain, argnums=2, has_aux=True)
    grad_xi, ys = grad_xi_score(
        thetas[-1], cntrst_thetas[-1], design, cond_sde
    )
    updates, opt_state = optx_opt.update(grad_xi, opt_state, design)
    design = optax.apply_updates(design, updates)

    return ImplicitState(thetas, cntrst_thetas, design, opt_state)
