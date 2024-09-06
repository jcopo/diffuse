from functools import partial
from typing import NamedTuple
import pdb

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray
from optax import GradientTransformation

from diffuse.conditional import CondSDE, CondState
from diffuse.filter import filter_step, generate_cond_sample
from diffuse.images import measure
from diffuse.sde import SDEState, euler_maryama_step


class ImplicitState(NamedTuple):
    thetas: Array
    cntrst_thetas: Array
    xi: PRNGKeyArray
    opt_state: optax.OptState


def logprob_y(theta, y, design, cond_sde):
    f_y = measure(design, theta, cond_sde.mask)
    return jax.scipy.stats.norm.logpdf(y, f_y, 1.)

def grad_log_prob(
    rng_key: PRNGKeyArray,
    theta: Array,
    cntrst_theta: Array,
    design: Array,
    cond_sde
):
    # sample y from p(y, theta_)
    y_ref = measure(design, theta, cond_sde.mask)
    logprob_ref = logprob_y(theta, y_ref, design, cond_sde)
    logprob_target = jax.vmap(logprob_y, in_axes=(None, 0, None, None))(
        cntrst_theta, y_ref, design, cond_sde
    )
    # logprob_target = jax.scipy.special.logsumexp(logprob_target, )
    logprob_means = jnp.mean(logprob_target, axis=1, keepdims=True)
    log_weights = jax.lax.stop_gradient(logprob_target - logprob_means)
    _norm = jax.scipy.special.logsumexp(log_weights, axis=1, keepdims=True)
    weights = jnp.exp(log_weights - _norm)

    weighted_logprobs = jnp.mean(weights * logprob_target, axis=1)

    return (logprob_ref - weighted_logprobs).mean(), y_ref


def calculate_drift_y(cond_sde, t, xi, x, y):
    beta_t = cond_sde.beta(cond_sde.tf - t)
    meas_x = measure(xi, x, cond_sde.mask)
    alpha_t = jnp.exp(cond_sde.beta.integrate(0., t))
    drift_y = beta_t * (y - meas_x) / alpha_t
    return drift_y


def calculate_drift_expt_post(cond_sde, t, xi, x, y):
    #pdb.set_trace()
    drifts = jax.vmap(calculate_drift_y, in_axes=(None, None, None, None, 0))(cond_sde, t, xi, x, y)
    #drifts = calculate_drift_y(cond_sde, t, xi, x, y)
    drift_y = drifts.mean(axis=0)
    return drift_y



def particle_step(particles, rng_key, drift_y, cond_sde, dt, t):
    def reverse_drift(state):
        return cond_sde.reverse_drift(state) + drift_y

    particles, _ = euler_maryama_step(
        SDEState(particles, t), dt, rng_key, reverse_drift, cond_sde.reverse_diffusion
    )
    
    return particles


def impl_step(state:ImplicitState, rng_key: PRNGKeyArray, past_y:Array, cond_sde:CondSDE, optx_opt:GradientTransformation, ts:Array, dt:float):

    thetas, cntrst_thetas, design, opt_state = state
    # update joint distribution
    #   1 - conditional sde on joint -> samples theta
    #   2 - get values y from samples of joint
    key_theta, key_y, key_cntrst, key_grad = jax.random.split(rng_key, 4)

    def update_joint(t, thetas, ys, key):
        drift_y = calculate_drift_y(cond_sde, t, design, thetas, ys)
        thetas = particle_step(thetas, key, drift_y, cond_sde, dt, t)
        return thetas

    keys_time = jax.random.split(key_theta, ts.shape[0])
    thetas.at[1:].set(jax.vmap(update_joint)(ts, thetas, past_y.position, keys_time)[:-1])
    #pdb.set_trace()

    # get ys
    ys = jax.vmap(measure, in_axes=(None, 0, None))(design, thetas, cond_sde.mask)
    #keys = jax.random.split(key_y, ts.shape[0])
    #ys = jax.vmap(cond_sde.path, in_axes=(0, 0, 0))(keys, state, ts)
    
    # update expected posterior
    #   1 - use y values to update post
    #   2 - Get samples contrastive theta
    def update_expected_posterior(t, cntrst_thetas, ys, key_cntrst):
        drift_y = calculate_drift_expt_post(cond_sde, t, design, cntrst_thetas, ys)
        cntrst_thetas = particle_step(cntrst_thetas, key_cntrst, drift_y, cond_sde, dt, t)
        return cntrst_thetas

    keys_time_c = jax.random.split(key_cntrst, ts.shape[0])
    #d_cntrst_thetas = jax.vmap(update_expected_posterior)(ts, cntrst_thetas, ys, keys_time_c)
    cntrst_thetas.at[1:].set(jax.vmap(update_expected_posterior)(ts, cntrst_thetas, ys, keys_time_c)[:-1])

    # get EIG gradient estimator
    #  1 - evaluate score_f on thetas and contrastives_theta
    #  2 - update design parameters with optax
    grad_xi_score = jax.grad(grad_log_prob, argnums=3, has_aux=True)
    grad_xi, ys = grad_xi_score(
        key_grad, thetas[-1], cntrst_thetas[-1], design, cond_sde
    )
    updates, opt_state = optx_opt.update(grad_xi, opt_state, design)
    design = optax.apply_updates(design, updates)

    return ImplicitState(thetas, cntrst_thetas, design, opt_state)



    # return implicitstate(thetas, contrastives_theta, xi, ys, opt_state, k + 1), xi