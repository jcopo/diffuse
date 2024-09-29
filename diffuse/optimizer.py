from functools import partial
from typing import NamedTuple, Tuple
import pdb

import jax
import jax.experimental
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from jaxtyping import Array, PRNGKeyArray
from optax import GradientTransformation
import einops
import matplotlib.pyplot as plt

from diffuse.conditional import CondSDE
from diffuse.sde import SDEState, euler_maryama_step
from diffuse.filter import stratified
from diffuse.inference import (
    calculate_past_contribution_score,
    calculate_drift_expt_post,
    logpdf_change_expected,
    logpdf_change_y,
    particle_step,
    logprob_y,
)


def plotter_line(array):
    total_frames = len(array)

    # Define the fractions
    fractions = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    n = len(fractions)
    # Create a figure with subplots
    fig, axs = plt.subplots(1, n, figsize=(n * 3, n))

    for idx, fraction in enumerate(fractions):
        # Calculate the frame index
        frame_index = int(fraction * total_frames)

        # Plot the image
        axs[idx].imshow(array[frame_index], cmap="gray")
        #axs[idx].set_title(f"Frame at {fraction*100}% of total")
        axs[idx].axis("off")  # Turn off axis labels
    #plt.title("Samples")
    plt.tight_layout()
    plt.show()


def plot_top_samples(thetas, cntrst_thetas, weights, weights_c, past_y, y_c):
    n = 50

    best_idx = jnp.argsort(weights)[-n:][::-1]
    worst_idx = jnp.argsort(weights)[:n]

    best_idx_c = jnp.argsort(weights_c)[-n:][::-1]
    worst_idx_c = jnp.argsort(weights_c)[:n]
    # Create a figure with subplots
    fig, axs = plt.subplots(4, n, figsize=(40, 12))
    fig.suptitle("Theta (top) and Contrastive Theta (bottom) Samples", fontsize=16)

    for idx in range(n):
        axs[0, idx].imshow(thetas[best_idx[idx]], cmap="gray")
        axs[1, idx].imshow(thetas[worst_idx[idx]], cmap="gray")
        axs[2, idx].imshow(cntrst_thetas[best_idx_c[idx]], cmap="gray")
        axs[3, idx].imshow(cntrst_thetas[worst_idx_c[idx]], cmap="gray")
        # set no axis labels
        axs[0, idx].axis("off")
        axs[1, idx].axis("off")
        axs[2, idx].axis("off")
        axs[3, idx].axis("off")

    plt.tight_layout()
    #plt.subplots_adjust(top=0.85, wspace=0.1, hspace=0.3)
    plt.show()

def plot_samples(thetas, cntrst_thetas, weights, weights_c, past_y, y_c):
    total_frames = len(thetas)

    # Define the fractions
    fractions = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    n = len(fractions)

    # Create a figure with subplots
    fig, axs = plt.subplots(2, n+1, figsize=(20, 6))
    fig.suptitle("Theta (top) and Contrastive Theta (bottom) Samples", fontsize=16)

    for idx, fraction in enumerate(fractions):
        # Calculate the frame index
        frame_index = int(fraction * total_frames)
        # plot past_y
        axs[0, 0].imshow(past_y, cmap="gray")
        axs[1, 0].imshow(y_c, cmap="gray")
        # Plot the image
        axs[0, 1+idx].imshow(thetas[frame_index], cmap="gray")
        axs[1, 1+idx].imshow(cntrst_thetas[frame_index], cmap="gray")

        # Set titles
        axs[0, 1+idx].set_title(f"Weight: {weights[frame_index]:.2f}", fontsize=10)
        axs[1, 1+idx].set_title(f"Weight: {weights_c[frame_index]:.2f}", fontsize=10)

        # Turn off axis labels
        axs[0, 1+idx].axis("off")
        axs[1, 1+idx].axis("off")

    # Adjust layout and display
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.1, hspace=0.3)
    plt.show()


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
    # _norm = jax.scipy.special.logsumexp(log_weights, keepdims=True)
    _norm = jax.scipy.special.logsumexp(log_weights, axis=1, keepdims=True)
    log_weights = log_weights - _norm
    log_weights = log_weights - _norm

    weighted_logprobs = jnp.mean(log_weights + logprob_target, axis=1)
    weighted_logprobs = jnp.mean(log_weights + logprob_target, axis=1)

    return (logprob_ref - weighted_logprobs).mean(), y_ref
    # return (logprob_ref - logprob_means).mean(), y_ref


def update_joint(
    sde_state: SDEState,
    ys: Array,
    ys_next: Array,
    key: PRNGKeyArray,
    cond_sde: CondSDE,
    mask_history: Array,
    design: Array,
    dt: float,
):
    r"""
    simulate \theta according to conditional sde:
    \(
    \theta_{t+dt} = \[ -\beta(t) / 2 \theta_t - \beta(t) \nabla_\theta log p(y^t_past | \theta_t, \xi_past) - \beta(t) \nabla_\theta \log p(\theta_t) \]dt + \sqrt(\beta(t) )DWt
    \)
    """
    drift_past = calculate_past_contribution_score(
        cond_sde, sde_state, mask_history, ys
    )
    logpdf = partial(
        logpdf_change_y,
        y_next=ys_next,
        design=mask_history,
        cond_sde=cond_sde,
        dt=dt,
    )
    positions = particle_step(sde_state, key, drift_past, cond_sde, dt, logpdf)

    return positions


def update_expected_posterior(
    cntrst_sde_state: SDEState,
    ys: Array,
    ys_next: Array,
    y_measured: Array,
    key: PRNGKeyArray,
    cond_sde: CondSDE,
    mask_history: Array,
    design: Array,
    dt: float,
):
    r"""
    simulate \theta according to conditional sde for expected posterior:
    .. math::
        $$
        \theta_{t+dt} = \[ -\beta(t) / 2 \theta_t - \beta(t) \nabla_\theta log p(y_past^t | \theta_t, \xi_past) - \beta(t) \sum_1^N \nabla_\theta log p(y_i^t | \theta_t, \xi) / N - \beta(t) \nabla_\theta \log p(\theta_t) \]dt + \sqrt(\beta(t) )DWt
        $$
    """
    drift_past = calculate_past_contribution_score(
        cond_sde, cntrst_sde_state, mask_history, y_measured
    )
    drift_y = calculate_drift_expt_post(cond_sde, cntrst_sde_state, design, ys)
    logpdf = partial(
        logpdf_change_expected,
        y_next=ys_next,
        design=mask_history,
        cond_sde=cond_sde,
        dt=dt,
    )
    positions, weights = particle_step(cntrst_sde_state, key, drift_y + drift_past, cond_sde, dt, logpdf)

    return positions, weights


def calculate_and_apply_gradient(
    thetas: Array,
    cntrst_thetas: Array,
    design: Array,
    cond_sde: CondSDE,
    optx_opt: GradientTransformation,
    opt_state: optax.OptState,
):
    grad_xi_score = jax.grad(information_gain, argnums=2, has_aux=True)
    grad_xi, ys = grad_xi_score(thetas[-1], cntrst_thetas, design, cond_sde)
    updates, new_opt_state = optx_opt.update(grad_xi, opt_state, design)
    new_design = optax.apply_updates(design, updates)

    return new_design, new_opt_state, ys


def impl_step(
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
    Implicit step with parallel update
    """

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
    #jax.experimental.io_callback(plotter_line, None, position[:-1, 0], ordered=True)
    #jax.experimental.io_callback(plotter_line, None, thetas[:, 0], ordered=True)
    thetas = jnp.concatenate([thetas[:2], position[:-1]])
    #jax.experimental.io_callback(plotter_line, None, thetas[:, 0], ordered=True)

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

    #jax.experimental.io_callback(plotter_line, None, thetas[:, 0])
    #jax.experimental.io_callback(plotter_line, None, cntrst_thetas[:, 0])
    # get EIG gradient estimator
    #  1 - evaluate score_f on thetas and contrastives_theta
    #  2 - update design parameters with optax
    design, opt_state, ys = calculate_and_apply_gradient(
        thetas[-1], cntrst_thetas[-1], design, cond_sde, optx_opt, opt_state
    )

    return ImplicitState(thetas, cntrst_thetas, design, opt_state)


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
    thetas, cntrst_thetas, design, opt_state = state
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

    ((thetas, weights), _) = step_joint(sde_state, past_y.position, past_y_next.position, key_joint)

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

    #plot = lambda _: jax.experimental.io_callback(plot_samples, None, thetas.position, cntrst_thetas.position, weights, weights_c, past_y.position, ys_next.mean(axis=0))
    plot = lambda _: jax.experimental.io_callback(plot_top_samples, None, thetas.position, cntrst_thetas.position, weights, weights_c, past_y.position, ys_next.mean(axis=0))

    jax.lax.cond(past_y.t > 1.98, plot, lambda _: None, None)
    # get EIG gradient estimator
    #  1 - evaluate score_f on thetas and contrastives_theta
    #  2 - update design parameters with optax
    n = 50
    best_idx = jnp.argsort(weights)[-n:][::-1]
    best_idx_c = jnp.argsort(weights_c)[-n:][::-1]
    design, opt_state, ys = calculate_and_apply_gradient(
        thetas.position, cntrst_thetas.position, design, cond_sde, optx_opt, opt_state
    )

    return ImplicitState(thetas.position, cntrst_thetas.position, design, opt_state)


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
