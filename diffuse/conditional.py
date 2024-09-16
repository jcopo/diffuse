from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Tuple
import pdb

import einops
import jax
import jax.numpy as jnp
from blackjax.smc.resampling import stratified
from jax.tree_util import register_pytree_node_class
import matplotlib.pyplot as plt
from jaxtyping import Array, PRNGKeyArray, PyTreeDef

from diffuse.sde import SDE, SDEState, euler_maryama_step


def plt_fracts(array):
    # Define the fractions
    fractions = [0.0, 0.1, 0.9, 0.95, 1.0]
    total_frames = array.shape[0]

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))

    for idx, fraction in enumerate(fractions):
        # Calculate the frame index
        frame_index = int(fraction * total_frames)

        # Plot the image
        axs[idx].imshow(array[frame_index], cmap="gray")
        axs[idx].set_title(f"Frame at {fraction*100}% of total")
        axs[idx].axis("off")  # Turn off axis labels

    plt.tight_layout()
    plt.show()


@register_pytree_node_class
class CondState(NamedTuple):
    x: jnp.ndarray  # Current Markov Chain State x_t
    y: jnp.ndarray  # Current Observation Path y_t
    xi: jnp.ndarray  # Measured position of y_t | x_t, xi_t
    t: float

    def tree_flatten(self):
        children = (self.x, self.y, self.xi, self.t)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@dataclass
class CondSDE(SDE):
    mask: Callable[[Array], Array]
    tf: float
    score: Callable[[Array, float], Array]

    def reverse_drift(self, state: SDEState) -> Array:
        x, t = state
        beta_t = self.beta(self.tf - t)
        s = self.score(x, self.tf - t)
        return 0.5 * beta_t * x + beta_t * s

    def reverse_diffusion(self, state: SDEState) -> Array:
        x, t = state
        return jnp.sqrt(self.beta(self.tf - t))

    def logpdf(self, obs: Array, state_p: CondState, dt: float):
        """
        y_{k-1} | y_{k}, x_k ~ N(.| y_k + rev_drift*dt, sqrt(dt)*rev_diff)
        Args:
            obs (Array): The observation y_{k-1}.
            state_p (CondState): The previous state containing:
                - x_p (Array): The particle state x_k.
                - y_p (Array): The observation y_k.
                - xi_p (Array): The measurement position.
                - t_p (float): The time step.
            dt (float): The time step size.

        Returns:
            float: The log probability density of the observation.
        """
        x_p, y_p, xi, t_p = state_p
        # mean = y_p + cond_reverse_drift(state_p, self) * dt
        mean = y_p + self.mask.measure(xi, cond_reverse_drift(state_p, self)) * dt
        std = jnp.sqrt(dt) * cond_reverse_diffusion(state_p, self)

        return jax.scipy.stats.norm.logpdf(obs, mean, std).sum()

    def cond_reverse_step(
        self, state: CondState, dt: float, key: PRNGKeyArray
    ) -> CondState:
        """
        x_{k-1} | x_k, y_k ~ N(.| x_k + rev_drift*dt, sqrt(dt)*rev_diff)
        """
        x, y, xi, t = state

        def revese_drift(state):
            x, t = state
            return cond_reverse_drift(CondState(x, y, xi, t), self)

        def reverse_diffusion(state):
            x, t = state
            return cond_reverse_diffusion(CondState(x, y, xi, t), self)

        # jax.debug.print("meas{}\n", measure(xi, img, self.mask))
        # jax.debug.print("y{}\n", y.shape)
        # jax.debug.print("diff{}\n", measure(xi, img, self.mask) - y )

        x, _ = euler_maryama_step(
            SDEState(x, t), dt, key, revese_drift, reverse_diffusion
        )
        y = self.mask.measure(xi, x)
        return CondState(x, y, xi, t - dt)


def cond_reverse_drift(state: CondState, cond_sde: CondSDE) -> Array:
    # stack together x and y and apply reverse drift
    x, y, xi, t = state
    img = cond_sde.mask.restore(xi, x, cond_sde.mask, y)
    return cond_sde.reverse_drift(SDEState(img, t))
    # drift_x = cond_sde.reverse_drift(SDEState(x, t))
    # beta_t = cond_sde.beta(cond_sde.tf - t)
    # meas_x = cond_sde.mask.measure(xi, x)
    # alpha_t = jnp.exp(cond_sde.beta.integrate(0.0, t))
    # here if needed we average over y
    # drift_y = beta_t * (y - meas_x) / alpha_t
    # f = lambda y: beta_t * (y - meas_x) / alpha_t
    # drifts = jax.vmap(f)(y)
    # drift_y = drifts.mean(axis=0)
    # return drift_x + drift_y


def cond_reverse_diffusion(state: CondState, cond_sde: CondSDE) -> Array:
    # stack together x and y and apply reverse diffusion
    x, y, xi, t = state
    img = cond_sde.mask.restore(xi, x, y)
    return cond_sde.reverse_diffusion(SDEState(img, t))


def pmcmc_step(state, ys, xi: Array, cond_sde: CondSDE):
    """
    Performs a single step of the Particle Markov Chain Monte Carlo (PMCMC) algorithm.

    Args:
        state (Tuple[Array, float]): A tuple containing:
            - particles (Array): Current particle states x_k. Shape: (n_particles, ...)
            - log_Z (float): Current log marginal likelihood
        ys (Tuple[Array, Array]): Tuple containing the current observation y and the next observation y_p.
        cond_sde (CondSDE): The conditional SDE object.

    Returns:
        Tuple[Tuple[Array, float], None]: A tuple containing:
            - A tuple with:
                - Updated particles x_{k-1}. Shape: (n_particles, ...)
                - Updated log marginal likelihood (log_Z)
            - None (for compatibility with jax.lax.scan)
    """
    particles, log_Z = state
    n_particles = particles.shape[0]
    u, u_next, dt, key = ys

    # update particles with SDE
    keys = jax.random.split(key, n_particles)
    particles = jax.vmap(
        cond_sde.cond_reverse_step, in_axes=(CondState(0, None, None, None), None, 0)
    )(CondState(particles, u.position, xi, u.t), dt, keys).x

    # weights current particles according to likelihood of observation and normalize
    cond_state = CondState(particles, u.position, xi, u.t)
    log_weights = jax.vmap(
        cond_sde.logpdf, in_axes=(None, CondState(0, None, None, None), None)
    )(u_next.position, cond_state, dt)
    # jax.debug.print("{}", log_weights)
    _norm = jax.scipy.special.logsumexp(log_weights, axis=0)
    log_weights = log_weights - _norm

    # resample particles according to weights
    idx = stratified(key, jnp.exp(log_weights), n_particles)
    # jax.debug.print("{}", idx)
    # particles = particles[idx]

    # update marginal likelihood Z
    log_Z = log_Z - jnp.log(n_particles) + _norm

    return (particles, log_Z), None


def pmcmc(
    x_p: Array,
    log_Z_p: float,
    key: PRNGKeyArray,
    y: Array,
    xi: Array,
    cond_sde: CondSDE,
):
    n_ts = 300
    ts = jnp.linspace(0.0, cond_sde.tf, n_ts)
    dts = jnp.diff(ts)
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
    # generate initial particles x_T from ref distribution
    x0s = jax.random.normal(key_x, x_p.shape)

    # filter particles x from path of y
    step = partial(pmcmc_step, cond_sde=cond_sde, xi=xi)
    keys = jax.random.split(key, len(dts))
    (particles, log_Z), _ = jax.lax.scan(step, (x0s, 0.0), (u_0Tm, u_1T, dts, keys))

    # accept-reject x
    prob_accept = jnp.minimum(0.0, log_Z - log_Z_p)
    return jax.lax.cond(
        jax.random.uniform(key) < jnp.exp(prob_accept),
        lambda: (x_p, log_Z_p),
        lambda: (particles, log_Z),
    )


def generate_cond_sample(
    y: Array,
    xi: Array,
    key: PRNGKeyArray,
    n_steps: int,
    cond_sde: CondSDE,
    x_shape: Tuple,
):
    # select starting x0
    n_particles = 100
    x0 = jnp.zeros((n_particles, *x_shape))

    # scan pcmc over x0 for n_steps
    keys = jax.random.split(key, n_steps)

    def step(state, key):
        x_p, log_Z_p = state
        n_state = pmcmc(x_p, log_Z_p, key, y, xi, cond_sde)
        return n_state, n_state

    end_state, hist = jax.lax.scan(step, (x0, 0.0), keys)
    return end_state, hist
