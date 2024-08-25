import jax
import jax.numpy as jnp
from jaxtyping import PyTreeDef, PRNGKeyArray, Array
from typing import NamedTuple, Callable, Tuple
from diffuse.sde import SDEState, SDE, euler_maryama_step
from dataclasses import dataclass
from blackjax.smc.resampling import stratified
from functools import partial


from diffuse.images import restore


class CondState(NamedTuple):
    x: Array  # Current Markov Chain State x_t
    y: Array  # Current Observation Path y_t
    xi: Array  # Measured position of y_t | x_t, xi_t
    t: float


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
        """
        x_p, y_p, t_p = state_p
        mean = y_p + cond_reverse_drift(state_p, self) * dt
        std = jnp.sqrt(dt) * cond_reverse_diffusion(state_p, self)

        return jax.scipy.stats.norm.logpdf(obs, mean, std).sum()

    def cond_reverse_step(
        self, state: CondState, dt: float, key: PRNGKeyArray
    ) -> CondState:
        """
        x_{k-1} | x_k, y_k ~ N(.| x_k + rev_drift*dt, sqrt(dt)*rev_diff)
        """
        x, y, t = state
        x = euler_maryama_step(
            SDEState(x, t), dt, key, cond_reverse_drift, cond_reverse_diffusion
        )
        y = self.mask(x)
        return CondState(x, y, t - dt)


def cond_reverse_drift(state: CondState, cond_sde: CondSDE) -> Array:
    # stack together x and y and apply reverse drift
    x, y, xi, t = state
    img = restore(xi, x, cond_sde.mask, y)
    return cond_sde.reverse_drift(SDEState(img, t))


def cond_reverse_diffusion(state: CondState, cond_sde: CondSDE) -> Array:
    # stack together x and y and apply reverse diffusion
    x, y, xi, t = state
    img = restore(xi, x, cond_sde.mask, y)
    return cond_sde.reverse_diffusion(SDEState(img, t))


def pmcmc_step(state, ys, cond_sde: CondSDE):
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
    y, y_p, dt, key = ys
    # weights current particles according to likelihood of observation and normalize
    log_weights = jax.vmap(cond_sde.logpdf, in_axes=(None, 0, None, None))(
        y, particles, y_p, dt
    )
    _norm = jax.scipy.special.logsumexp(log_weights, axis=0)
    log_weights = log_weights - _norm

    # resample particles according to weights
    idx = stratified(jnp.exp(log_weights), key)
    particles = particles[idx]

    # update particles with SDE
    keys = jax.random.split(key, n_particles)
    particles = jax.vmap(cond_sde.cond_reverse_step, in_axes=(0, None, 0))(
        particles, dt, keys
    )

    # update marginal likelihood Z
    log_Z = log_Z - jnp.log(n_particles) + _norm

    return (particles, log_Z), None


def pmcmc(x_p: Array, log_Z_p: float, key: PRNGKeyArray, y: Array, cond_sde: CondSDE):
    ts = jnp.linspace(0.0, cond_sde.tf, 100)
    dts = jnp.diff(ts)
    key_y, key_x = jax.random.split(key)

    # generate path for y
    state = SDEState(y, jnp.zeros(y.shape))
    ys = cond_sde.path(key_y, state, ts)
    # generate initial particles x_T from ref distribution
    x0s = jax.random.normal(key_x, x_p.shape)

    # filter particles x from path of y
    step = partial(pmcmc_step, cond_sde=cond_sde)
    keys = jax.random.split(key, len(dts))
    (particles, log_Z), _ = jax.lax.scan(step, (x0s, 0.0), (ys[:1], ys[1:], dts, keys))

    # accept-reject x
    return jax.lax.cond(
        jnp.exp(log_Z) > jnp.exp(log_Z_p), lambda: (particles, log_Z), lambda: (x_p, log_Z_p)
    ) 


def generate_cond_sample(
    y: Array, key: PRNGKeyArray, n_steps: int, cond_sde: CondSDE, x_shape: Tuple
):
    # start from obervatio y0

    # select starting x0
    x0 = jnp.zeros(x_shape)

    # scan pcmc over x0 for n_steps
    keys = jax.random.split(key, n_steps)

    def step(state, key):
        x_p, log_Z_p = state
        n_state = pmcmc(x_p, log_Z_p, key, y, cond_sde)
        return n_state, n_state

    end_state, hist = jax.lax.scan(step, (x0, 0.0), keys)
    return end_state, hist
