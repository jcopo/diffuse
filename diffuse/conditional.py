import jax
import jax.numpy as jnp
from jaxtyping import PyTreeDef, PRNGKeyArray, Array
from typing import NamedTuple, Callable
from diffuse.sde import SDEState, SDE, euler_maryama_step
from dataclasses import dataclass
from blackjax.smc.resampling import stratified


class CondState(NamedTuple):
    x: Array # Current Markov Chain State x_0, x_1, ..., x_T
    y: Array # Current Observation Path y_0, y_1, ..., y_T
    t: float


@dataclass
class CondSDE(SDE):
    tf: float
    score: Callable[[Array, float], Array]
    mask: Callable[[Array], Array]

    def reverse_drift(self, state: SDEState) -> Array:
        x, t = state
        beta_t = self.beta(self.tf - t)
        s = self.score(x, self.tf - t)
        return 0.5 * beta_t * x + beta_t * s

    def reverse_diffusion(self, state: SDEState) -> Array:
        x, t = state
        return jnp.sqrt(self.beta(self.tf - t))

    def logpdf(self, obs:Array, state_p:CondState, dt:float):
        """
        y_{k-1} | y_{k}, x_k ~ N(.| y_k + rev_drift*dt, sqrt(dt)*rev_diff)
        """
        x_p, y_p, t_p = state_p
        mean = y_p + cond_reverse_drift(state_p) * dt
        std = jnp.sqrt(dt) * cond_reverse_diffusion(state_p)

        return jax.scipy.stats.norm.logpdf(obs, mean, std).sum()
    
    def cond_reverse_step(self, state: CondState, dt: float, key: PRNGKeyArray) -> CondState:
        """
        x_{k-1} | x_k, y_k ~ N(.| x_k + rev_drift*dt, sqrt(dt)*rev_diff)
        """
        x, y, t = state
        rndm = jax.random.normal(key, x.shape)
        #x = x + self.reverse_drift(SDEState(x, t)) * dt + self.reverse_diffusion(SDEState(x, t)) * rndm
        x = euler_maryama_step(SDEState(x, t), dt, key, cond_reverse_drift, cond_reverse_diffusion)
        y = self.mask(x)
        return CondState(x, y, t - dt)


def cond_reverse_drift(state: CondState) -> Array:
    # stack together x and y and apply reverse drift
    pass


def cond_reverse_diffusion(state: CondState) -> Array:
    # stack together x and y and apply reverse diffusion
    pass
    

def pmcmc_step(particles, y, y_p):
    """
    particles: samples x_k (n_particles, ...)

    returns 
    Samples x_{k-1} (n_particles, ...)
    """

    # weights current particles according to likelihood of observation and normalize
    log_weights = jax.vmap(logpdf, in_axes=(None, 0, None, None))(y, particles, yp, dt)
    _norm = jax.scipy.special.logsumexp(log_weights, axis=0)
    log_weights = log_weights - _norm

    # resample particles according to weights
    idx = stratified(jnp.exp(log_weights), key)
    particles = particles[idx]

    # update particles with SDE
    particles = jax.vmap(cond_reverse_step, in_axes=(0, None, 0))(particles, dt, keys)

    #update marginal likelihood Z
    log_Z = log_Z - jnp.log(nparticles) + _norm


def pmcmc():
    # generate path for y

    # generate initial particles x0 from ref distribution

    # filter particles x from path of y
    # -> scan pmcmc_step over y

    # accept-reject x


def generate_cond_sample():
    # start from obervation y0

    # select starting x0

    # scan pcmc_step over x0 for n_steps