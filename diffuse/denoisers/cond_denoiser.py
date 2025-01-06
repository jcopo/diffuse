from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from blackjax.smc.resampling import stratified

from diffuse.integrator.base import Integrator, IntegratorState
from diffuse.diffusion.sde import SDE, SDEState
from diffuse.base_forward_model import ForwardModel


class CondDenoiserState(NamedTuple):
    integrator_state: IntegratorState
    weights: Array


@dataclass
class CondDenoiser:
    """Conditional denoiser for conditional diffusion"""

    integrator: Integrator
    # logpdf: Callable[[SDEState, Array], Array]  # x -> t -> logpdf(x, t)
    sde: SDE
    score: Callable[[Array, float], Array]  # x -> t -> score(x, t)
    forward_model: ForwardModel
    _resample: bool = False

    def init(
        self, position: Array, rng_key: PRNGKeyArray, dt: float
    ) -> CondDenoiserState:
        n_particles = position.shape[0]
        weights = jnp.ones(n_particles) / n_particles
        keys = jax.random.split(rng_key, n_particles)
        integrator_state = self.integrator.init(
            position, keys, jnp.array(0.0), jnp.array(dt)
        )
        return CondDenoiserState(integrator_state, weights)

    def step(
        self, state: CondDenoiserState, score: Callable[[Array, float], Array]
    ) -> CondDenoiserState:
        r"""
        sample p(\theta_t-1 | \theta_t, \y_t-1, \xi)
        """
        integrator_state, weights = state
        integrator_state_next = self.integrator(integrator_state, score)

        position = integrator_state_next.position
        # if self._resample:
        #     weights = self.logpdf(integrator_state_next, integrator_state)
        #     position, weights = self._resampling(position, weights, rng_key)

        return CondDenoiserState(integrator_state_next, weights)

    def posterior_logpdf(
        self, rng_key: PRNGKeyArray, t: float, y_meas: Array, design_mask: Array
    ):
        tf = self.sde.tf
        y_t = self.y_noiser(
            design_mask, rng_key, SDEState(y_meas, 0), tf - t
        ).position

        # will be called backward in time
        # with t = tf - t, t from 0 to tf
        def _posterior_logpdf(x, t):
            tf = self.sde.tf
            alpha_t = jnp.exp(self.sde.beta.integrate(0.0, tf - t))
            #guidance = jax.grad(self.forward_model.logprob_y)(x, y_t, design) #/ alpha_t
            guidance = self.forward_model.grad_logprob_y(x, y_t, design_mask) / alpha_t
            return guidance + self.score(x, t)

        return _posterior_logpdf

    def pooled_posterior_logpdf(
        self,
        rng_key: PRNGKeyArray,
        t: float,
        y_cntrst: Array,
        y_past: Array,
        design: Array,
        mask_history: Array,
    ):
        rng_key1, rng_key2 = jax.random.split(rng_key)
        vec_noiser = jax.vmap(
            self.y_noiser, in_axes=(None, None, SDEState(0, None), None)
        )
        y_t = vec_noiser(
            self.forward_model.make(design), rng_key1, SDEState(y_cntrst, 0), t
        ).position
        mask = self.forward_model.make(design)

        # will be called backward in time
        # with t = tf - t, t from 0 to tf
        def _pooled_posterior_logpdf(x, t):
            tf = self.sde.tf
            alpha_t = jnp.exp(self.sde.beta.integrate(0.0, tf - t))
            guidance = jax.vmap(
                #jax.grad(self.forward_model.logprob_y), in_axes=(None, 0, None)
                self.forward_model.grad_logprob_y, in_axes=(None, 0, None)
            )(x, y_t, mask) / alpha_t
            past_contribution = self.posterior_logpdf(rng_key2, t, y_past, mask_history)
            # import pdb; pdb.set_trace()
            # jax.debug.print("guidance: {}", guidance)
            return guidance.mean(axis=0) + past_contribution(x, t)

        return _pooled_posterior_logpdf

    def y_noiser(
        self, mask: Array, key: PRNGKeyArray, state: SDEState, ts: float
    ) -> SDEState:
        r"""
        Generate y^{(t)} = \sqrt{\bar{\alpha}_t} y + \sqrt{1-\bar{\alpha}_t} A_\xi \epsilon
        """
        y, t = state

        int_b = self.sde.beta.integrate(ts, t)
        alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)

        rndm = jax.random.normal(key, y.shape)
        res = alpha * y + jnp.sqrt(beta) * self.forward_model.measure_from_mask(
            mask, rndm
        )
        return SDEState(res, ts)

    def _resampling(
        self, position: Array, weights: Array, rng_key: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        """Resample particles based on weights if effective sample size is in target range"""
        _norm = jax.scipy.special.logsumexp(weights, axis=0)
        log_weights = weights - _norm
        weights = jnp.exp(log_weights)

        ess_val = ess(log_weights)
        n_particles = position.shape[0]
        key_resample = jax.random.split(rng_key)[1]
        idx = stratified(key_resample, weights, n_particles)

        return jax.lax.cond(
            (ess_val < 0.6 * n_particles) & (ess_val > 0.2 * n_particles),
            lambda x: (x[idx], weights[idx]),
            lambda x: (x, weights),
            position,
        )
