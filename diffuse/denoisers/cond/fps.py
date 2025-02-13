from dataclasses import dataclass
from typing import Tuple

import einops
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState


@dataclass
class FPSDenoiser(CondDenoiser):
    """Filtering Posterior Sampling Denoiser https://openreview.net/pdf?id=tplXNcHZs1"""

    def resampler(
        self,
        state_next: CondDenoiserState,
        measurement_state: MeasurementState,
        rng_key: PRNGKeyArray,
    ) -> Tuple[Array, Array]:
        mask = measurement_state.mask_history
        y = measurement_state.y
        t = state_next.integrator_state.t
        tf = self.sde.tf

        position = state_next.integrator_state.position
        y_noised = self.y_noiser(mask, rng_key, SDEState(y, 0), tf - t).position
        alpha_t = jnp.exp(self.sde.beta.integrate(0.0, tf - t))
        logsprobs = self.forward_model.logprob_y_t(position, y_noised, mask, alpha_t)

        position, log_weights = self._resample(position, logsprobs, rng_key)

        integrator_state_next = state_next.integrator_state._replace(position=position)
        return CondDenoiserState(integrator_state_next, log_weights)

    def posterior_logpdf(
        self, rng_key: PRNGKeyArray, y_meas: Array, design_mask: Array
    ):
        def _posterior_logpdf(x, t):
            y_t = self.y_noiser(design_mask, rng_key, SDEState(y_meas, 0), t).position

            alpha_t = jnp.exp(self.sde.beta.integrate(0.0, t) / 2)
            guidance = self.forward_model.grad_logprob_y(x, y_t, design_mask) / alpha_t

            return guidance + self.score(x, t)

        return _posterior_logpdf

    def pooled_posterior_logpdf(
        self,
        rng_key: PRNGKeyArray,
        y_cntrst: Array,
        y_past: Array,
        design: Array,
        mask_history: Array,
    ):
        rng_key1, rng_key2 = jax.random.split(rng_key)
        vec_noiser = jax.vmap(
            self.y_noiser, in_axes=(None, None, SDEState(0, None), None)
        )
        mask = self.forward_model.make(design)

        def _pooled_posterior_logpdf(x, t):
            y_t = vec_noiser(mask, rng_key1, SDEState(y_cntrst, 0), t).position
            alpha_t = jnp.exp(self.sde.beta.integrate(0.0, t) / 2)
            guidance: Array = (
                jax.vmap(self.forward_model.grad_logprob_y, in_axes=(None, 0, None))(
                    x, y_t, mask
                )
                / alpha_t
            )
            past_contribution = self.posterior_logpdf(rng_key2, y_past, mask_history)
            return guidance.mean(axis=0) + past_contribution(x, t)

        return _pooled_posterior_logpdf

    def y_noiser(
        self, mask: Array, key: PRNGKeyArray, state: SDEState, ts: float
    ) -> SDEState:
        r"""
        Generate y^{(t)} = \sqrt{\bar{\alpha}_t} y + \sqrt{1-\bar{\alpha}_t} A_\xi \epsilon
        """
        y, t = state.position, state.t

        int_b = self.sde.beta.integrate(ts, t)
        alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)

        rndm = jax.random.normal(key, y.shape)

        res = alpha * y + jnp.sqrt(beta) * einops.einsum(
            mask, rndm, "... , ... c -> ... c"
        )
        return SDEState(res, ts)
