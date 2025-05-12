from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.cond import CondDenoiser


@dataclass
class TMPDenoiser(CondDenoiser):
    """Conditional denoiser using Tweedie's Moments from https://arxiv.org/pdf/2310.06721v3"""

    def posterior_logpdf(
        self, rng_key: PRNGKeyArray, y_meas: Array, design_mask: Array
    ):
        def _posterior_logpdf(x, t):
            alpha, _ = self.sde.alpha_beta(t)
            scale = (1 - alpha) / jnp.sqrt(alpha)

            def tweedie_fn(x_):
                return self.sde.tweedie(SDEState(x_, t), self.score).position

            def efficient(v):
                restored_v = self.forward_model.restore_from_mask(
                    design_mask, jnp.zeros_like(x), v
                )
                _, tangents = jax.jvp(tweedie_fn, (x,), (restored_v,))
                measured_tangents = self.forward_model.measure_from_mask(
                    rng_key, design_mask, tangents
                )
                return scale * measured_tangents + self.forward_model.std ** 2 * v


            denoised = tweedie_fn(x)
            b = y_meas - self.forward_model.measure_from_mask(rng_key, design_mask, denoised)

            res, _ = jax.scipy.sparse.linalg.cg(efficient, b, maxiter=3)
            restored_res = self.forward_model.restore_from_mask(
                design_mask, jnp.zeros_like(x), res
            )
            _, guidance = jax.jvp(tweedie_fn, (x,), (restored_res,))
            score_val = self.score(x, t)

            return score_val + guidance

        return _posterior_logpdf

    def pooled_posterior_logpdf(
        self,
        rng_key: PRNGKeyArray,
        y_cntrst: Array,
        y_past: Array,
        design: Array,
        mask_history: Array,
    ):
        _, rng_key2 = jax.random.split(rng_key)
        mask = self.forward_model.make(design)

        def _pooled_posterior_logpdf(x, t):
            # Compute alpha, beta
            int_b = jnp.squeeze(self.sde.beta.integrate(t, self.sde.beta.t0))
            alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)
            scale = beta / alpha

            # Compute (A C A^T + \sigma^2 I) v efficiently
            def tweedie_fn(x_):
                return self.sde.tweedie(SDEState(x_, t), self.score).position

            def efficient(v):
                restored_v = self.forward_model.restore_from_mask(
                    mask, jnp.zeros_like(x), v
                )
                _, tangents = jax.jvp(tweedie_fn, (x,), (restored_v,))
                measured_tangents = self.forward_model.measure_from_mask(mask, tangents)
                return scale * measured_tangents + self.forward_model.std * v

            # Compute residual: (y - AE[X_0|X_t])
            score_val = self.score(
                x, t
            )  # Don't use tweedie function to avoid computing the score twice
            denoised = (x + beta * score_val) / alpha

            b = jnp.mean(
                y_cntrst - self.forward_model.measure_from_mask(mask, denoised), axis=0
            )

            res, _ = jax.scipy.sparse.linalg.cg(efficient, b, maxiter=5)
            restored_res = self.forward_model.restore_from_mask(
                mask, jnp.zeros_like(x), res
            )
            _, guidance = jax.jvp(tweedie_fn, (x,), (restored_res,))

            past_contribution = self.posterior_logpdf(rng_key2, y_past, mask_history)
            return guidance + past_contribution(x, t)

        return _pooled_posterior_logpdf
