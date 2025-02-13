from dataclasses import dataclass

import jax
import jax.experimental
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.cond import CondDenoiser


@dataclass
class DPSDenoiser(CondDenoiser):
    """Conditional denoiser using Diffusion Posterior Sampling with Tweedie's formula"""

    def posterior_logpdf(
        self, rng_key: PRNGKeyArray, y_meas: Array, design_mask: Array
    ):
        # Using Tweedie's formula for posterior sampling
        def _posterior_logpdf(x, t):
            # Apply Tweedie's formula to get denoised prediction
            denoised = self.sde.tweedie(SDEState(x, t), self.score).position

            # Compute residual: A^T(y - AE[X_0|X_t])
            v = self.forward_model.grad_logprob_y(denoised, y_meas, design_mask)

            int_b = self.sde.beta.integrate(t, 0.0)
            alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)

            # Compute score and guidance in one JVP operation
            def score_fn(x_):
                return self.score(x_, t)

            score_val, tangents = jax.jvp(score_fn, (x,), (v,))
            scale = 1 / (jnp.linalg.norm(v) + 1e-3)
            guidance = (
                scale
                * (v + beta * tangents)
                / (alpha * self.forward_model.sigma_prob + 1e-3)
            )

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
        _, subkey = jax.random.split(rng_key)
        mask = self.forward_model.make(design)

        def _pooled_posterior_logpdf(x, t):
            # Apply Tweedie's formula
            denoised = self.sde.tweedie(SDEState(x, t), self.score).position

            # Compute guidance using denoised prediction for contrast samples
            residual = jax.vmap(
                self.forward_model.grad_logprob_y, in_axes=(None, 0, None)
            )(denoised, y_cntrst, mask).mean(axis=0)

            int_b = self.sde.beta.integrate(t, 0.0)
            alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)

            # Compute (I + ∇score)ᵀv using forward-mode autodiff
            def score_fn(x_):
                return self.score(x_, t)

            _, tangents = jax.jvp(score_fn, (x,), (residual,))

            scale = 1.0
            guidance = (
                scale
                * (residual + beta * tangents)
                / (alpha * self.forward_model.sigma_prob + 1e-3)
            )

            past_contribution = self.posterior_logpdf(subkey, y_past, mask_history)
            return guidance + past_contribution(x, t)

        return _pooled_posterior_logpdf
