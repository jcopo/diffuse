from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.base_forward_model import MeasurementState


@dataclass
class TMPDenoiser(CondDenoiser):
    """Conditional denoiser using Tweedie's Moments from https://arxiv.org/pdf/2310.06721v3"""

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Single step of TMP sampling.

        Modifies the score to include measurement term and uses integrator for the update.
        """
        y_meas = measurement_state.y
        design_mask = measurement_state.mask_history

        # Define modified score function that includes measurement term
        def modified_score(x: Array, t: float) -> Array:
            noise_level = self.sde.noise_level(t)
            alpha = 1 - noise_level
            scale = noise_level / jnp.sqrt(alpha)

            def tweedie_fn(x_):
                return self.sde.tweedie(SDEState(x_, t), self.score).position

            def efficient(v):
                restored_v = self.forward_model.restore(v, measurement_state)
                _, tangents = jax.jvp(tweedie_fn, (x,), (restored_v,))
                measured_tangents = self.forward_model.apply(tangents, measurement_state)
                return scale * measured_tangents + self.forward_model.std**2 * v

            denoised = tweedie_fn(x)
            b = y_meas - self.forward_model.apply(denoised, measurement_state)

            res, _ = jax.scipy.sparse.linalg.cg(efficient, b, maxiter=3)
            restored_res = self.forward_model.restore(res, measurement_state)
            _, guidance = jax.jvp(tweedie_fn, (x,), (restored_res,))
            score_val = self.score(x, t)

            return score_val + guidance

        # Use integrator to compute next state
        integrator_state_next = self.integrator(state.integrator_state, modified_score)
        state_next = CondDenoiserState(integrator_state_next, state.log_weights)

        return state_next
