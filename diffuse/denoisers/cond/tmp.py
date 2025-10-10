# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from dataclasses import dataclass
import jax
from jaxtyping import Array, PRNGKeyArray

from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.base_forward_model import MeasurementState
from diffuse.predictor import Predictor


@dataclass
class TMPDenoiser(CondDenoiser):
    """Conditional denoiser using Tweedie's Moment Projection (TMP).

    Implements TMP which modifies the score function to incorporate measurement
    information through Tweedie's formula and moment matching.

    Args:
        integrator: Numerical integrator for solving the reverse SDE
        model: Diffusion model defining the forward process
        predictor: Predictor for computing score/noise/velocity
        forward_model: Forward measurement operator

    References:
        Boys, B., Girolami, M., Pidstrigach, J., Reich, S., Mosca, A., & Akyildiz, Ö. D. (2023).
        Tweedie moment projected diffusions for inverse problems. arXiv:2310.06721
    """

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Single step of TMP sampling.

        Modifies the score to include measurement term and uses integrator for the update.

        Args:
            rng_key: Random number generator key
            state: Current conditional denoiser state
            measurement_state: Measurement information

        Returns:
            Updated conditional denoiser state
        """
        y_meas = measurement_state.y

        # Define modified score function that includes measurement term
        def modified_score(x: Array, t: Array) -> Array:
            sigma_t = self.model.noise_level(t)
            alpha_t = self.model.signal_level(t)
            scale = sigma_t / alpha_t

            def tweedie_fn(x_):
                return self.model.tweedie(SDEState(x_, t), self.predictor.score).position

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
            score_val = self.predictor.score(x, t)

            return score_val + guidance

        # Create modified predictor for guidance
        modified_predictor = Predictor(self.model, modified_score, "score")

        # Use integrator to compute next state
        integrator_state_next = self.integrator(state.integrator_state, modified_predictor)
        state_next = CondDenoiserState(integrator_state_next, state.log_weights)

        return state_next
