from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.diffusion.sde import SDEState


@dataclass
class PiGDMDenoiser(CondDenoiser):
    """Pseudoinverse-Guided Diffusion Models (ΠiGDM).

    Guidance: grad = ∇_x [(A†r)ᵀ · x̂₀] where r = y - Ax̂₀

    Args:
        zeta: Step size scaling factor
        epsilon: Numerical stability constant
        cg_maxiter: CG iterations for pseudo-inverse (0 = use Aᵀ instead)
        cg_reg: Tikhonov regularization for CG solve

    Reference: Song et al., "Pseudoinverse-Guided Diffusion Models for Inverse Problems"
    """

    zeta: float = 1e-2
    epsilon: float = 1e-1
    cg_maxiter: int = 0
    cg_reg: float = 1e-2

    def _pseudoinverse(self, residual: Array, measurement_state: MeasurementState) -> Array:
        if self.cg_maxiter <= 0:
            return self.forward_model.adjoint(residual, measurement_state)

        def normal_op(v):
            Av = self.forward_model.apply(v, measurement_state)
            AtAv = self.forward_model.adjoint(Av, measurement_state)
            return AtAv + self.cg_reg * v

        rhs = self.forward_model.adjoint(residual, measurement_state)
        direction, _ = jax.scipy.sparse.linalg.cg(normal_op, rhs, maxiter=self.cg_maxiter)
        return direction

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        x_t = state.integrator_state.position
        t_current = self.integrator.timer(state.integrator_state.step)

        def guidance_loss(x: Array) -> Array:
            x0_hat = self.model.tweedie(SDEState(x, t_current), self.predictor.score).position
            residual = measurement_state.y - self.forward_model.apply(x0_hat, measurement_state)
            direction = jax.lax.stop_gradient(self._pseudoinverse(residual, measurement_state))
            return jnp.sum(direction * x0_hat)

        loss_val, grad = jax.value_and_grad(guidance_loss)(x_t)
        step_size = self.zeta / (jnp.sqrt(jnp.abs(loss_val)) + self.epsilon)

        integrator_state_uncond = self.integrator(state.integrator_state, self.predictor)
        position_corrected = integrator_state_uncond.position + step_size * grad

        integrator_state_next = integrator_state_uncond._replace(position=position_corrected)
        return state._replace(integrator_state=integrator_state_next)
