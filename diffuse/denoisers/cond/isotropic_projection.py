# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from jaxtyping import PRNGKeyArray

from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState


@dataclass
class IPDenoiser(CondDenoiser):
    """Closed-form isotropic projection step used by C-DPS."""

    cg_tol: float = 1e-5
    cg_maxiter: int = 50
    sample_posterior: bool = False
    min_transition_variance: float = 1e-6
    min_gamma: float = 1e-6
    min_noise_level: float = 1e-5
    lambda_reg: float = 0.0
    measurement_seed: Optional[int] = 0

    def __post_init__(self):
        self.resample = False

    def _measurement_noise(self, measurement_state: MeasurementState, dtype) -> jnp.ndarray:
        eps_y_attr = getattr(measurement_state, "eps_y", None)
        if eps_y_attr is not None:
            return jnp.asarray(eps_y_attr, dtype=dtype)

        if self.measurement_seed is None:
            raise ValueError(
                "IsotropicProjectionDenoiser requires either measurement_state.eps_y "
                "or a non-None measurement_seed to generate synchronized noise."
            )
        key_eps = jax.random.PRNGKey(self.measurement_seed)
        return jax.random.normal(key_eps, measurement_state.y.shape, dtype=dtype)

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        integrator_state = state.integrator_state
        x_t = integrator_state.position
        step_idx = integrator_state.step

        dtype = x_t.dtype
        eps = jnp.asarray(1e-12, dtype=dtype)
        min_tau_sq = jnp.asarray(self.min_transition_variance, dtype=dtype)
        min_gamma = jnp.asarray(self.min_gamma, dtype=dtype)
        min_sigma = jnp.asarray(self.min_noise_level, dtype=dtype)

        t_current = jnp.asarray(self.integrator.timer(step_idx), dtype=dtype)
        t_next = jnp.asarray(self.integrator.timer(step_idx + 1), dtype=dtype)

        alpha_t = self.model.signal_level(t_current)
        alpha_s = self.model.signal_level(t_next)
        sigma_t = jnp.maximum(self.model.noise_level(t_current), min_sigma)
        sigma_s = jnp.maximum(self.model.noise_level(t_next), min_sigma)

        sigma_t_sq = sigma_t * sigma_t
        sigma_s_sq = sigma_s * sigma_s

        a_ts = alpha_t / alpha_s
        tau_sq = jnp.maximum(sigma_t_sq - (a_ts * a_ts) * sigma_s_sq, min_tau_sq)
        c_ts = (a_ts * a_ts) / tau_sq

        sigma_n = jnp.asarray(getattr(self.forward_model, "std", 1.0), dtype=dtype)
        sigma_n_sq = sigma_n * sigma_n
        gamma_s = jnp.maximum(alpha_s * alpha_s * sigma_n_sq + sigma_s_sq, min_gamma)
        inv_gamma_s = 1.0 / gamma_s
        inv_sqrt_gamma_s = jnp.reciprocal(jnp.sqrt(gamma_s))

        y0 = jnp.asarray(measurement_state.y, dtype=dtype)
        eps_y = self._measurement_noise(measurement_state, dtype)
        y_s = alpha_s * y0 + sigma_s * eps_y

        def apply_A(v):
            return self.forward_model.apply(v, measurement_state)

        def apply_AT(z):
            return self.forward_model.adjoint(z, measurement_state)

        score_t = self.predictor.score(x_t, t_current)
        b_s = sigma_s_sq * apply_A(score_t)

        rhs_prior = (a_ts / tau_sq) * x_t
        rhs_meas = inv_gamma_s * apply_AT(y_s - b_s)
        rhs = rhs_prior + rhs_meas

        def normal_operator(v):
            measured = apply_A(v)
            return (c_ts + self.lambda_reg) * v + inv_gamma_s * apply_AT(measured)

        mu_post, _ = cg(
            normal_operator,
            rhs,
            x0=x_t,
            tol=self.cg_tol,
            maxiter=self.cg_maxiter,
        )

        if self.sample_posterior:
            key_eps1, key_eps2 = jax.random.split(rng_key, 2)
            eps1 = jax.random.normal(key_eps1, x_t.shape, dtype=dtype)
            eps2 = jax.random.normal(key_eps2, y0.shape, dtype=dtype)
            z = jnp.sqrt(c_ts) * eps1 + inv_sqrt_gamma_s * apply_AT(eps2)
            noise_post, _ = cg(
                normal_operator,
                z,
                x0=jnp.zeros_like(x_t),
                tol=self.cg_tol,
                maxiter=self.cg_maxiter,
            )
            x_s = mu_post + noise_post
        else:
            x_s = mu_post

        integrator_state_next = integrator_state._replace(
            position=x_s,
            step=step_idx + 1,
        )
        return state._replace(integrator_state=integrator_state_next)
