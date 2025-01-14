from dataclasses import dataclass
from typing import Callable, Tuple

import einops
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.denoisers.base import BaseCondDenoiser
from diffuse.integrator.base import Integrator, IntegratorState
from diffuse.diffusion.sde import SDE, SDEState
from diffuse.base_forward_model import ForwardModel, MeasurementState



@dataclass
class CondDenoiser(BaseCondDenoiser):
    """Conditional denoiser implementation"""

    # Required attributes from base class
    integrator: Integrator
    sde: SDE
    score: Callable[[Array, float], Array]
    forward_model: ForwardModel
    _resample: bool = False

    def init(
        self, position: Array, rng_key: PRNGKeyArray, dt: float
    ) -> CondDenoiserState:
        """Initialize denoiser state"""
        pass

    def step(
        self, state: CondDenoiserState, score: Callable[[Array, float], Array]
    ) -> CondDenoiserState:
        """Single step update"""
        pass

    def batch_step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        score: Callable[[Array, float], Array],
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        """Batch update step"""
        pass

    def posterior_logpdf(
        self,
        rng_key: PRNGKeyArray,
        t: float,
        y_meas: Array,
        design_mask: Array,
    ):
        """Compute posterior log probability density"""
        pass

    def pooled_posterior_logpdf(
        self,
        rng_key: PRNGKeyArray,
        t: float,
        y_cntrst: Array,
        y_past: Array,
        design: Array,
        mask_history: Array,
    ):
        """Compute pooled posterior log probability density"""
        pass

    def y_noiser(
        self,
        mask: Array,
        key: PRNGKeyArray,
        state: SDEState,
        ts: float
    ) -> SDEState:
        """Add noise to measurements"""
        pass

    def _resampling(
        self,
        position: Array,
        log_weights: Array,
        rng_key: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        """Resample particles based on weights"""
        pass