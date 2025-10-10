# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple

from jaxtyping import Array, PRNGKeyArray

from diffuse.integrator.base import Integrator
from diffuse.diffusion.sde import SDE, SDEState
from diffuse.base_forward_model import ForwardModel, MeasurementState


class PnPDenoiserState(NamedTuple):
    position: Array
    auxiliary: Array
    log_weights: Array


@dataclass
class PnPDenoiser:
    """Conditional denoiser implementation"""

    # Required attributes from base class
    integrator: Integrator
    sde: SDE
    score: Callable[[Array, float], Array]
    forward_model: ForwardModel
    _resample: bool = False

    def init(self, position: Array, rng_key: PRNGKeyArray, dt: float) -> PnPDenoiserState:
        """Initialize denoiser state"""
        pass

    def step(self, state: PnPDenoiserState, score: Callable[[Array, float], Array]) -> PnPDenoiserState:
        """Single step update"""
        pass

    def batch_step(
        self,
        rng_key: PRNGKeyArray,
        state: PnPDenoiserState,
        score: Callable[[Array, float], Array],
        measurement_state: MeasurementState,
    ) -> PnPDenoiserState:
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

    def y_noiser(self, mask: Array, key: PRNGKeyArray, state: SDEState, ts: float) -> SDEState:
        """Add noise to measurements"""
        pass

    def _resampling(self, position: Array, log_weights: Array, rng_key: PRNGKeyArray) -> Tuple[Array, Array]:
        """Resample particles based on weights"""
        pass
