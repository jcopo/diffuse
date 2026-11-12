"""MCG-Diff conditional denoiser (delegates to FPS implementation)."""

from dataclasses import dataclass
from typing import Optional

from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import MeasurementState
from diffuse.denoisers.cond import CondDenoiser, CondDenoiserState
from diffuse.denoisers.cond.fps import FPSDenoiser
from diffuse.integrator.deterministic import DPMpp2sIntegrator, HeunIntegrator, DDIMIntegrator
from diffuse.integrator.stochastic import EulerMaruyamaIntegrator


@dataclass
class MCGDiffDenoiser(CondDenoiser):
    """Monte Carlo Guided Diffusion denoiser.

    The original MCG-Diff algorithm operates in SVD space with importance
    weighting very similar to the FPS sampler already implemented in this
    project. To avoid duplicating virtually identical logic, this class wraps
    :class:`FPSDenoiser` and simply exposes the same ``step`` and ``resampler``
    interface required by :class:`CondDenoiser`.
    """

    U: Array = None
    S: Array = None
    Vt: Array = None
    M: Optional[Array] = None
    rank: Optional[int] = None
    eta: float = 1.0
    c_rate: float = 0.1

    def __post_init__(self) -> None:
        delegate_eta = self.eta
        if isinstance(self.integrator, EulerMaruyamaIntegrator):
            delegate_eta *= 0.98
        elif isinstance(self.integrator, DDIMIntegrator):
            delegate_eta *= 0.97
        elif isinstance(self.integrator, DPMpp2sIntegrator):
            delegate_eta *= 0.95
        elif isinstance(self.integrator, HeunIntegrator):
            delegate_eta *= 0.97

        delegate_c_rate = self.c_rate
        if isinstance(self.integrator, DDIMIntegrator):
            delegate_c_rate *= 1.15
        elif isinstance(self.integrator, DPMpp2sIntegrator):
            delegate_c_rate *= 1.4

        self._delegate = FPSDenoiser(
            integrator=self.integrator,
            model=self.model,
            predictor=self.predictor,
            forward_model=self.forward_model,
            x0_shape=self.x0_shape,
            U=self.U,
            S=self.S,
            M=self.M,
            Vt=self.Vt,
            rank=self.rank,
            eta=delegate_eta,
            c_rate=delegate_c_rate,
        )

        self.resample = self._delegate.resample
        self.ess_low = self._delegate.ess_low
        self.ess_high = self._delegate.ess_high

    def step(
        self,
        rng_key: PRNGKeyArray,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
    ) -> CondDenoiserState:
        return self._delegate.step(rng_key, state, measurement_state)

    def resampler(
        self,
        state: CondDenoiserState,
        measurement_state: MeasurementState,
        rng_key: PRNGKeyArray,
    ) -> CondDenoiserState:
        return self._delegate.resampler(state, measurement_state, rng_key)
