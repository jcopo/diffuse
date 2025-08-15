"""Test configuration system for diffusion tests.

This module provides a centralized configuration system for tests, similar to
the triax configuration approach, but designed for pytest fixtures.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import jax
import jax.numpy as jnp
import pytest

from diffuse.base_forward_model import ForwardModel
from diffuse.denoisers.cond import DPSDenoiser, FPSDenoiser, TMPDenoiser
from diffuse.diffusion.sde import SDE, LinearSchedule, CosineSchedule
from diffuse.integrator.deterministic import (
    DDIMIntegrator,
    DPMpp2sIntegrator,
    EulerIntegrator,
    HeunIntegrator,
)
from diffuse.integrator.stochastic import EulerMaruyamaIntegrator
from diffuse.timer.base import VpTimer
from diffuse.examples.gaussian_mixtures.mixture import rho_t

from diffuse.examples.gaussian_mixtures.forward_models.matrix_product import MatrixProduct
from diffuse.examples.gaussian_mixtures.initialization import (
    init_simple_mixture,
    init_grid_mixture,
    init_bimodal_setup,
)


@dataclass
class TestConfig:
    """Configuration for diffusion tests."""

    # Basic parameters
    key: jax.random.PRNGKey
    t_init: float = 0.0
    t_final: float = 1.0
    n_samples: int = 300
    n_steps: int = 300
    d: int = 2
    sigma_y: float = 0.1
    space_min: float = -5
    space_max: float = 5
    space_points: int = 300

    # Adaptive percentile parameters
    adaptive_percentiles: bool = True
    percentile_strategy: str = "uniform_noise"
    n_percentile_points: int = 11

    # Component specifications
    schedule_name: str = "LinearSchedule"
    timer_name: str = "vp"
    integrator_class: Type = EulerIntegrator
    integrator_params: Dict[str, Any] = None
    denoiser_class: Type = DPSDenoiser

    # Derived objects (will be populated by get_test_config)
    sde: Optional[SDE] = None
    timer: Optional[Any] = None
    integrator: Optional[Any] = None
    forward_model: Optional[ForwardModel] = None
    mix_state: Optional[Any] = None
    A: Optional[jnp.ndarray] = None
    y_target: Optional[jnp.ndarray] = None
    space: Optional[jnp.ndarray] = None
    ts: Optional[jnp.ndarray] = None
    perct: Optional[List[float]] = None

    # Ready-to-use denoisers (will be populated by get_test_config)
    denoiser: Optional[Any] = None
    cond_denoiser: Optional[Any] = None

    def __post_init__(self):
        if self.integrator_params is None:
            self.integrator_params = {}


# Configuration templates for different test scenarios
SCHEDULE_CONFIGS = {
    "LinearSchedule": {"b_min": 0.1, "b_max": 7.0},
    "CosineSchedule": {"b_min": 0.1, "b_max": 14.0},
}

# Simplified integrator configs - keeping essential combinations
INTEGRATOR_CONFIGS = [
    (EulerMaruyamaIntegrator, {}),
    (DDIMIntegrator, {"stochastic_churn_rate": 1.0, "churn_min": 0.5, "churn_max": 2.0}),
    (DPMpp2sIntegrator, {"stochastic_churn_rate": 1.0, "churn_min": 0.5, "churn_max": 2.0}),
    (HeunIntegrator, {"stochastic_churn_rate": 1.0, "churn_min": 0.5, "churn_max": 2.0}),
    (EulerIntegrator, {"stochastic_churn_rate": 1.0, "churn_min": 0.5, "churn_max": 2.0}),
]

TIMER_CONFIGS = {
    "vp": lambda n_steps, t_final: VpTimer(n_steps=n_steps, eps=0.001, tf=t_final),
    # "heun": lambda n_steps, t_final: HeunTimer(n_steps=n_steps, rho=7.0, sigma_min=0.002, sigma_max=1.0), # HeunTimer should be used only with sampling methods that are defined on noise levels
}

DENOISER_CLASSES = [DPSDenoiser, TMPDenoiser, FPSDenoiser]

PERCENTILES = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.95, 0.98, 1.0]


# Adaptive percentile functions
def compute_noise_level_derivative(sde: SDE, t: float, dt: float = 1e-4) -> float:
    """
    Compute the derivative of noise level with respect to time.

    Args:
        sde: SDE object with noise schedule
        t: Time point
        dt: Small time step for numerical differentiation

    Returns:
        Derivative of noise level at time t
    """
    if t <= dt:
        return (sde.noise_level(t + dt) - sde.noise_level(t)) / dt
    elif t >= sde.tf - dt:
        return (sde.noise_level(t) - sde.noise_level(t - dt)) / dt
    else:
        return (sde.noise_level(t + dt) - sde.noise_level(t - dt)) / (2 * dt)


def uniform_noise_percentiles(sde: SDE, n_points: int = 11) -> List[float]:
    """
    Generate percentiles that are uniform in noise level space.

    This samples uniformly in sqrt(1 - α_t) space, providing more points
    during high noise phases of the diffusion process.

    Args:
        sde: SDE object with noise schedule
        n_points: Number of time points to generate

    Returns:
        List of percentiles (0.0 to 1.0)
    """
    # Get noise levels at start and end
    noise_start = sde.noise_level(0.0)
    noise_end = sde.noise_level(sde.tf)

    # Sample uniformly in noise_level space (noise_level now returns σ(t), not σ²(t))
    target_noise_levels = jnp.linspace(noise_start, noise_end, n_points)

    # Convert noise levels back to time percentiles
    percentiles = []
    for target_noise in target_noise_levels:
        # Binary search to find time corresponding to target noise level
        t_min, t_max = 0.0, sde.tf
        for _ in range(50):  # Binary search iterations
            t_mid = (t_min + t_max) / 2
            noise_mid = sde.noise_level(t_mid)
            if noise_mid < target_noise:
                t_min = t_mid
            else:
                t_max = t_mid

        # Convert time to percentile
        percentile = (t_min + t_max) / 2 / sde.tf
        percentiles.append(float(percentile))

    return percentiles


def derivative_based_percentiles(sde: SDE, n_points: int = 11) -> List[float]:
    """
    Generate percentiles that are denser where noise level changes rapidly.

    This approach samples more points where d(noise_level)/dt is large,
    focusing on the most dynamic phases of the diffusion process.

    Args:
        sde: SDE object with noise schedule
        n_points: Number of time points to generate

    Returns:
        List of percentiles (0.0 to 1.0)
    """
    # Sample candidate time points
    n_candidates = 1000
    candidate_times = jnp.linspace(0.0, sde.tf, n_candidates)

    # Compute derivatives at candidate points
    derivatives = []
    for t in candidate_times:
        deriv = abs(compute_noise_level_derivative(sde, float(t)))
        derivatives.append(deriv)

    derivatives = jnp.array(derivatives)

    # Normalize derivatives to create a probability distribution
    derivatives_norm = derivatives / jnp.sum(derivatives)

    # Create cumulative distribution
    cumulative = jnp.cumsum(derivatives_norm)

    # Sample uniformly from cumulative distribution
    uniform_samples = jnp.linspace(0.0, 1.0, n_points)

    # Find corresponding time points
    percentiles = []
    for u in uniform_samples:
        # Find the index where cumulative >= u
        idx = jnp.searchsorted(cumulative, u)
        idx = jnp.clip(idx, 0, len(candidate_times) - 1)

        # Convert time to percentile
        percentile = float(candidate_times[idx] / sde.tf)
        percentiles.append(percentile)

    return percentiles


def logarithmic_percentiles(sde: SDE, n_points: int = 11) -> List[float]:
    """
    Generate percentiles that are logarithmically spaced in noise level.

    This provides more detail in low noise regions (early diffusion stages)
    and less detail in high noise regions (late diffusion stages).

    Args:
        sde: SDE object with noise schedule
        n_points: Number of time points to generate

    Returns:
        List of percentiles (0.0 to 1.0)
    """
    # Get noise levels at start and end
    noise_start = sde.noise_level(0.0)
    noise_end = sde.noise_level(sde.tf)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    noise_start = max(noise_start, epsilon)
    noise_end = max(noise_end, epsilon)

    # Sample logarithmically in noise space (noise_level now returns σ(t))
    log_noise_levels = jnp.linspace(jnp.log(noise_start), jnp.log(noise_end), n_points)
    target_noise_levels = jnp.exp(log_noise_levels)

    # Convert noise levels back to time percentiles
    percentiles = []
    for target_noise in target_noise_levels:
        # Binary search to find time corresponding to target noise level
        t_min, t_max = 0.0, sde.tf
        for _ in range(50):  # Binary search iterations
            t_mid = (t_min + t_max) / 2
            noise_mid = sde.noise_level(t_mid)
            if noise_mid < target_noise:
                t_min = t_mid
            else:
                t_max = t_mid

        # Convert time to percentile
        percentile = (t_min + t_max) / 2 / sde.tf
        percentiles.append(float(percentile))

    return percentiles


def hybrid_percentiles(sde: SDE, n_points: int = 11) -> List[float]:
    """
    Generate percentiles using a hybrid approach.

    Combines uniform noise sampling with derivative-based sampling
    to provide good coverage across all noise levels.

    Args:
        sde: SDE object with noise schedule
        n_points: Number of time points to generate

    Returns:
        List of percentiles (0.0 to 1.0)
    """
    # Split points between uniform noise and derivative-based
    n_uniform = n_points // 2
    n_derivative = n_points - n_uniform

    # Get percentiles from both methods
    uniform_perct = uniform_noise_percentiles(sde, n_uniform)
    derivative_perct = derivative_based_percentiles(sde, n_derivative)

    # Combine and sort
    combined = uniform_perct + derivative_perct
    combined = sorted(list(set(combined)))  # Remove duplicates and sort

    # If we have too many points, subsample
    if len(combined) > n_points:
        indices = jnp.linspace(0, len(combined) - 1, n_points).astype(int)
        combined = [combined[i] for i in indices]

    return combined


def compute_adaptive_percentiles(sde: SDE, n_points: int = 11, strategy: str = "logarithmic") -> List[float]:
    """
    Compute adaptive percentiles based on noise schedule.

    Args:
        sde: SDE object with noise schedule
        n_points: Number of time points to generate
        strategy: Strategy for adaptive sampling:
            - 'uniform_noise': Uniform in sqrt(noise_level) space
            - 'derivative': Dense where noise level changes rapidly
            - 'logarithmic': Logarithmic spacing in noise level
            - 'hybrid': Combination of uniform and derivative
            - 'fixed': Use traditional fixed percentiles

    Returns:
        List of percentiles (0.0 to 1.0) corresponding to meaningful noise levels
    """
    if strategy == "uniform_noise":
        return uniform_noise_percentiles(sde, n_points)
    elif strategy == "derivative":
        return derivative_based_percentiles(sde, n_points)
    elif strategy == "logarithmic":
        return logarithmic_percentiles(sde, n_points)
    elif strategy == "hybrid":
        return hybrid_percentiles(sde, n_points)
    elif strategy == "fixed":
        # Use traditional fixed percentiles
        if n_points <= len(PERCENTILES):
            indices = jnp.linspace(0, len(PERCENTILES) - 1, n_points).astype(int)
            return [PERCENTILES[i] for i in indices]
        else:
            return jnp.linspace(0.0, 1.0, n_points).tolist()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _create_sde(schedule_name: str, t_final: float = 1.0) -> SDE:
    """Create SDE with given schedule and final time."""
    schedule_params = SCHEDULE_CONFIGS[schedule_name]

    if schedule_name == "LinearSchedule":
        beta = LinearSchedule(
            b_min=schedule_params["b_min"],
            b_max=schedule_params["b_max"],
            t0=0.0,
            T=t_final,
        )
    else:  # CosineSchedule
        beta = CosineSchedule(
            b_min=schedule_params["b_min"],
            b_max=schedule_params["b_max"],
            t0=0.0,
            T=t_final,
        )

    return SDE(beta=beta)


def _create_timer(timer_name: str, n_steps: int, t_final: float = 1.0):
    """Create timer with given parameters."""
    if timer_name in TIMER_CONFIGS:
        return TIMER_CONFIGS[timer_name](n_steps, t_final)
    else:
        raise ValueError(f"Unknown timer: {timer_name}")


def _create_mixture(key: jax.random.PRNGKey, d: int = 2):
    """Create mixture based on dimensionality."""
    if d > 2:
        return init_grid_mixture(key, d)
    else:
        return init_simple_mixture(key, d)


def get_test_config(conditional: bool = False, **kwargs) -> TestConfig:
    """
    Create unified test configuration for both basic and conditional diffusion tests.

    Args:
        conditional: If True, create conditional setup; if False, create basic setup
        **kwargs: Override any default parameters

    Returns:
        TestConfig with appropriate setup
    """
    # Create base config with defaults
    config = TestConfig(key=jax.random.PRNGKey(42), **kwargs)

    # Common setup for both conditional and basic
    config.sde = _create_sde(config.schedule_name, config.t_final)
    config.timer = _create_timer(config.timer_name, config.n_steps, config.t_final)
    config.space = jnp.linspace(config.space_min, config.space_max, config.space_points)
    config.ts = jnp.linspace(config.t_init, config.t_final, config.n_steps)

    # Compute adaptive percentiles based on noise schedule
    if config.adaptive_percentiles:
        config.perct = compute_adaptive_percentiles(
            config.sde, n_points=config.n_percentile_points, strategy=config.percentile_strategy
        )
    else:
        config.perct = PERCENTILES

    if conditional:
        # Conditional setup
        from diffuse.base_forward_model import MeasurementState
        from diffuse.denoisers.denoiser import Denoiser
        from diffuse.examples.gaussian_mixtures.cond_mixture import compute_posterior, compute_xt_given_y
        from diffuse.examples.gaussian_mixtures.mixture import pdf_mixtr, cdf_mixtr, sampler_mixtr

        # Create mixture and forward model
        # config.mix_state, config.A, config.y_target, sigma_y = init_circular_setup(config.key, config.d)
        config.mix_state, config.A, config.y_target, sigma_y = init_bimodal_setup(config.key, config.d)
        config.forward_model = MatrixProduct(A=config.A, std=sigma_y)

        # Create integrator
        config.integrator = config.integrator_class(sde=config.sde, timer=config.timer, **config.integrator_params)

        # Generate sample for denoiser creation
        key_sample = jax.random.PRNGKey(123)
        x_sample = sampler_mixtr(key_sample, config.mix_state, 1)[0]
        y_sample = config.forward_model.measure(key_sample, x_sample)

        # Create posterior and score functions
        posterior_state = compute_posterior(config.mix_state, y_sample, config.A, config.sigma_y)
        measurement_state = MeasurementState(y=y_sample, mask_history=config.A)

        def conditional_pdf(x, t):
            mix_state_t = compute_xt_given_y(posterior_state, config.sde, t)
            return pdf_mixtr(mix_state_t, x)

        def conditional_cdf(x, t):
            mix_state_t = compute_xt_given_y(posterior_state, config.sde, t)
            return cdf_mixtr(mix_state_t, x)

        def conditional_score(x, t):
            return jax.grad(conditional_pdf)(x, t) / conditional_pdf(x, t)

        def unconditional_score(x, t):
            pdf = rho_t(x, t, init_mix_state=config.mix_state, sde=config.sde)
            return jax.grad(lambda x: rho_t(x, t, init_mix_state=config.mix_state, sde=config.sde))(x) / pdf

        # Create denoisers
        config.cond_denoiser = config.denoiser_class(
            integrator=config.integrator,
            sde=config.sde,
            score=unconditional_score,
            forward_model=config.forward_model,
            x0_shape=x_sample.shape,
        )

        config.denoiser = Denoiser(
            integrator=config.integrator,
            sde=config.sde,
            score=conditional_score,
            x0_shape=x_sample.shape,
        )

        # Store conditional-specific functions
        config.pdf = conditional_pdf
        config.cdf = conditional_cdf
        config.score = conditional_score
        config.posterior_state = posterior_state
        config.measurement_state = measurement_state

    else:
        # Basic unconditional setup
        config.mix_state = _create_mixture(config.key, config.d)

        def pdf(x, t):
            return rho_t(x, t, init_mix_state=config.mix_state, sde=config.sde)

        def score(x, t):
            return jax.grad(pdf)(x, t) / pdf(x, t)

        config.pdf = pdf
        config.score = score

    return config


# Backward compatibility functions
def get_basic_test_config(**kwargs) -> TestConfig:
    """Create basic test configuration - backward compatibility."""
    return get_test_config(conditional=False, **kwargs)


def get_conditional_test_config(**kwargs) -> TestConfig:
    """Create conditional test configuration - backward compatibility."""
    return get_test_config(conditional=True, **kwargs)


def get_parametrized_configs() -> List[pytest.param]:
    """
    Generate all combinations of parameters for comprehensive testing.

    This creates all possible combinations of:
    - Schedules: LinearSchedule, CosineSchedule
    - Timers: vp, heun
    - Integrators: EulerMaruyama, DDIM, Heun, Euler

    Total combinations: 2 × 2 × 4 = 16 test cases
    """
    configs = []

    schedules = list(SCHEDULE_CONFIGS.keys())
    timers = list(TIMER_CONFIGS.keys())
    integrators = INTEGRATOR_CONFIGS

    for schedule in schedules:
        for timer in timers:
            for integrator_class, integrator_params in integrators:
                config_dict = {
                    "schedule_name": schedule,
                    "timer_name": timer,
                    "integrator_class": integrator_class,
                    "integrator_params": integrator_params,
                }

                test_id = f"{integrator_class.__name__}_{timer}_{schedule}"
                configs.append(pytest.param(config_dict, id=test_id))

    return configs


def get_conditional_configs() -> List[pytest.param]:
    """
    Generate all combinations for conditional denoiser testing.

    This creates all possible combinations of:
    - Schedules: LinearSchedule, CosineSchedule
    - Timers: vp, heun
    - Integrators: EulerMaruyama, DDIM, Heun, Euler
    - Denoisers: DPSDenoiser, TMPDenoiser, FPSDenoiser

    Total combinations: 2 × 2 × 4 × 3 = 48 test cases
    """
    configs = []

    schedules = list(SCHEDULE_CONFIGS.keys())
    timers = list(TIMER_CONFIGS.keys())
    integrators = INTEGRATOR_CONFIGS
    denoisers = DENOISER_CLASSES

    for schedule in schedules:
        for timer in timers:
            for integrator_class, integrator_params in integrators:
                for denoiser_class in denoisers:
                    config_dict = {
                        "schedule_name": schedule,
                        "timer_name": timer,
                        "integrator_class": integrator_class,
                        "integrator_params": integrator_params,
                        "denoiser_class": denoiser_class,
                    }

                    test_id = f"{denoiser_class.__name__}_{integrator_class.__name__}_{timer}_{schedule}"
                    configs.append(pytest.param(config_dict, id=test_id))

    return configs
