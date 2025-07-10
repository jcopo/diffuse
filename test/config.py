"""Test configuration system for diffusion tests.

This module provides a centralized configuration system for tests, similar to
the triax configuration approach, but designed for pytest fixtures.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import pytest

from diffuse.base_forward_model import ForwardModel
from diffuse.denoisers.cond import CondDenoiser, DPSDenoiser, FPSDenoiser, TMPDenoiser
from diffuse.denoisers.denoiser import Denoiser
from diffuse.diffusion.sde import SDE, LinearSchedule, CosineSchedule
from diffuse.integrator.deterministic import (
    DDIMIntegrator,
    DPMpp2sIntegrator,
    EulerIntegrator,
    HeunIntegrator,
)
from diffuse.integrator.stochastic import EulerMaruyamaIntegrator
from diffuse.timer.base import HeunTimer, VpTimer

from examples.gaussian_mixtures.forward_models.matrix_product import MatrixProduct
from examples.gaussian_mixtures.initialization import (
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
    basic_denoiser: Optional[Any] = None

    def __post_init__(self):
        if self.integrator_params is None:
            self.integrator_params = {}


# Configuration templates for different test scenarios
SCHEDULE_CONFIGS = {
    "LinearSchedule": {"b_min": 0.0001, "b_max": 20.0},
    "CosineSchedule": {"b_min": 0.1, "b_max": 20.0},
}

INTEGRATOR_CONFIGS = [
    (EulerMaruyamaIntegrator, {}),
    (DDIMIntegrator, {}),
    (
        DDIMIntegrator,
        {
            "stochastic_churn_rate": 0.0,
            "churn_min": 0.0,
            "churn_max": 0.0,
            "noise_inflation_factor": 1.0001,
        },
    ),
    (
        HeunIntegrator,
        {
            "stochastic_churn_rate": 1.0,
            "churn_min": 0.5,
            "churn_max": 2.0,
            "noise_inflation_factor": 1.0001,
        },
    ),
    (
        DPMpp2sIntegrator,
        {
            "stochastic_churn_rate": 1.0,
            "churn_min": 0.5,
            "churn_max": 2.0,
            "noise_inflation_factor": 1.0001,
        },
    ),
    (EulerIntegrator, {"stochastic_churn_rate": 0.0}),
]

TIMER_CONFIGS = {
    "vp": lambda n_steps, t_final: VpTimer(n_steps=n_steps, eps=0.001, tf=t_final),
    "heun": lambda n_steps, t_final: HeunTimer(n_steps=n_steps, rho=7.0, sigma_min=0.002, sigma_max=1.0),
}

DENOISER_CLASSES = [DPSDenoiser, TMPDenoiser, FPSDenoiser]

PERCENTILES = [0.0, 0.05, 0.1, 0.3, 0.6, 0.7, 0.73, 0.75, 0.8, 0.9]


def create_sde(schedule_name: str, t_final: float = 1.0) -> SDE:
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

    return SDE(beta=beta, tf=t_final)


def create_timer(timer_name: str, n_steps: int, t_final: float = 1.0):
    """Create timer with given parameters."""
    if timer_name in TIMER_CONFIGS:
        return TIMER_CONFIGS[timer_name](n_steps, t_final)
    else:
        raise ValueError(f"Unknown timer: {timer_name}")


def create_mixture(key: jax.random.PRNGKey, d: int = 2):
    """Create mixture based on dimensionality."""
    if d > 2:
        return init_grid_mixture(key, d)
    else:
        return init_simple_mixture(key, d)


def get_test_config(**kwargs) -> TestConfig:
    """
    Create test configuration with ALL components initialized and ready to use.

    Args:
        **kwargs: Override any default parameters

    Returns:
        TestConfig with everything pre-configured, including denoisers
    """
    from diffuse.base_forward_model import MeasurementState
    from diffuse.denoisers.denoiser import Denoiser
    from examples.gaussian_mixtures.cond_mixture import compute_posterior, compute_xt_given_y
    from examples.gaussian_mixtures.mixture import pdf_mixtr, cdf_mixtr, sampler_mixtr

    # Create base config with defaults
    config = TestConfig(key=jax.random.PRNGKey(42), **kwargs)

    # Create SDE
    config.sde = create_sde(config.schedule_name, config.t_final)

    # Create timer
    config.timer = create_timer(config.timer_name, config.n_steps, config.t_final)

    # Create mixture and forward model
    config.mix_state, config.A, config.y_target, sigma_y = init_bimodal_setup(config.key, config.d)
    config.forward_model = MatrixProduct(A=config.A, std=sigma_y)

    # Create integrator
    config.integrator = config.integrator_class(sde=config.sde, timer=config.timer, **config.integrator_params)

    # Create derived arrays
    config.space = jnp.linspace(config.space_min, config.space_max, config.space_points)
    config.ts = jnp.linspace(config.t_init, config.t_final, config.n_steps)
    config.perct = PERCENTILES

    # Generate sample for denoiser creation (using a fixed key for reproducibility)
    key_sample = jax.random.PRNGKey(123)  # Fixed key for consistent denoiser shapes
    x_sample = sampler_mixtr(key_sample, config.mix_state, 1)[0]
    y_sample = config.forward_model.measure(key_sample, x_sample)

    # Create posterior and score functions for this sample
    posterior_state = compute_posterior(config.mix_state, y_sample, config.A, config.sigma_y)

    def pdf(x, t):
        mix_state_t = compute_xt_given_y(posterior_state, config.sde, t)
        return pdf_mixtr(mix_state_t, x)

    def cdf(x, t):
        mix_state_t = compute_xt_given_y(posterior_state, config.sde, t)
        return cdf_mixtr(mix_state_t, x)

    def score(x, t):
        return jax.grad(pdf)(x, t) / pdf(x, t)

    # Create ready-to-use denoisers
    config.denoiser = config.denoiser_class(
        integrator=config.integrator,
        sde=config.sde,
        score=score,
        forward_model=config.forward_model,
        x0_shape=x_sample.shape,
    )

    config.basic_denoiser = Denoiser(
        integrator=config.integrator,
        sde=config.sde,
        score=score,
        x0_shape=x_sample.shape,
    )

    # Store everything needed by tests
    config.generate_samples = lambda key, n_samples=1: sampler_mixtr(key, config.mix_state, n_samples)
    config.compute_posterior = lambda y: compute_posterior(config.mix_state, y, config.A, config.sigma_y)
    config.create_measurement_state = lambda y: MeasurementState(y=y, mask_history=jnp.ones_like(y))
    config.pdf = pdf
    config.cdf = cdf
    config.score = score
    config.posterior_state = posterior_state

    return config


def get_parametrized_configs() -> List[pytest.param]:
    """
    Get list of pytest.param objects for parametrization with proper test IDs.

    Returns:
        List of pytest.param objects that can be used with pytest indirect parametrization
    """
    configs = []

    # Basic combinations
    schedules = ["LinearSchedule", "CosineSchedule"]
    timers = ["vp", "heun"]
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

                # Create test ID for pytest -k filtering
                test_id = f"{integrator_class.__name__}_{timer}_{schedule}"

                configs.append(pytest.param(config_dict, id=test_id))

    return configs


def get_conditional_configs() -> List[pytest.param]:
    """
    Get configurations for conditional denoiser tests with proper test IDs.

    Returns:
        List of pytest.param objects for conditional denoiser parametrization
    """
    configs = []

    # Basic combinations
    schedules = ["LinearSchedule", "CosineSchedule"]
    timers = ["vp", "heun"]
    integrators = INTEGRATOR_CONFIGS

    for schedule in schedules:
        for timer in timers:
            for integrator_class, integrator_params in integrators:
                for denoiser_class in DENOISER_CLASSES:
                    config_dict = {
                        "schedule_name": schedule,
                        "timer_name": timer,
                        "integrator_class": integrator_class,
                        "integrator_params": integrator_params,
                        "denoiser_class": denoiser_class,
                    }

                    # Create test ID for pytest -k filtering
                    test_id = f"{denoiser_class.__name__}_{integrator_class.__name__}_{timer}_{schedule}"

                    configs.append(pytest.param(config_dict, id=test_id))

    return configs
