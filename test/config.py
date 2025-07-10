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
from examples.gaussian_mixtures.mixture import rho_t

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
    cond_denoiser: Optional[Any] = None

    def __post_init__(self):
        if self.integrator_params is None:
            self.integrator_params = {}


# Configuration templates for different test scenarios
SCHEDULE_CONFIGS = {
    "LinearSchedule": {"b_min": 0.0001, "b_max": 20.0},
    "CosineSchedule": {"b_min": 0.1, "b_max": 20.0},
}

# Simplified integrator configs - keeping essential combinations
INTEGRATOR_CONFIGS = [
    (EulerMaruyamaIntegrator, {}),
    (DDIMIntegrator, {}),
    (HeunIntegrator, {"stochastic_churn_rate": 1.0, "churn_min": 0.5, "churn_max": 2.0}),
    (EulerIntegrator, {"stochastic_churn_rate": 0.0}),
]

TIMER_CONFIGS = {
    "vp": lambda n_steps, t_final: VpTimer(n_steps=n_steps, eps=0.001, tf=t_final),
    "heun": lambda n_steps, t_final: HeunTimer(n_steps=n_steps, rho=7.0, sigma_min=0.002, sigma_max=1.0),
}

DENOISER_CLASSES = [DPSDenoiser, TMPDenoiser, FPSDenoiser]

PERCENTILES = [0.0, 0.05, 0.1, 0.3, 0.6, 0.7, 0.73, 0.75, 0.8, 0.9]


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

    return SDE(beta=beta, tf=t_final)


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
    config.perct = PERCENTILES

    if conditional:
        # Conditional setup
        from diffuse.base_forward_model import MeasurementState
        from diffuse.denoisers.denoiser import Denoiser
        from examples.gaussian_mixtures.cond_mixture import compute_posterior, compute_xt_given_y
        from examples.gaussian_mixtures.mixture import pdf_mixtr, cdf_mixtr, sampler_mixtr

        # Create mixture and forward model
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
    Get simplified list of pytest.param objects for parametrization.
    Reduced to essential combinations for faster testing.
    """
    configs = []

    # Essential combinations - reduced from full matrix for speed
    essential_configs = [
        # Basic LinearSchedule tests
        {
            "schedule_name": "LinearSchedule",
            "timer_name": "vp",
            "integrator_class": EulerIntegrator,
            "integrator_params": {"stochastic_churn_rate": 0.0},
        },
        {
            "schedule_name": "LinearSchedule",
            "timer_name": "vp",
            "integrator_class": DDIMIntegrator,
            "integrator_params": {},
        },
        # CosineSchedule with different timer
        {
            "schedule_name": "CosineSchedule",
            "timer_name": "heun",
            "integrator_class": HeunIntegrator,
            "integrator_params": {"stochastic_churn_rate": 1.0, "churn_min": 0.5, "churn_max": 2.0},
        },
        # Stochastic integrator
        {
            "schedule_name": "LinearSchedule",
            "timer_name": "vp",
            "integrator_class": EulerMaruyamaIntegrator,
            "integrator_params": {},
        },
    ]

    for config_dict in essential_configs:
        test_id = (
            f"{config_dict['integrator_class'].__name__}_{config_dict['timer_name']}_{config_dict['schedule_name']}"
        )
        configs.append(pytest.param(config_dict, id=test_id))

    return configs


def get_conditional_configs() -> List[pytest.param]:
    """
    Get simplified configurations for conditional denoiser tests.
    Reduced to essential combinations for faster testing.
    """
    configs = []

    # Essential conditional combinations - one per denoiser type
    essential_configs = [
        # DPS with LinearSchedule
        {
            "schedule_name": "LinearSchedule",
            "timer_name": "vp",
            "integrator_class": EulerIntegrator,
            "integrator_params": {"stochastic_churn_rate": 0.0},
            "denoiser_class": DPSDenoiser,
        },
        # TMP with CosineSchedule
        {
            "schedule_name": "CosineSchedule",
            "timer_name": "heun",
            "integrator_class": HeunIntegrator,
            "integrator_params": {"stochastic_churn_rate": 1.0, "churn_min": 0.5, "churn_max": 2.0},
            "denoiser_class": TMPDenoiser,
        },
        # FPS with different integrator
        {
            "schedule_name": "LinearSchedule",
            "timer_name": "vp",
            "integrator_class": DDIMIntegrator,
            "integrator_params": {},
            "denoiser_class": FPSDenoiser,
        },
    ]

    for config_dict in essential_configs:
        test_id = f"{config_dict['denoiser_class'].__name__}_{config_dict['integrator_class'].__name__}_{config_dict['timer_name']}_{config_dict['schedule_name']}"
        configs.append(pytest.param(config_dict, id=test_id))

    return configs
