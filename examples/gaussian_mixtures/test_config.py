import jax
import jax.numpy as jnp

from diffuse.diffusion.sde import SDE, LinearSchedule, CosineSchedule
from diffuse.integrator.deterministic import (
    DDIMIntegrator,
    DPMpp2sIntegrator,
    EulerIntegrator,
    HeunIntegrator,
)
from diffuse.integrator.stochastic import EulerMaruyamaIntegrator
from diffuse.timer.base import HeunTimer, VpTimer
from diffuse.denoisers.cond import DPSDenoiser, FPSDenoiser, TMPDenoiser

from examples.gaussian_mixtures.forward_models.matrix_product import MatrixProduct
from examples.gaussian_mixtures.initialization import (
    init_simple_mixture,
    init_grid_mixture,
    init_bimodal_setup,
)


# Simple, unified configuration
CONFIG = {
    "schedules": {
        "LinearSchedule": {"b_min": 0.0001, "b_max": 20.0},
        "CosineSchedule": {"b_min": 0.1, "b_max": 20.0},
    },
    "integrators": [
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
    ],
    "timers": [
        (
            "vp",
            lambda n_steps, t_final: VpTimer(n_steps=n_steps, eps=0.001, tf=t_final),
        ),
        (
            "heun",
            lambda n_steps, t_final: HeunTimer(n_steps=n_steps, rho=7.0, sigma_min=0.002, sigma_max=1.0),
        ),
    ],
    "cond_denoisers": [DPSDenoiser, TMPDenoiser, FPSDenoiser],
    "percentiles": [0.0, 0.05, 0.1, 0.3, 0.6, 0.7, 0.73, 0.75, 0.8, 0.9],
    "defaults": {
        "t_init": 0.0,
        "t_final": 1.0,
        "n_samples": 300,
        "n_steps": 300,
        "d": 2,
        "sigma_y": 0.1,
        "space_min": -5,
        "space_max": 5,
        "space_points": 300,
    },
}


def create_sde(schedule_name, t_final=1.0):
    """Create SDE with given schedule and final time."""
    schedule_params = CONFIG["schedules"][schedule_name]
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


def create_timer(timer_name, n_steps, t_final=1.0):
    """Create timer with given parameters."""
    timer_map = dict(CONFIG["timers"])
    if timer_name in timer_map:
        return timer_map[timer_name](n_steps, t_final)
    else:
        raise ValueError(f"Unknown timer: {timer_name}")


def create_mixture(key, d=2):
    """Create mixture based on dimensionality."""
    if d > 2:
        return init_grid_mixture(key, d)
    else:
        return init_simple_mixture(key, d)


def create_basic_setup(key=None, **overrides):
    """
    Create basic test setup with consistent interface.

    Args:
        key: JAX random key
        **overrides: Override any default parameters

    Returns:
        Dict with all necessary test objects
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    # Merge defaults with overrides
    params = CONFIG["defaults"].copy()
    params.update(overrides)

    # Create mixture
    # mix_state = create_mixture(key, params["d"])
    mix_state, A, y_target, sigma_y = init_bimodal_setup(key, params["d"])

    # Create forward model for conditional tests
    # key_obs = jax.random.split(key)[1]
    # A = jax.random.normal(key_obs, (1, params["d"]))
    forward_model = MatrixProduct(A=A, std=sigma_y)
    params["y_target"] = y_target

    # Create space grid
    space = jnp.linspace(params["space_min"], params["space_max"], params["space_points"])

    # Create time grid
    ts = jnp.linspace(params["t_init"], params["t_final"], params["n_steps"])

    return {
        "key": key,
        "mix_state": mix_state,
        "forward_model": forward_model,
        "A": A,
        "space": space,
        "ts": ts,
        "perct": CONFIG["percentiles"],
        **params,
    }
