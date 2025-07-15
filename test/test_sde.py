"""Unified tests for SDE forward/backward processes and conditional sampling.

This module combines tests for both basic SDE processes and conditional denoising,
using shared utilities to reduce code duplication.

Mathematical Framework:
- Forward SDE: dx = f(x,t)dt + g(t)dW, where f is drift, g is diffusion coefficient, W is Brownian motion
- Backward SDE: dx = [f(x,t) - g²(t)∇log p_t(x)]dt + g(t)dW̃, where ∇log p_t(x) is the score function
- Conditional sampling: p(x|y) ∝ p(y|x)p(x), implemented via guidance or posterior sampling
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import einops

from examples.gaussian_mixtures.cond_mixture import compute_xt_given_y
from examples.gaussian_mixtures.mixture import sampler_mixtr
from examples.gaussian_mixtures.plotting import plot_2d_mixture_and_samples
from diffuse.diffusion.sde import SDEState
from diffuse.denoisers.denoiser import Denoiser

from .config import get_parametrized_configs, get_conditional_configs, get_test_config
from .test_utils import (
    create_plots,
    validate_distributions,
    compute_and_store_mmd,
    assert_mmd_threshold,
    print_mmd_summary,
    plot_debug,
)

# Enable float64 accuracy for precise tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def collect_mmd():
    """Automatically collect and print MMD distances from all tests.

    Maximum Mean Discrepancy (MMD) measures the difference between two
    probability distributions by comparing their mean embeddings in a
    reproducing kernel Hilbert space (RKHS).
    """
    yield
    print_mmd_summary()


@pytest.fixture
def basic_config():
    """Fixture providing basic unconditional test configuration.

    Sets up parameters for sampling from mixture distribution p(x) = Σᵢ wᵢ N(x; μᵢ, Σᵢ)
    where wᵢ are mixture weights, μᵢ are component means, Σᵢ are covariances.
    """
    return get_test_config(conditional=False)


@pytest.fixture
def forward_config(request):
    """Fixture that creates basic config with schedule parametrization.

    Configures noise schedule β(t) which determines the forward process variance:
    - Linear: β(t) = β_min + (β_max - β_min)t
    - Cosine: β(t) follows cosine annealing schedule
    """
    schedule_name = request.param
    return get_test_config(conditional=False, schedule_name=schedule_name, timer_name="vp")


@pytest.fixture
def backward_config(request):
    """Fixture that creates basic config with full parametrization.

    Configures reverse-time SDE integration using score function s_θ(x,t) ≈ ∇log p_t(x)
    to solve dx = [f(x,t) - g²(t)s_θ(x,t)]dt + g(t)dW̃ backwards from noise to data.
    """
    return get_test_config(conditional=False, **request.param)


@pytest.fixture
def conditional_config(request):
    """Fixture that creates conditional test config from parametrized input.

    Sets up conditional sampling p(x|y) using Bayes' rule and conditional score:
    ∇log p(x|y) = ∇log p(x) + ∇log p(y|x) where p(y|x) is the likelihood.
    """
    return get_test_config(conditional=True, **request.param)


@pytest.fixture
def cond_denoiser_config(request):
    """Fixture that creates conditional denoiser test config from parametrized input.

    Configures denoisers that incorporate measurement y via guidance:
    x_{t-1} = μ_θ(x_t,t) + σ_t ε + λ∇log p(y|x_t) where λ controls conditioning strength.
    """
    return get_test_config(conditional=True, **request.param)


# === Basic SDE Tests ===


@pytest.mark.parametrize("forward_config", ["LinearSchedule", "CosineSchedule"], indirect=True)
def test_forward_sde_mixture(forward_config, plot_if_enabled):
    """Test forward SDE process with different noise schedules.

    Validates the forward noising process: x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
    where ᾱ_t = ∏ᵢ₌₁ᵗ(1-βᵢ) and ε ~ N(0,I). Tests that samples at time t
    follow the expected marginal distribution p_t(x_t|x_0).
    """
    # Generate initial samples
    samples_mixt = sampler_mixtr(forward_config.key, forward_config.mix_state, forward_config.n_samples)

    # Setup noising process
    keys = jax.random.split(forward_config.key, forward_config.n_samples * forward_config.n_steps).reshape(
        (forward_config.n_samples, forward_config.n_steps, -1)
    )
    t0 = jnp.array([forward_config.t_init] * forward_config.n_samples)
    state_mixt = SDEState(position=samples_mixt, t=t0)

    # Run forward process
    noised_samples = jax.vmap(jax.vmap(forward_config.sde.path, in_axes=(0, None, 0)), in_axes=(0, 0, None))(
        keys, state_mixt, forward_config.ts
    )
    noised_positions = einops.rearrange(noised_samples.position, "n_samples n_steps d -> n_steps n_samples d")

    # Create plots using unified plotting function
    plot_title = f"Forward SDE - {forward_config.schedule_name}"
    create_plots(forward_config, noised_positions, plot_title, plot_if_enabled)


@pytest.mark.parametrize("backward_config", get_parametrized_configs(), indirect=True)
def test_backward_sde_mixture(backward_config, plot_if_enabled):
    """Test backward SDE process with different integrators and schedules.

    Validates reverse-time integration dx = [f(x,t) - g²(t)∇log p_t(x)]dt + g(t)dW̃
    using numerical schemes (Euler, DDIM, DPM++, Heun). Tests that generated samples
    x_0 ~ p_data when starting from x_T ~ N(0,I) and integrating backwards.
    """
    x0_shape = backward_config.mix_state.means.shape[1]

    # Setup denoising process
    integrator = backward_config.integrator_class(
        sde=backward_config.sde, timer=backward_config.timer, **backward_config.integrator_params
    )
    denoise = Denoiser(
        integrator=integrator, sde=backward_config.sde, score=backward_config.score, x0_shape=(x0_shape,)
    )

    # Generate samples
    key_samples, _ = jax.random.split(backward_config.key)
    _, hist_position = denoise.generate(key_samples, backward_config.n_steps, backward_config.n_samples)
    hist_position = hist_position.squeeze()

    # Create plots
    plot_title = f"Backward SDE - {backward_config.integrator_class.__name__} (Timer: {backward_config.timer_name}, Schedule: {backward_config.schedule_name})"
    create_plots(backward_config, hist_position, plot_title, plot_if_enabled)


# === Conditional SDE Tests ===


@pytest.mark.parametrize("conditional_config", get_parametrized_configs(), indirect=True)
def test_backward_sde_conditional_mixture(conditional_config, plot_if_enabled):
    """Test backward SDE for conditional mixture using ground truth conditional score.

    Validates conditional sampling p(x|y) = p(y|x)p(x)/p(y) by using the exact
    conditional score ∇log p(x|y) = ∇log p(x) + ∇log p(y|x). Compares generated
    samples against analytical posterior distribution using MMD.
    """
    # Generate random keys
    key_gen, key_samples = jax.random.split(conditional_config.key)

    # Use pre-configured denoiser directly (with conditional score)
    state, hist_position = conditional_config.denoiser.generate(
        key_gen, conditional_config.n_steps, conditional_config.n_samples
    )
    hist_position = hist_position.squeeze()

    # Create plots
    plot_title = f"Conditional SDE - {conditional_config.integrator_class.__name__} (Timer: {conditional_config.timer_name}, Schedule: {conditional_config.schedule_name})"
    create_plots(conditional_config, hist_position, plot_title, plot_if_enabled)

    # Generate reference samples from posterior
    samples_from_posterior = sampler_mixtr(
        key_samples, conditional_config.posterior_state, conditional_config.n_samples
    )

    # Debug plot to compare generated vs reference samples
    debug_title = f"Conditional Comparison - {conditional_config.integrator_class.__name__}"
    plot_debug(
        state.integrator_state.position, samples_from_posterior, conditional_config, debug_title, plot_if_enabled
    )

    # Compute and validate MMD distance
    result_key_parts = [
        conditional_config.integrator_class.__name__,
        conditional_config.schedule_name,
        conditional_config.timer_name,
    ]
    mmd_distance = compute_and_store_mmd(conditional_config, state, samples_from_posterior, result_key_parts)

    assert_mmd_threshold(mmd_distance, 0.05)


@pytest.mark.parametrize("cond_denoiser_config", get_conditional_configs(), indirect=True)
def test_backward_conditional_denoisers(cond_denoiser_config, plot_if_enabled):
    """Test conditional denoisers with actual measurement conditioning.

    Validates guided diffusion using approximate conditional score:
    ∇log p(x_t|y) ≈ ∇log p(x_t) + λ∇_{x_t}log p(y|x_t)
    where the likelihood gradient guides sampling toward measurements y.
    Tests DPS (Diffusion Posterior Sampling) using MMD for evaluation.
    """
    # Generate random keys
    key_meas, key_gen, key_samples = jax.random.split(cond_denoiser_config.key, 3)

    # Generate observation for measurement state
    measurement_state = cond_denoiser_config.measurement_state

    # Use pre-configured conditional denoiser (with conditioning handled by denoiser)
    state, hist_position = cond_denoiser_config.cond_denoiser.generate(
        key_gen, measurement_state, cond_denoiser_config.n_steps, cond_denoiser_config.n_samples
    )
    hist_position = hist_position.squeeze()

    # Create plots
    plot_title = f"{cond_denoiser_config.denoiser_class.__name__} - {cond_denoiser_config.integrator_class.__name__} (Timer: {cond_denoiser_config.timer_name}, Schedule: {cond_denoiser_config.schedule_name})"
    create_plots(cond_denoiser_config, hist_position, plot_title, plot_if_enabled)

    # Compute posterior for the actual measurement y
    end_state = compute_xt_given_y(cond_denoiser_config.posterior_state, cond_denoiser_config.sde, 0.0)
    samples_from_posterior = sampler_mixtr(key_samples, end_state, cond_denoiser_config.n_samples)

    # Debug plot to compare conditional denoiser results vs reference
    debug_title = f"{cond_denoiser_config.denoiser_class.__name__} vs Reference"
    plot_debug(
        state.integrator_state.position, samples_from_posterior, cond_denoiser_config, debug_title, plot_if_enabled
    )

    # Compute and validate MMD distance
    result_key_parts = [
        cond_denoiser_config.denoiser_class.__name__,
        cond_denoiser_config.integrator_class.__name__,
        cond_denoiser_config.schedule_name,
        cond_denoiser_config.timer_name,
    ]
    mmd_distance = compute_and_store_mmd(
        cond_denoiser_config, state, samples_from_posterior, result_key_parts, print_result=True
    )

    assert_mmd_threshold(mmd_distance, 0.05)


# === Distribution Validation Tests ===


def test_distribution_validation_basic(basic_config):
    """Test distribution validation for basic mixture using statistical tests.

    Validates that generated samples follow the target distribution using:
    - Kolmogorov-Smirnov test: D_n = sup_x|F_n(x) - F(x)| where F_n is empirical CDF
    - Two-sample tests comparing generated vs reference samples
    - P-values test the null hypothesis that samples come from the same distribution
    """
    # This is a smoke test to ensure our validation utilities work
    from examples.gaussian_mixtures.mixture import cdf_t

    # Generate some test samples
    samples = sampler_mixtr(basic_config.key, basic_config.mix_state, 100)

    # Create mock position array (simplified for testing)
    position = samples.reshape(1, -1, basic_config.d)

    # Test validation function (should pass since samples come from the right distribution)
    validate_distributions(
        position,
        basic_config.timer,
        1,  # n_steps = 1 for simple test
        [0.0],  # single percentile
        lambda x, t: cdf_t(x, t, basic_config.mix_state, basic_config.sde),
        p_threshold=0.001,  # Very lenient for this test
    )
