from typing import NamedTuple, Tuple
from dataclasses import dataclass
import jax
from jax import numpy as jnp
import optax
from optax import GradientTransformation
from jaxtyping import PRNGKeyArray, Array

from diffuse.denoisers.cond_denoiser import CondDenoiser, CondDenoiserState
from diffuse.base_forward_model import ForwardModel, MeasurementState
from diffuse.integrator.base import IntegratorState
from diffuse.utils.plotting import plot_lines


class BEDState(NamedTuple):
    denoiser_state: CondDenoiserState
    cntrst_denoiser_state: CondDenoiserState
    design: Array
    opt_state: optax.OptState


def _vmapper(fn, type):
    def _set_axes(path, value):
        # Vectorize only particles and rng_key fields
        if any(field in str(path) for field in ["position", "rng_key", "weights"]):
            return 0
        return None

    # Create tree with selective vectorization
    in_axes = jax.tree_util.tree_map_with_path(_set_axes, type)
    return jax.vmap(fn, in_axes=(in_axes, None))


@dataclass
class ExperimentOptimizer:
    denoiser: CondDenoiser
    mask: ForwardModel
    optimizer: GradientTransformation
    base_shape: Tuple[int, ...]

    def init(
        self, rng_key: PRNGKeyArray, n_samples: int, n_samples_cntrst: int, dt: float
    ) -> BEDState:
        design = self.mask.init_design(rng_key)
        opt_state = self.optimizer.init(design)

        key_init, key_t, key_c = jax.random.split(rng_key, 3)
        thetas, cntrst_thetas = _init_start_time(
            key_t, n_samples, n_samples_cntrst, self.base_shape
        )
        denoiser_state = self.denoiser.init(thetas, key_init, dt)
        cntrst_denoiser_state = self.denoiser.init(cntrst_thetas, key_c, dt)

        return BEDState(
            denoiser_state=denoiser_state,
            cntrst_denoiser_state=cntrst_denoiser_state,
            design=design,
            opt_state=opt_state,
        )

    def step(
        self,
        state: BEDState,
        rng_key: PRNGKeyArray,
        measurement_state: MeasurementState,
    ) -> BEDState:
        denoiser_state, cntrst_denoiser_state = (
            state.denoiser_state,
            state.cntrst_denoiser_state,
        )
        y = measurement_state.y
        design, opt_state = state.design, state.opt_state
        rng_key, key_t, key_c, key_d = jax.random.split(rng_key, 4)

        # step theta
        t = state.denoiser_state.integrator_state.t
        score_likelihood = self.denoiser.posterior_logpdf(rng_key, t, y, design)
        denoiser_state = _vmapper(self.denoiser.step, denoiser_state)(
            denoiser_state, score_likelihood
        )

        # update design
        thetas = denoiser_state.integrator_state.position
        cntrst_thetas = cntrst_denoiser_state.integrator_state.position
        design, opt_state, y_cntrst = calculate_and_apply_gradient(
            thetas, cntrst_thetas, design, self.mask, self.optimizer, opt_state
        )

        #jax.debug.print("thetas: {}", thetas)
        # jax.experimental.io_callback( plot_lines, None, cntrst_thetas, t)
        # step cntrst_theta
        score_likelihood = self.denoiser.pooled_posterior_logpdf(
            key_t, t, y_cntrst, y, design
        )
        cntrst_denoiser_state = _vmapper(self.denoiser.step, cntrst_denoiser_state)(
            cntrst_denoiser_state, score_likelihood
        )
        denoiser_state, cntrst_denoiser_state = _fix_time(denoiser_state, cntrst_denoiser_state)

        return BEDState(
            denoiser_state=denoiser_state,
            cntrst_denoiser_state=cntrst_denoiser_state,
            design=design,
            opt_state=opt_state,
        )

    def get_design(
        self,
        state: BEDState,
        rng_key: PRNGKeyArray,
        measurement_state: MeasurementState,
        n_steps: int = 100,
    ):
        def step(state, rng_key):
            state = self.step(state, rng_key, measurement_state)
            return state, state.denoiser_state.integrator_state.position

        keys = jax.random.split(rng_key, n_steps)
        return jax.lax.scan(step, state, keys)


def _init_start_time(
    key_init: PRNGKeyArray,
    n_samples: int,
    n_samples_cntrst: int,
    ground_truth_shape: Tuple[int, ...],
) -> Tuple[Array, Array]:
    """
    Initialize thetas for just the start time of the conditional sampling
    """
    key_t, key_c = jax.random.split(key_init)
    thetas = jax.random.normal(key_t, (n_samples, *ground_truth_shape))
    cntrst_thetas = jax.random.normal(key_c, (n_samples_cntrst, *ground_truth_shape))
    return thetas, cntrst_thetas


def calculate_and_apply_gradient(
    thetas: Array,
    cntrst_thetas: Array,
    design: Array,
    mask: ForwardModel,
    optx_opt: GradientTransformation,
    opt_state: optax.OptState,
):
    grad_xi_score = jax.grad(information_gain, argnums=2, has_aux=True)
    grad_xi, ys = grad_xi_score(thetas, cntrst_thetas, design, mask)
    updates, new_opt_state = optx_opt.update(grad_xi, opt_state, design)
    new_design = optax.apply_updates(design, updates)

    return new_design, new_opt_state, ys


def information_gain(
    theta: Array, cntrst_theta: Array, design: Array, mask: ForwardModel
):
    r"""
    Information gain estimator
    Estimator \sum_i log p(y_i | theta_i, design) - \sum_j w_{ij} log p(y_i | theta_j, design)
    """
    # sample y from p(y, theta_)
    y_ref = mask.measure(design, theta)
    logprob_ref = mask.logprob_y(theta, y_ref, design)
    logprob_target = jax.vmap(mask.logprob_y, in_axes=(None, 0, None))(
        cntrst_theta, y_ref, design
    )
    # logprob_target = jax.scipy.special.logsumexp(logprob_target, )
    logprob_means = jnp.mean(logprob_target, axis=0, keepdims=True)
    log_weights = jax.lax.stop_gradient(logprob_target - logprob_means)
    # _norm = jax.scipy.special.logsumexp(log_weights, keepdims=True)
    _norm = jax.scipy.special.logsumexp(log_weights, axis=1, keepdims=True)
    log_weights = log_weights - _norm

    weighted_logprobs = jnp.mean(log_weights + logprob_target, axis=1)

    return (logprob_ref - weighted_logprobs).mean(), y_ref


def _fix_time(denoiser_state: CondDenoiserState, cntrst_denoiser_state: CondDenoiserState):
    # Create new integrator states with fixed time
    new_denoiser_integrator = denoiser_state.integrator_state._replace(
        t=denoiser_state.integrator_state.t[0],
        dt=denoiser_state.integrator_state.dt[0]
    )
    new_cntrst_integrator = cntrst_denoiser_state.integrator_state._replace(
        t=cntrst_denoiser_state.integrator_state.t[0],
        dt=cntrst_denoiser_state.integrator_state.dt[0]
    )

    # Return new denoiser states with updated integrator states
    return (
        denoiser_state._replace(integrator_state=new_denoiser_integrator),
        cntrst_denoiser_state._replace(integrator_state=new_cntrst_integrator)
    )
