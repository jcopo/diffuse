from typing import NamedTuple, Tuple

import jax
from jax import numpy as jnp
import optax
from jaxtyping import PRNGKeyArray, Array

from diffuse.denoisers.denoiser import Denoiser
from diffuse.base_forward_model import ForwardModel


class BEDState(NamedTuple):
    thetas: Array
    weights: Array
    cntrst_thetas: Array
    cntrst_weights: Array
    design: Array
    opt_state: optax.OptState

class ExperimentOptimizer:
    denoiser: Denoiser
    mask: ForwardModel
    optimizer: optax.GradientTransformation
    base_shape: Tuple[int, ...]

    def init(self, rng_key: PRNGKeyArray, n_samples: int, n_samples_cntrst: int) -> BEDState:
        design = self.mask.init_design(rng_key)
        opt_state = self.optimizer.init(design)

        thetas, cntrst_thetas = _init_start_time(rng_key, n_samples, n_samples_cntrst, self.base_shape)
        weights = jnp.ones(n_samples) / n_samples
        cntrst_weights = jnp.ones(n_samples_cntrst) / n_samples_cntrst

        return BEDState(thetas=thetas, weights=weights, cntrst_thetas=cntrst_thetas, cntrst_weights=cntrst_weights, design=design, opt_state=opt_state)


    def step(self, state: BEDState, rng_key: PRNGKeyArray, measurement_state: MeasurementState) -> BEDState:


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