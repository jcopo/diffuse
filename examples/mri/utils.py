from dataclasses import dataclass
from functools import partial

import einops
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.base_forward_model import MeasurementState



