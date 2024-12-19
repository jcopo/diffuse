from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from diffuse.integrator.base import IntegratorState


@dataclass
class Euler:
    """Euler deterministic integrator for ODEs"""

