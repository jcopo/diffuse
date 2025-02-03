from dataclasses import dataclass

import jax
from jaxtyping import Array, PRNGKeyArray
import jax.numpy as jnp
from examples.mri.forward_models.base import baseMask, PARAMS_SIGMA_RDM

@dataclass
class maskCartesian(baseMask):
    img_shape: tuple # (H, W, C)
    task: str
    data_model: str
    budget: float = 0.01  # Add budget parameter, default to 1%
    temperature: float = 0.1  # Temperature for Gumbel-Softmax

    def init_design(self, key: PRNGKeyArray) -> Array:
        H, W, _ = self.img_shape
        return jax.random.uniform(key, shape=(H, W))

    def make(self, xi: Array, key: PRNGKeyArray) -> Array:
        H, W, _ = self.img_shape
        u = jax.random.uniform(key, shape=(H, W))
        mask = self.budget * xi < u
        mask = jax.lax.stop_gradient(mask.astype(jnp.float32))
        
        return mask
        