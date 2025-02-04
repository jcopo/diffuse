import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array
from dataclasses import dataclass

from examples.mri.forward_models.base import baseMask, PARAMS_SIGMA_RDM, PARAMS_SPARSITY

def rescale(x, sparsity, sigma):
    xbar = x.mean()
    r = sparsity / xbar
    beta = (1-sparsity) / (1 - xbar)

    le_hard = r < 1
    hard_rescale = jnp.where(le_hard, x * r, (1 - (1 - x) * beta))

    le_soft = jax.nn.sigmoid((1 - r) / sigma) 
    soft_rescale = le_soft * x + (1 - le_soft) * (1 - (1 - x) * beta)

    return soft_rescale + jax.lax.stop_gradient(hard_rescale - soft_rescale)

def sampler(rng_key, P, sparsity, sigma=.1):
    P /= P.max()
    P = jax.nn.softplus(P)
    scale_factor = rescale(P, sparsity, sigma)
    P *= scale_factor

    U = jax.random.uniform(rng_key, P.shape)
    M_hard = (U < P).astype(jnp.float32) 

    M_soft = jax.nn.sigmoid((P - U) / sigma)
    
    return M_soft + jax.lax.stop_gradient(M_hard - M_soft)


@dataclass
class maskRandom(baseMask):
    img_shape: tuple
    task: str
    data_model: str

    def init_design(self, key: PRNGKeyArray) -> Array:
        H, W, _ = self.img_shape
        return jax.random.uniform(key, (H, W))
    
    def make(self, xi: Array, key: PRNGKeyArray) -> Array:
        return sampler(key, xi, sparsity=PARAMS_SPARSITY[self.data_model], sigma=PARAMS_SIGMA_RDM[self.data_model])
    