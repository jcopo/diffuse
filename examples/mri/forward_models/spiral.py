from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from examples.mri.forward_models.base import baseMask, PARAMS_FOV, PARAMS_KMAX, PARAMS_SIGMA


@partial(jax.vmap, in_axes=(0, 0, 0, None, None))
def generate_spiral_arm(angle_offset, k_max, fov, img_shape, sigma):
    y, x = jnp.mgrid[: img_shape[0], : img_shape[1]]
    
    center_x = img_shape[1] // 2
    center_y = img_shape[0] // 2
    
    # Convert grid coordinates to polar coordinates
    x_centered = x - center_x
    y_centered = y - center_y
    r = jnp.sqrt(x_centered**2 + y_centered**2)
    theta = jnp.arctan2(y_centered, x_centered)
    
    # Normalize coordinates
    r_norm = r / center_x
    
    # Unwrap theta to [-π, π]
    theta = theta - angle_offset
    theta = theta - 2 * jnp.pi * jnp.floor((theta + jnp.pi) / (2 * jnp.pi))
    
    # Calculate spiral equation
    # For an Archimedean spiral: r = a * theta
    # We want the spiral to make k_max turns within FOV
    a = fov / (2 * jnp.pi * k_max)
    
    # For each point, find which turn of the spiral it's closest to
    n = jnp.round(r_norm / (2 * jnp.pi * a))
    theta_n = theta + 2 * jnp.pi * n
    
    # Calculate the radius of the spiral at this angle
    r_spiral = a * theta_n
    
    # Calculate distance to spiral
    distance = jnp.abs(r_norm - r_spiral)
    
    # Apply sigmoid for smooth transition
    sharpness = 30000000.0
    spiral_arm = jax.nn.sigmoid(-sharpness * (distance - sigma))
    
    # Apply circular mask
    mask_radius = 0.95
    circle_mask = jax.nn.sigmoid(-sharpness * (r_norm - mask_radius))
    
    # Add dense sampling at the center
    center_radius = 0.01  # Controls size of central region
    center_density = jax.nn.sigmoid(-sharpness * (r_norm - center_radius))
    
    # Combine center sampling with spiral arm
    return jnp.maximum(spiral_arm * circle_mask, center_density)


@dataclass
class maskSpiral(baseMask):
    num_spiral: int
    img_shape: tuple
    task: str
    data_model: str

    def __post_init__(self):
        min_dim = min(self.img_shape[:-1])
        self.sigma = PARAMS_SIGMA[self.data_model] / min_dim

    def init_design(self, key: PRNGKeyArray) -> Array:
        k1, k2, k3 = jax.random.split(key, 3)
        
        # FOV controls spiral spacing
        fov = jax.random.uniform(k1, shape=(1,), minval=PARAMS_FOV[self.data_model]['minval'], maxval=PARAMS_FOV[self.data_model]['maxval'])[0]
        
        # k_max controls number of turns
        k_max = jax.random.uniform(k2, shape=(1,), minval=PARAMS_KMAX[self.data_model]['minval'], maxval=PARAMS_KMAX[self.data_model]['maxval'])[0]
        
        # base_angle controls the starting angle
        base_angle = jax.random.uniform(k3, shape=(1,), minval=0.0, maxval=2*jnp.pi)[0]
        
        return jnp.array([fov, k_max, base_angle])

    def make(self, xi: Array) -> Array:
        fov, k_max, base_angle = xi
        
        # Generate angle offsets for each spiral arm
        angles = base_angle + jnp.arange(self.num_spiral) * (2 * jnp.pi / self.num_spiral)
        k_maxs = jnp.full_like(angles, k_max)
        fovs = jnp.full_like(angles, fov)
        
        # Generate all spiral arms
        spiral_arms = generate_spiral_arm(angles, k_maxs, fovs, self.img_shape[:-1], self.sigma)
        
        # Combine spiral arms using a smooth, differentiable max operation
        mask = jnp.max(spiral_arms, axis=0)
        
        return mask
