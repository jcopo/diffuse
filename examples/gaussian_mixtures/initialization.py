import jax
import jax.numpy as jnp
from examples.gaussian_mixtures.mixture import MixState


def init_simple_mixture(key, d=1, n_components=3):
    """
    Create a simple Gaussian mixture with random components.
    
    Args:
        key: JAX random key
        d: Dimensionality 
        n_components: Number of mixture components
    
    Returns:
        MixState: Initialized mixture state
    """
    keys = jax.random.split(key, 3)
    
    # Random means in reasonable range
    means = jax.random.uniform(keys[0], (n_components, d), minval=-3, maxval=3)
    
    # Small random covariances
    if d == 1:
        covs = 0.1 * jax.random.uniform(keys[1], (n_components, 1, 1), minval=0.5, maxval=1.5)
    else:
        # For multivariate case, use identity matrices with small random scaling
        identity = jnp.eye(d)
        scales = 0.1 * jax.random.uniform(keys[1], (n_components, 1, 1), minval=0.5, maxval=1.5)
        covs = scales * identity[None, :, :]
    
    # Normalized random weights
    weights = jax.random.uniform(keys[2], (n_components,))
    weights = weights / jnp.sum(weights)
    
    return MixState(means, covs, weights)


def init_grid_mixture(key, d=2, grid_size=5):
    """
    Create a mixture with components arranged on a grid (for high-dimensional cases).
    
    Args:
        key: JAX random key
        d: Dimensionality
        grid_size: Grid size (total components = grid_size^2)
    
    Returns:
        MixState: Initialized mixture state
    """
    n_components = grid_size * grid_size
    
    # Create grid positions
    grid_1d = jnp.linspace(-2, 2, grid_size)
    grid_positions = jnp.array([(i, j) for i in grid_1d for j in grid_1d])
    
    # Extend to higher dimensions by tiling
    if d > 2:
        repeats = (d + 1) // 2
        means = jnp.tile(grid_positions, (1, repeats))[:, :d]
    else:
        means = grid_positions
    
    # Identity covariance matrices
    covs = jnp.repeat(jnp.eye(d)[None, :, :], n_components, axis=0)
    
    # Random weights
    weights = jax.random.uniform(key, (n_components,))
    weights = weights / jnp.sum(weights)
    
    return MixState(means, covs, weights)


def init_fixed_mixture(d=2):
    """
    Create a fixed mixture for reproducible testing.
    
    Args:
        d: Dimensionality
        
    Returns:
        MixState: Fixed mixture state
    """
    if d == 1:
        means = jnp.array([[-2.0], [0.0], [2.0]])
        covs = jnp.array([[[0.5]], [[0.3]], [[0.7]]])
    elif d == 2:
        means = jnp.array([[-1.0, -1.0], [1.0, 1.0], [2.0, -2.0]])
        covs = jnp.array([
            [[0.5, 0.1], [0.1, 0.5]],
            [[0.7, -0.1], [-0.1, 0.7]], 
            [[0.3, 0.0], [0.0, 1.0]]
        ])
    else:
        # For higher dimensions, extend 2D case
        means_2d = jnp.array([[-1.0, -1.0], [1.0, 1.0], [2.0, -2.0]])
        padding = jnp.zeros((3, d-2))
        means = jnp.concatenate([means_2d, padding], axis=1)
        
        # Identity covariance for extra dimensions
        covs = jnp.repeat(jnp.eye(d)[None, :, :], 3, axis=0)
    
    weights = jnp.array([0.3, 0.4, 0.3])
    
    return MixState(means, covs, weights)