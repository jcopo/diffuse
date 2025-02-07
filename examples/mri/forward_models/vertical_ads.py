from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array
from examples.mri.forward_models.base import baseMask
from examples.mri.forward_models.base import MeasurementState

def generate_vertical_line_discrete(pos_idx: int, shape) -> Array:
    """Generate a vertical line at a discrete position.
    
    Args:
        pos_idx: Integer index of the position (0 to W-1)
        shape: Tuple of (H, W) for the image dimensions
    """
    H, W = shape
    mask = jnp.zeros((H, W))
    mask = mask.at[:, pos_idx].set(1.0)
    return mask

def generate_centered_rectangle(shape, width_frac):
    H, W = shape
    width = int(W * width_frac)
    
    mask = jnp.zeros((H, W))
    start_x = (W - width) // 2
    
    mask = mask.at[:, start_x:start_x+width].set(1.0)
    return mask

@dataclass
class maskVerticalDiscrete(baseMask):
    num_lines: int
    img_shape: tuple  # (H, W, C)
    task: str
    data_model: str

    def init_design(self, key: PRNGKeyArray) -> Array:
        # Generate random discrete positions
        W = self.img_shape[1]
        positions = jax.random.randint(
            key, shape=(self.num_lines,), minval=0, maxval=W
        )
        return positions
    
    def init_measurement(self, ground_truth: Array) -> MeasurementState:
        mask = generate_centered_rectangle(self.img_shape[:-1], 0.05)
        y = self.measure_from_mask(mask, ground_truth)
        return MeasurementState(y=y, mask_history=mask)

    def make(self, xi: Array) -> Array:
        # xi should contain discrete indices
        xi = xi.astype(jnp.int32)
        
        lines = jax.vmap(generate_vertical_line_discrete, in_axes=(0, None))(
            xi, self.img_shape[:-1]
        )

        # Accumulate lines while preventing overlap
        hist_lines = jnp.zeros(self.img_shape[:-1])
        for line in lines:
            inv_hist_lines = 1 - hist_lines
            hist_lines = inv_hist_lines * line + hist_lines

        return hist_lines
