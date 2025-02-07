from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array
from examples.mri.forward_models.base import baseMask, PARAMS_SIGMA_VERTICAL
from examples.mri.forward_models.base import MeasurementState
from functools import partial
import optax
def generate_vertical_line_soft(x_pos, shape, sigma):
    H, W = shape

    x_pos = x_pos * W - W/2
    xs = jnp.linspace(-W/2, W/2, W)
    ys = jnp.linspace(-H/2, H/2, H)
    _, grid_x = jnp.meshgrid(ys, xs, indexing='ij')
    
    # Create the vertical line
    line_mask = jax.nn.sigmoid(-jnp.abs(grid_x - x_pos) / sigma)
    
    # Apply the edge constraints smoothly
    return line_mask

def generate_vertical_line_hard(x_pos, shape):
    H, W = shape
    pixel_x = jnp.floor(x_pos * (W - 1)).astype(int)
    mask = jnp.zeros((H, W))
    mask = mask.at[:, pixel_x].set(1.0)
    return mask

@partial(jax.vmap, in_axes=(0, None, None))
def generate_vertical_line(x_pos, shape, sigma):
    mask_hard = generate_vertical_line_hard(x_pos, shape)
    mask_soft = generate_vertical_line_soft(x_pos, shape, sigma)
    return mask_soft + jax.lax.stop_gradient(mask_hard - mask_soft)

def generate_centered_rectangle(shape, width_frac):
    H, W = shape
    width = int(W * width_frac)
    
    mask = jnp.zeros((H, W))
    start_x = (W - width) // 2
    
    mask = mask.at[:, start_x:start_x+width].set(1.0)
    return mask

@dataclass
class maskVertical(baseMask):
    num_lines: int
    img_shape: tuple  # (H, W, C)
    task: str
    data_model: str

    def init_design(self, key: PRNGKeyArray) -> Array:
        # Generate x positions for vertical lines
        x_positions = jax.random.uniform(
            key, shape=(self.num_lines,), minval=0, maxval=1
        )
        return x_positions
    
    def projection_design(self, xi: Array) -> Array:
        return optax.projections.projection_box(xi, lower=0.0, upper=1.0)
    
    def init_measurement(self, ground_truth: Array) -> MeasurementState:
        mask = generate_centered_rectangle(self.img_shape[:-1], 0.05)
        y = self.measure_from_mask(mask, ground_truth)
        return MeasurementState(y=y, mask_history=mask)

    def make(self, xi: Array) -> Array:

        lines = generate_vertical_line(
            xi, self.img_shape[:-1], 40, # PARAMS_SIGMA_VERTICAL[self.data_model]
        )

        # Accumulate lines while preventing overlap using jax.lax.scan
        def scan_fn(hist_lines, line):
            inv_hist_lines = 1 - hist_lines
            new_hist_lines = inv_hist_lines * line + hist_lines
            return new_hist_lines, None

        hist_lines = jnp.zeros(self.img_shape[:-1])
        final_hist_lines, _ = jax.lax.scan(scan_fn, hist_lines, lines)

        return final_hist_lines