from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array
from examples.mri.forward_models.base import baseMask, PARAMS_SIGMA_VERTICAL
from examples.mri.forward_models.base import MeasurementState

def generate_vertical_line_soft(x_pos, shape, sigma):
    H, W = shape
    x_pos = x_pos * W - W/2

    xs = jnp.linspace(-W/2, W/2, W)
    ys = jnp.linspace(-H/2, H/2, H)
    _, grid_x = jnp.meshgrid(ys, xs, indexing='ij')
    
    mask_x = jax.nn.sigmoid(-jnp.abs(grid_x - x_pos) / sigma)
    return mask_x

def generate_vertical_line_hard(x_pos, shape):
    H, W = shape
    pixel_x = jnp.floor(x_pos * (W - 1)).astype(int)
    mask = jnp.zeros((H, W))
    mask = mask.at[:, pixel_x].set(1.0)
    return mask

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
            key, shape=(self.num_lines,), minval=-0.1, maxval=0.1
        )
        return x_positions
    
    def init_measurement(self, ground_truth: Array) -> MeasurementState:
        mask = generate_centered_rectangle(self.img_shape[:-1], 0.05)
        y = self.measure_from_mask(mask, ground_truth)
        return MeasurementState(y=y, mask_history=mask)

    def make(self, xi: Array) -> Array:
        xi = jax.nn.sigmoid(xi)

        lines = jax.vmap(generate_vertical_line, in_axes=(0, None, None))(
            xi, self.img_shape[:-1], PARAMS_SIGMA_VERTICAL[self.data_model]
        )

        # Accumulate lines while preventing overlap
        hist_lines = jnp.zeros(self.img_shape[:-1])
        for line in lines:
            inv_hist_lines = 1 - hist_lines
            hist_lines = inv_hist_lines * line + hist_lines

        return hist_lines