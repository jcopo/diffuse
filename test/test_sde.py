
import jax.numpy as jnp
import jax.random as random

from diffuse.sde import SDE, LinearSchedule, SDEState


def test_path():
    beta = LinearSchedule(b_min=0.5, b_max=1.5, t0=0.0, T=1.0)
    sde = SDE(beta=beta)

    key = random.PRNGKey(0)
    state = SDEState(position=jnp.array([1.0, 2.0]), t=0.0)
    dt = 0.1

    new_state = sde.path(key, state, dt)

    assert new_state.t == state.t + dt
    assert new_state.position.shape == state.position.shape
