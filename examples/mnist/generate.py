
import jax

from diffuse.denoisers.denoiser import Denoiser
from diffuse.integrator.stochastic import EulerMaruyama
from diffuse.utils.plotting import plot_lines
import numpy as np
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from diffuse.diffusion.sde import SDE, LinearSchedule
from diffuse.neural_network.unet import UNet



def initialize_experiment(key: PRNGKeyArray, n_t: int):
    # Load MNIST dataset
    data = np.load("dataset/mnist.npz")
    xs = jnp.array(data["X"])
    xs = xs.reshape(xs.shape[0], xs.shape[1], xs.shape[2], 1)  # Add channel dimension
    ground_truth = jax.random.choice(key, xs)

    # Initialize parameters
    tf = 2.0

    # Define beta schedule and SDE
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)

    # Initialize ScoreNetwork
    score_net = UNet(tf / n_t, 64, upsampling="pixel_shuffle")
    nn_trained = jnp.load("weights/ann_2999.npz", allow_pickle=True)
    params = nn_trained["params"].item()

    # Define neural network score function
    def nn_score(x, t):
        return score_net.apply(params, x, t)

    # Set up conditional SDE
    sde = SDE(beta=beta, tf=tf)

    return sde, ground_truth, tf, n_t, nn_score


def main(n_samples=150, n_t=300):
    key_init = jax.random.PRNGKey(0)
    sde, ground_truth, tf, n_t, nn_score = initialize_experiment(key_init, n_t)

    integrator = EulerMaruyama(sde=sde)
    denoiser = Denoiser(integrator=integrator, sde=sde, score=nn_score, n_steps=n_t, x0_shape=ground_truth.shape)

    keys = jax.random.split(key_init, n_samples)
    vec_generator = jax.jit(jax.vmap(denoiser.generate))
    state, hist = vec_generator(keys)

    plot_lines(state.integrator_state.position)
    return state, hist

if __name__ == "__main__":
    main()
