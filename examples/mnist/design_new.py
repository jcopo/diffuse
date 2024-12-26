import jax
from jax import numpy as jnp
import optax
import numpy as np

from diffuse.neural_network.unet import UNet
from diffuse.diffusion.sde import SDE, LinearSchedule
from diffuse.integrator.stochastic import EulerMaruyama
from diffuse.denoisers.cond_denoiser import CondDenoiser
from diffuse.bayesian_design import ExperimentOptimizer
from examples.mnist.images import SquareMask

from jaxtyping import PRNGKeyArray


SIZE = 7


def initialize_experiment(key: PRNGKeyArray):
    # Load MNIST dataset
    data = np.load("dataset/mnist.npz")
    xs = jnp.array(data["X"])
    xs = xs.reshape(xs.shape[0], xs.shape[1], xs.shape[2], 1)  # Add channel dimension

    # Initialize parameters
    tf = 2.0
    n_t = 300
    dt = tf / n_t

    # Define beta schedule and SDE
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)

    # Initialize ScoreNetwork
    score_net = UNet(dt, 64, upsampling="pixel_shuffle")
    nn_trained = jnp.load("weights/ann_2999.npz", allow_pickle=True)
    params = nn_trained["params"].item()

    # Define neural network score function
    def nn_score(x, t):
        return score_net.apply(params, x, t)

    # Set up mask and measurement
    ground_truth = jax.random.choice(key, xs)
    mask = SquareMask(SIZE, ground_truth.shape)

    # SDE
    sde = SDE(beta=beta, tf=tf)

    return sde, mask, ground_truth, dt, n_t, nn_score


def main(key: PRNGKeyArray):
    # Initialize experiment forward model
    sde, mask, ground_truth, dt, n_t, nn_score = initialize_experiment(key)
    n_samples = 150
    n_samples_cntrst = 151
    num_measurements = 10

    # Conditional Denoiser
    integrator = EulerMaruyama(sde)
    denoiser = CondDenoiser(
        integrator, sde, nn_score, mask
    )  # , n_t, ground_truth.shape)

    # init design
    measurement_state = mask.init_measurement()

    # ExperimentOptimizer
    optimizer = optax.chain(optax.adam(learning_rate=0.1), optax.scale(-1))
    experiment_optimizer = ExperimentOptimizer(
        denoiser, mask, optimizer, ground_truth.shape
    )

    exp_state = experiment_optimizer.init(key, n_samples, n_samples_cntrst, dt)

    for i in range(num_measurements):
        optimal_state, hist_implicit = experiment_optimizer.get_design(
            exp_state, key, measurement_state
        )

        # make new measurement
        new_measurement = mask.measure(optimal_state.design, ground_truth)
        measurement_state = mask.update_measurement(
            measurement_state, new_measurement, optimal_state.design
        )

        exp_state = experiment_optimizer.init(key, n_samples, n_samples_cntrst, dt)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    main(key)
