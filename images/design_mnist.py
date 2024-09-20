from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, PRNGKeyArray
from optax import GradientTransformation
import matplotlib.pyplot as plt

from diffuse.filter import generate_cond_sample
from diffuse.sde import SDE, SDEState
from diffuse.conditional import CondSDE
from diffuse.images import SquareMask
from diffuse.optimizer import ImplicitState, impl_step
from diffuse.sde import LinearSchedule
from diffuse.unet import UNet
import einops
import pdb


def initialize_experiment(key):
    # Load MNIST dataset
    data = np.load("dataset/mnist.npz")
    xs = jnp.array(data["X"])
    xs = xs.reshape(xs.shape[0], xs.shape[1], xs.shape[2], 1)  # Add channel dimension

    # Initialize parameters
    tf = 2.0
    n_t = 299

    # Define beta schedule and SDE
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)

    # Initialize ScoreNetwork
    score_net = UNet(tf / n_t, 64, upsampling="pixel_shuffle")
    nn_trained = jnp.load("ann_2999.npz", allow_pickle=True)
    params = nn_trained["params"].item()

    # Define neural network score function
    def nn_score(x, t):
        return score_net.apply(params, x, t)

    # Set up mask and measurement
    ground_truth = jax.random.choice(key, xs)
    mask = SquareMask(10, ground_truth.shape)

    # Set up conditional SDE
    cond_sde = CondSDE(beta=beta, mask=mask, tf=2.0, score=nn_score)
    sde = SDE(beta=beta)

    return sde, cond_sde, mask, ground_truth, tf, n_t, nn_score


def optimize_design(
    key_step: PRNGKeyArray,
    implicit_state: ImplicitState,
    past_y: Array,
    optimizer: GradientTransformation,
    ts: Array,
    dt: float,
    cond_sde: CondSDE,
):
    opt_steps = 5_000
    keys_opt = jax.random.split(key_step, opt_steps)

    def step(new_state, key):
        new_state = impl_step(
            new_state, key, past_y, cond_sde=cond_sde, optx_opt=optimizer, ts=ts, dt=dt
        )
        return new_state, new_state.design

    new_state, hist_design = jax.lax.scan(step, implicit_state, keys_opt)
    return new_state, hist_design


#@jax.jit
def main(key):
    key_init, key_step = jax.random.split(key)

    sde, cond_sde, mask, ground_truth, tf, n_t, nn_score = initialize_experiment(key_init)
    dt = tf / (n_t - 1)
    num_meas = 6
    n_samples = 30
    n_samples_cntrst = 30

    # init design and measurement hist
    design = jax.random.uniform(key_init, (2,), minval=0, maxval=28)
    design_0 = jnp.zeros_like(design)
    y = cond_sde.mask.measure(design_0, ground_truth)

    measurement_history = jnp.zeros((num_meas, *y.shape))
    measurement_history = measurement_history.at[0].set(y)

    # init optimizer
    optimizer = optax.chain(optax.adam(learning_rate=1e-2), optax.scale(-1))
    opt_state = optimizer.init(design)

    # run denoising process to initialize the queue of implicit opt
    init_samples = jax.random.normal(key_init, (n_samples+n_samples_cntrst, *ground_truth.shape))
    tfs = jnp.zeros((n_samples+n_samples_cntrst,)) + tf
    ts = jnp.linspace(0, tf, n_t)
    dts = jnp.diff(ts)
    state_f = SDEState(position=init_samples, t=tfs)

    # denoise
    revert_sde = partial(sde.reverso, score=nn_score, dts=dts)

    # make right shape
    keys = jax.random.split(key_init, n_samples+n_samples_cntrst)
    _, state_Ts = jax.vmap(revert_sde)(keys, state_f)
    state_Ts = jax.tree.map(lambda arr: einops.rearrange(arr, 'n h ... -> h n ...'), state_Ts)

    # init thetas
    thetas = state_Ts.position[:, :n_samples]
    cntrst_thetas = state_Ts.position[:, n_samples:]
    implicit_state = ImplicitState(thetas, cntrst_thetas, design, opt_state)

    # stock in joint_y all measurements
    joint_y = y
    design_step = jax.jit(partial(optimize_design, optimizer=optimizer, ts=ts, dt=dt, cond_sde=cond_sde))
    for n_meas in range(num_meas):

        key_noise, key_opt, key_gen = jax.random.split(key_step, 3)
        # make noised path for measurements
        keys_noise = jax.random.split(key_noise, n_t)
        state_0 = SDEState(joint_y, jnp.zeros_like(y))
        past_y = jax.vmap(sde.path, in_axes=(0, None, 0))(keys_noise, state_0, ts)
        past_y = SDEState(past_y.position[::-1], past_y.t)


        # optimize design
        optimal_state, hist_implicit = design_step(key_opt, implicit_state, past_y)
        opt_hist = hist_implicit

        # make new measurement
        new_measurement = cond_sde.mask.measure(optimal_state.design, ground_truth)
        measurement_history = measurement_history.at[n_meas].set(new_measurement)

        # add measured data to joint_y
        joint_y = cond_sde.mask.restore(optimal_state.design, joint_y, new_measurement)
        print(joint_y[10, 20])

        # logging
        print(f"Design_start: {design} Design_end:{optimal_state.design}")
        fig, axs = plt.subplots(1, 2)
        ax1, ax2 = axs
        ax1.scatter(opt_hist[:, 0], opt_hist[:, 1], marker='+')
        ax1.imshow(ground_truth, cmap="gray")
        ax2.scatter(opt_hist[:, 0], opt_hist[:, 1], marker='+')
        ax2.imshow(joint_y, cmap="gray")

        plt.tight_layout()
        plt.show()

        print(jnp.max(joint_y))

        # reinitiazize implicit state
        design = jax.random.uniform(key_step, (2,), minval=0, maxval=28)
        opt_state = optimizer.init(design)
        key_t, key_c = jax.random.split(key_gen)
        thetas = generate_cond_sample(joint_y, optimal_state.design, key_t, cond_sde, ground_truth.shape, n_t, n_samples)[1][0]
        cntrst_thetas = generate_cond_sample(joint_y, optimal_state.design, key_c, cond_sde, ground_truth.shape, n_t, n_samples_cntrst)[1][0]
        key_step, _ = jax.random.split(key_step)

        implicit_state = ImplicitState(thetas, cntrst_thetas, design, opt_state)
        for i in range(20):
            plotter_line(implicit_state.thetas[:, i])

        return implicit_state


rng_key = key = jax.random.PRNGKey(0)
state = main(rng_key)
