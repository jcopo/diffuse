import os
from functools import partial
from typing import Tuple

import einops
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jax_tqdm import scan_tqdm
from jaxtyping import Array, PRNGKeyArray
import optax
from optax import GradientTransformation

from diffuse.samplopt.conditional import CondSDE
from diffuse.samplopt.optimizer import ImplicitState, impl_one_step, impl_full_scan
from diffuse.diffusion.sde import LinearSchedule, SDE, SDEState
from diffuse.neural_network.unet import UNet

from examples.mri.wmh.create_dataset import WMH
from examples.mri.utils import maskSpiral, plotter_line_measure, plotter_line_obs

config = {
    "modality": "FLAIR",
    "slice_size_template": 49,
    "begin_slice": 26,
    "flair_template_path": "/lustre/fswork/projects/rech/hlp/uha64uw/projet_p/WMH/MNI-FLAIR-2.0mm.nii.gz",
    # "path_dataset": "/Users/geoffroyoudoumanessah/Documents/these/projects/datasets/WMH",
    "path_dataset": "/lustre/fswork/projects/rech/hlp/uha64uw/projet_p/WMH",
    "save_path": "/lustre/fswork/projects/rech/hlp/uha64uw/projet_p/WMH/models/",
    "n_epochs": 4000,
    "batch_size": 1,
    "num_workers": 0,
    "n_t": 32,
    "tf": 2.0,
    "lr": 2e-4,
}


def initialize_experiment(key: PRNGKeyArray):
    wmh = WMH(config)
    wmh.setup()
    xs = wmh.get_train_dataloader()

    # Define beta schedule and SDE
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)

    # Initialize ScoreNetwork
    checkpoint = jnp.load(
        os.path.join(config["save_path"], "ann_3795.npz"), allow_pickle=True
    )

    nn_unet = UNet(config["tf"] / config["n_t"], 64, upsampling="pixel_shuffle")
    params = checkpoint["params"].item()

    def nn_score_(x, t, scoreNet, params):
        return scoreNet.apply(params, x, t)

    nn_score = partial(nn_score_, scoreNet=nn_unet, params=params)

    # Set up mask and measurement
    ground_truth = next(iter(xs))[0]
    mask_spiral = maskSpiral(
        img_shape=(92, 112), num_spiral=2, num_samples=50000, sigma=0.2
    )

    # Set up conditional SDE
    cond_sde = CondSDE(beta=beta, mask=mask_spiral, tf=config["tf"], score=nn_score)
    sde = SDE(beta=beta)

    return sde, cond_sde, mask_spiral, ground_truth, config["tf"], 300, nn_score


config_jacopo = {
    "path_img": "mni_FLAIR.nii.gz",
    "path_mask": "mni_wmh.nii.gz",
    "path_model": "wmh_diff.npz",
}

import torch
import torch.nn.functional as F
import torchio as tio


def _initialize_experiment(key: PRNGKeyArray):
    img = tio.ScalarImage(config_jacopo["path_img"])
    mask = tio.LabelMap(config_jacopo["path_mask"])

    data_masked = torch.concatenate(
        [
            img[tio.DATA][0, ..., 40, None],
            mask[tio.DATA][0, ..., 40, None].type(torch.float32),
        ],
        dim=-1,
    )

    padding = (0, 0, 0, 3, 0, 1)
    ground_truth = jnp.array(F.pad(data_masked, padding, "constant", 0))

    # Initialize parameters
    tf = 2.0
    n_t = 300

    # Define beta schedule and SDE
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)

    # Initialize ScoreNetwork
    score_net = UNet(tf / n_t, 64, upsampling="pixel_shuffle")
    nn_trained = jnp.load(config_jacopo["path_model"], allow_pickle=True)
    params = nn_trained["params"].item()

    # Define neural network score function
    def nn_score(x, t):
        return score_net.apply(params, x, t)

    # Set up mask and measurement
    mask = maskSpiral(img_shape=(92, 112), num_spiral=3, num_samples=50000, sigma=0.2)

    # Set up conditional SDE
    cond_sde = CondSDE(beta=beta, mask=mask, tf=tf, score=nn_score)
    sde = SDE(beta=beta)

    return sde, cond_sde, mask, ground_truth, tf, n_t, nn_score


def optimize_design(
    key_step: PRNGKeyArray,
    implicit_state: ImplicitState,
    past_y: Array,
    mask_history: Array,
    optimizer: GradientTransformation,
    ts: Array,
    dt: float,
    cond_sde: CondSDE,
):
    opt_steps = 1_000
    keys_opt = jax.random.split(key_step, opt_steps)

    @scan_tqdm(opt_steps)
    def step(new_state, tup):
        _, key = tup
        # new_state = impl_step( new_state, key, past_y, mask_history, cond_sde=cond_sde, optx_opt=optimizer, ts=ts, dt=dt)
        new_state = impl_full_scan(
            new_state,
            key,
            past_y,
            mask_history,
            cond_sde=cond_sde,
            optx_opt=optimizer,
            ts=ts,
            dt=dt,
        )

        return new_state, new_state.design

    new_state, hist_design = jax.lax.scan(
        step, implicit_state, (jnp.arange(0, opt_steps), keys_opt)
    )
    return new_state, hist_design


def optimize_design_one_step(
    key_step: PRNGKeyArray,
    implicit_state: ImplicitState,
    past_y: Array,
    mask_history: Array,
    optimizer: GradientTransformation,
    ts: Array,
    dt: float,
    cond_sde: CondSDE,
):
    opt_steps = ts.shape[0] - 1
    keys_opt = jax.random.split(key_step, opt_steps)

    @scan_tqdm(opt_steps)
    def step(new_state, tup):
        _, y, y_next, key = tup
        new_state = impl_one_step(
            new_state,
            key,
            y,
            y_next,
            mask_history,
            cond_sde=cond_sde,
            optx_opt=optimizer,
        )
        thetas, cntrst_thetas, design, *_ = new_state
        return new_state, (thetas, cntrst_thetas, design)

    y = jax.tree.map(lambda x: x[:-1], past_y)
    y_next = jax.tree.map(lambda x: x[1:], past_y)
    new_state, hist = jax.lax.scan(
        step, implicit_state, (jnp.arange(0, opt_steps), y, y_next, keys_opt)
    )
    thetas, cntrst_thetas, design_hist = hist
    state = ImplicitState(thetas, cntrst_thetas, new_state.design, new_state.opt_state)
    return state, design_hist


def init_trajectory(
    key_init,
    sde,
    nn_score,
    n_samples,
    n_samples_cntrst,
    tf,
    ts,
    dts,
    ground_truth_shape,
):
    """
    Unitialize full trajectory for conditional sampling using parallel update from Marion et al 2024
    """
    # Initialize samples
    init_samples = jax.random.normal(
        key_init, (n_samples + n_samples_cntrst, *ground_truth_shape)
    )
    tfs = jnp.zeros((n_samples + n_samples_cntrst,)) + tf

    # Denoise
    state_f = SDEState(position=init_samples, t=tfs)
    revert_sde = partial(sde.reverso, score=nn_score, dts=dts)

    # Make right shape
    keys = jax.random.split(key_init, n_samples + n_samples_cntrst)
    _, state_Ts = jax.vmap(revert_sde)(keys, state_f)
    state_Ts = jax.tree.map(
        lambda arr: einops.rearrange(arr, "n h ... -> h n ..."), state_Ts
    )

    # Init thetas
    thetas = state_Ts.position[:, :n_samples]
    cntrst_thetas = state_Ts.position[:, n_samples:]

    return thetas, cntrst_thetas


def init_start_time(
    key_init: PRNGKeyArray,
    n_samples: int,
    n_samples_cntrst: int,
    ground_truth_shape: Tuple[int, ...],
) -> Tuple[Array, Array]:
    """
    Initialize thetas for just the start time of the conditional sampling
    """
    key_t, key_c = jax.random.split(key_init)
    thetas = jax.random.normal(key_t, (n_samples, *ground_truth_shape))
    cntrst_thetas = jax.random.normal(key_c, (n_samples_cntrst, *ground_truth_shape))
    return thetas, cntrst_thetas


def main(key):
    key_init, key_step = jax.random.split(key)

    sde, cond_sde, mask, ground_truth, tf, n_t, nn_score = initialize_experiment(
        key_init
    )
    dt = tf / (n_t - 1)
    num_meas = 20
    n_samples = 150
    n_samples_cntrst = 151

    # Time initialization (kept outside the function)
    ts = jnp.linspace(0, tf, n_t)

    # init design and measurement hist
    design = jax.random.uniform(key_init, (2,), minval=0.1, maxval=2.0)
    y = cond_sde.mask.measure(design, ground_truth)

    measurement_history = jnp.zeros((num_meas, *y.shape), dtype=jnp.complex64)
    measurement_history = measurement_history.at[0].set(y)

    # init optimizer
    optimizer = optax.chain(optax.adam(learning_rate=1e-2), optax.scale(-1))
    opt_state = optimizer.init(design)

    ts = jnp.linspace(0, tf, n_t)

    # init thetas
    # thetas, cntrst_thetas = init_trajectory(key_init, sde, nn_score, n_samples, n_samples_cntrst, tf, ts, dts, ground_truth.shape)
    thetas, cntrst_thetas = init_start_time(
        key_init, n_samples, n_samples_cntrst, ground_truth.shape
    )
    implicit_state = ImplicitState(thetas, cntrst_thetas, design, opt_state)

    # stock in joint_y all measurements
    joint_y = y
    mask_history = mask.make(design)
    plt.imshow(mask_history, cmap="gray")
    plt.show()
    # design_step = jax.jit(partial(optimize_design, optimizer=optimizer, ts=ts, dt=dt, cond_sde=cond_sde))
    design_step = partial(
        optimize_design_one_step, optimizer=optimizer, ts=ts, dt=dt, cond_sde=cond_sde
    )
    for n_meas in range(num_meas):
        key_noise, key_opt, key_gen = jax.random.split(key_step, 3)
        # make noised path for measurements
        keys_noise = jax.random.split(key_noise, n_t)
        state_0 = SDEState(joint_y, jnp.zeros_like(y))
        past_y = jax.jit(jax.vmap(sde.path, in_axes=(0, None, 0)))(
            keys_noise, state_0, ts
        )
        past_y = SDEState(past_y.position[::-1], past_y.t)
        plotter_line_obs(past_y.position)

        # optimize design
        optimal_state, hist_implicit = design_step(
            key_opt, implicit_state, past_y, mask_history
        )
        opt_hist = hist_implicit

        # make new measurement
        new_measurement = cond_sde.mask.measure(optimal_state.design, ground_truth)
        measurement_history = measurement_history.at[n_meas].set(new_measurement)

        # add measured data to joint_y and update history of mask location
        joint_y = cond_sde.mask.supp_measure(
            optimal_state.design, joint_y, new_measurement
        )
        mask_history = cond_sde.mask.supp_mask(
            optimal_state.design, mask_history, cond_sde.mask.make(optimal_state.design)
        )
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(
            np.abs(joint_y[..., 0]),
            cmap="gray",
            norm=matplotlib.colors.LogNorm(
                vmin=np.min(np.abs(joint_y[..., 0])),
                vmax=np.max(np.abs(joint_y[..., 0])),
            ),
        )
        axs[1].imshow(mask_history, cmap="gray")
        plt.show()

        # logging
        print(f"Design_start: {design} Design_end:{optimal_state.design}")
        fig, axs = plt.subplots(1, 2)
        ax1, ax2 = axs
        ax1.imshow(ground_truth[..., 0], cmap="gray")
        ax2.plot(opt_hist[:, 0], marker="+", label="FOV")
        ax2.plot(opt_hist[:, 1], marker="x", label="K_max")
        ax2.legend()

        plt.tight_layout()
        plt.show()

        print(jnp.max(joint_y))
        for i in range(10):
            plotter_line_measure(optimal_state.thetas[:, i])
            plotter_line_measure(optimal_state.cntrst_thetas[:, i])

        # reinitiazize implicit state
        design = jax.random.uniform(key_init, (2,), minval=1.0, maxval=2.0)
        opt_state = optimizer.init(design)
        key_t, key_c = jax.random.split(key_gen)
        # thetas = generate_cond_sample(joint_y, optimal_state.design, key_t, cond_sde, ground_truth.shape, n_t, n_samples)[1][0]

        # tmp = generate_cond_sampleV2(joint_y, mask_history, key_t, cond_sde, ground_truth.shape, n_t, n_samples)
        # plt.imshow(tmp[0][0].position[-1, ..., 0], cmap="gray")
        # plt.show()
        # cntrst_thetas = generate_cond_sampleV2(joint_y, mask_history, key_c, cond_sde, ground_truth.shape, n_t, n_samples_cntrst)[1][0]
        key_step, _ = jax.random.split(key_step)

        # implicit_state = ImplicitState(
        #     optimal_state.thetas, optimal_state.cntrst_thetas, design, opt_state
        # )
        thetas, cntrst_thetas = init_start_time(
            key_step, n_samples, n_samples_cntrst, ground_truth.shape
        )
        implicit_state = ImplicitState(thetas, cntrst_thetas, design, opt_state)

    return implicit_state
