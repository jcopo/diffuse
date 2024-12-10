import jax
import jax.numpy as jnp
from matplotlib.colors import LogNorm


from diffuse.samplopt.conditional import CondSDE
from diffuse.samplopt.inference import generate_cond_sample
from diffuse.diffusion.sde import SDE, SDEState, LinearSchedule
from diffuse.neural_network.unet import UNet

from examples.mri.wmh.create_dataset import WMH
from examples.mri.wmh.design import main
from examples.mri.utils import maskAno, maskSpiral, slice_inverse_fourier

import matplotlib.pyplot as plt
from functools import partial
import os


USER = "upd68za"
config = {
    "modality": "FLAIR",
    "slice_size_template": 49,
    "begin_slice": 26,
    "flair_template_path": f"/lustre/fswork/projects/rech/ijy/{USER}/diffuse/data/MNI-FLAIR-2.0mm.nii.gz",
    # "path_dataset": "/Users/geoffroyoudoumanessah/Documents/these/projects/datasets/WMH",
    "path_dataset": f"/lustre/fswork/projects/rech/ijy/{USER}/diffuse/data/WMH",
    "save_path": f"/lustre/fswork/projects/rech/ijy/{USER}/diffuse/data",
    "n_epochs": 4000,
    "batch_size": 32,
    "num_workers": 0,
    "n_t": 32,
    "tf": 2.0,
    "lr": 2e-4,
}

# Retrieve trained Parameters
checkpoint = jnp.load(
    os.path.join(config["save_path"], "wmh_diff.npz"), allow_pickle=True
)
params = checkpoint["params"].item()

# Get the Datasets
wmh = WMH(config)
wmh.setup()
train_loader = wmh.get_train_dataloader().dataset

# Get the ScoreNet
nn_unet = UNet(config["tf"] / config["n_t"], 64, upsampling="pixel_shuffle")


def nn_score_(x, t, scoreNet, params):
    return scoreNet.apply(params, x, t)


nn_score = partial(nn_score_, scoreNet=nn_unet, params=params)
# train_loader = wmh.get_test_dataloader().dataset


key = jax.random.PRNGKey(0)
beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)

# checkpoint = jnp.load(
#     os.path.join(config["save_path"], "ann_3795.npz"), allow_pickle=True
# )


# nn_unet = UNet(config["tf"] / config["n_t"], 64, upsampling="pixel_shuffle")
# params = checkpoint["params"].item()


def nn_score_(x, t, scoreNet, params):
    return scoreNet.apply(params, x, t)


nn_score = partial(nn_score_, scoreNet=nn_unet, params=params)
sde = SDE(beta=beta)
def main():
    idx = 20
    x = jnp.array(train_loader[idx])
    plt.imshow(x[..., 0], cmap="gray")
    plt.show()
    plt.imshow(x[..., 1], cmap="gray")
    plt.show()

    size = (92, 112)

    mask_spiral = maskSpiral(img_shape=size, num_spiral=3, num_samples=50000, sigma=0.2)
    cond_sde = CondSDE(beta=beta, mask=mask_spiral, tf=2.0, score=nn_score)

    xi = jnp.array([.2, 2.0, jnp.pi/2])  # FOV, k_max
    y = mask_spiral.measure(xi, x)
    x_sub = slice_inverse_fourier(y[..., 0])

    mask = mask_spiral.make(xi)

    end_state, hist = generate_cond_sample(y, mask, key, cond_sde, x.shape, 200, 100)
    return end_state, hist


if __name__ == "__main__":
    main()
