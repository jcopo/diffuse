import os
from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from scipy.spatial.distance import directed_hausdorff

from diffuse.conditional import CondSDE
from diffuse.inference import generate_cond_sampleV2
from diffuse.sde import SDE, LinearSchedule
from diffuse.unet import UNet

from vraie_vie.create_dataset import WMH
from vraie_vie.utils import maskAno

from tqdm import tqdm


def dice_coefficient(y_true, y_pred):
    intersection = jnp.sum(y_true * y_pred)
    return (2.0 * intersection) / (jnp.sum(y_true) + jnp.sum(y_pred) + 1e-6)


def average_volume_difference(y_true, y_pred):
    return 100 * (jnp.sum(y_pred) - jnp.sum(y_true)) / jnp.sum(y_true + 1e-6)


def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()
    return tp / (tp + fn + 1e-6)


def modified_hausdorff_distance(y_true, y_pred):
    if jnp.sum(y_true) == 0 or jnp.sum(y_pred) == 0:
        return float("inf")
    return directed_hausdorff(y_true.nonzero()[0], y_pred.nonzero()[0])[0]


def compute_metrics(metrics, real_ano, predict_ano):
    metrics["Dice"].append(dice_coefficient(real_ano, predict_ano))
    metrics["Average Volume Difference (%)"].append(
        average_volume_difference(real_ano, predict_ano)
    )
    metrics["Sensitivity (%)"].append(sensitivity(real_ano, predict_ano))
    metrics["Modified Hausdorff Distance"].append(
        modified_hausdorff_distance(real_ano, predict_ano)
    )
    metrics["F1 Score"].append(
        f1_score(real_ano.flatten(), predict_ano.flatten(), average="binary")
    )
    return metrics


config = {
    "modality": "FLAIR",
    "slice_size_template": 49,
    "begin_slice": 26,
    "flair_template_path": "/lustre/fswork/projects/rech/hlp/uha64uw/projet_p/WMH/MNI-FLAIR-2.0mm.nii.gz",
    # "path_dataset": "/Users/geoffroyoudoumanessah/Documents/these/projects/datasets/WMH",
    "path_dataset": "/lustre/fswork/projects/rech/hlp/uha64uw/projet_p/WMH",
    "save_path": "/lustre/fswork/projects/rech/hlp/uha64uw/projet_p/WMH/models/",
    "n_epochs": 4000,
    "batch_size": 32,
    "num_workers": 0,
    "n_t": 32,
    "tf": 2.0,
    "lr": 2e-4,
}

if __name__ == "__main__":
    wmh = WMH(config)
    wmh.setup()
    test_loader = wmh.get_test_dataloader()

    key = jax.random.PRNGKey(0)
    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)

    checkpoint = jnp.load(
        os.path.join(config["save_path"], "ann_3795.npz"), allow_pickle=True
    )

    nn_unet = UNet(config["tf"] / config["n_t"], 64, upsampling="pixel_shuffle")
    params = checkpoint["params"].item()

    def nn_score_(x, t, scoreNet, params):
        return scoreNet.apply(params, x, t)

    nn_score = partial(nn_score_, scoreNet=nn_unet, params=params)

    sde = SDE(beta=beta)

    size = (92, 112)

    mask_ano = maskAno(img_shape=size)
    cond_sde = CondSDE(beta=beta, mask=mask_ano, tf=2.0, score=nn_score)

    xi = jnp.array(0.0)
    mask = mask_ano.make(xi)

    size = (92, 112, 2)
    n_particules = 300
    generate = partial(
        generate_cond_sampleV2,
        mask_history=mask,
        cond_sde=cond_sde,
        key=key,
        x_shape=size,
        n_ts=1000,
        n_particules=n_particules,
    )

    metrics_results = {
        "Dice": [],
        "Average Volume Difference (%)": [],
        "Sensitivity (%)": [],
        "Modified Hausdorff Distance": [],
        "F1 Score": [],
    }
    for x in tqdm(test_loader):
        y = mask_ano.measure(xi, x)
        res = generate(y, mask)

        unif_idx = jax.random.randint(key, (1,), 0, n_particules)
        res = res[0][0].position[unif_idx]

        real_ano = np.array(x[..., 1])
        predict_ano = np.array(res[..., 1])

        metrics_results = compute_metrics(metrics_results, real_ano, predict_ano)

        df = pd.DataFrame(metrics_results)
        df.to_csv("metrics_results.csv", index=False)
