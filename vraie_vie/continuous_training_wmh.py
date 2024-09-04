from functools import partial

import jax
import jax.numpy as jnp

import optax
from optax import EmaState, EmptyState, ScaleByAdamState, ScaleByScheduleState

from create_dataset import WMH

import os
import sys

sys.path.append(os.path.abspath(".."))

from diffuse.score_matching import score_match_loss
from diffuse.sde import SDE, LinearSchedule
from diffuse.unet import UNet

import numpy as np


from tqdm import tqdm


jax.config.update("jax_enable_x64", False)


if __name__ == "__main__":
    config = {
        "modality": "FLAIR",
        "slice_size_template": 91,
        "flair_template_path": "/lustre/fswork/projects/rech/hlp/uha64uw/aistat24/WMH/MNI-FLAIR-2.0mm.nii.gz",
        "path_dataset": "/lustre/fswork/projects/rech/hlp/uha64uw/aistat24/WMH",
        "save_path": "/lustre/fswork/projects/rech/hlp/uha64uw/aistat24/WMH/models/",
        "n_epochs": 4000,
        "batch_size": 32,
        "num_workers": 0,
        "n_t": 32,
        "tf": 2.0,
        "lr": 2e-4,
        "begin_epoch": 240,
    }

    wmh = WMH(config)
    wmh.setup()
    train_loader = wmh.get_train_dataloader()

    checkpoint = jnp.load(
        os.path.join(config["save_path"], f"ann_{config["begin_epoch"]}.npz"),
        allow_pickle=True,
    )

    params = checkpoint["params"].item()

    ema_state = EmaState(
        count=checkpoint["ema_state"][0], ema=checkpoint["ema_state"][1]
    )

    opt_state = (
        EmptyState(),
        (
            ScaleByAdamState(
                count=checkpoint["opt_state_2"][0],
                mu=checkpoint["opt_state_2"][1],
                nu=checkpoint["opt_state_2"][2],
            ),
            ScaleByScheduleState(checkpoint["opt_state_3"][0]),
        ),
    )

    key = jax.random.PRNGKey(0)

    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
    sde = SDE(beta)

    nn_unet = UNet(config["tf"] / config["n_t"], 64, upsampling="pixel_shuffle")

    def weight_fun(t):
        int_b = sde.beta.integrate(t, 0).squeeze()
        return 1 - jnp.exp(-int_b)

    loss = partial(score_match_loss, lmbda=jax.vmap(weight_fun), network=nn_unet)

    def step(key, params, opt_state, ema_state, data, optimizer, ema_kernel, sde, cfg):
        val_loss, g = jax.value_and_grad(loss)(
            params, key, data, sde, cfg["n_t"], cfg["tf"]
        )
        updates, opt_state = optimizer.update(g, opt_state)
        params = optax.apply_updates(params, updates)
        ema_params, ema_state = ema_kernel.update(params, ema_state)
        return params, opt_state, ema_state, val_loss, ema_params

    until_steps = int(0.95 * config["n_epochs"]) * len(train_loader)
    schedule = optax.cosine_decay_schedule(
        init_value=config["lr"], decay_steps=until_steps, alpha=1e-2
    )

    optimizer = optax.adam(learning_rate=schedule)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optimizer)
    ema_kernel = optax.ema(0.99)

    batch_update = jax.jit(
        partial(step, optimizer=optimizer, ema_kernel=ema_kernel, sde=sde, cfg=config)
    )

    for epoch in range(config["begin_epoch"], config["n_epochs"]):
        list_loss = []
        iterator = tqdm(train_loader, desc="Training", file=sys.stdout)

        for batch in iterator:
            key, subkey = jax.random.split(key)
            params, opt_state, ema_state, val_loss, ema_params = batch_update(
                subkey, params, opt_state, ema_state, batch
            )

            iterator.set_postfix({"loss": val_loss})
            list_loss.append(val_loss)

        print(
            f"Epoch {epoch}, loss {val_loss}, mean_loss {sum(list_loss) / len(list_loss)}"
        )

        if epoch % 5 == 0:
            np.savez(
                os.path.join(config["save_path"], f"ann_{epoch}.npz"),
                params=params,
                ema_state=ema_state,
                opt_state_1=opt_state[0],
                opt_state_2=opt_state[1][0],
                opt_state_3=opt_state[1][1],
            )

    np.savez(
        os.path.join(config["save_path"], f"ann_end.npz"),
        params=params,
        ema_state=ema_state,
        opt_state_1=opt_state[0],
        opt_state_2=opt_state[1][0],
        opt_state_3=opt_state[1][1],
    )
