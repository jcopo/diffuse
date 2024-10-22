from functools import partial
import argparse
import jax
import jax.numpy as jnp

import optax
from optax import EmaState, EmptyState, ScaleByAdamState, ScaleByScheduleState

import os
import sys

sys.path.append(os.path.abspath("/lustre/fswork/projects/rech/hlp/uha64uw/projet_p/diffuse"))

from diffuse.score_matching import score_match_loss
from diffuse.sde import SDE, LinearSchedule
from diffuse.unet import UNet

import numpy as np


from tqdm import tqdm
import yaml

from vraie_vie.wmh.create_dataset import (
    get_train_dataloader as get_wmh_train_dataloader,
)
from vraie_vie.brats.create_dataset import (
    get_train_dataloader as get_brats_train_dataloader,
)


jax.config.update("jax_enable_x64", False)


def train(config, train_loader, continue_training=False):
    key = jax.random.PRNGKey(0)

    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
    sde = SDE(beta)

    nn_unet = UNet(config["tf"] / config["n_t"], 64, upsampling="pixel_shuffle")

    key, subkey = jax.random.split(key)
    init_params = nn_unet.init(
        subkey,
        jnp.ones((config["batch_size"], *train_loader.dataset[0].shape[1:])),
        jnp.ones((config["batch_size"],)),
    )

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

    if continue_training:
        checkpoint = jnp.load(
            os.path.join(config["save_path"], f"ann_{config['begin_epoch']}.npz"),
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

    else:
        params = init_params
        opt_state = optimizer.init(params)
        ema_state = ema_kernel.init(params)

    for epoch in range(config["n_epochs"]):
        list_loss = []
        iterator = tqdm(train_loader, desc="Training", file=sys.stdout)

        for batch in iterator:
            key, subkey = jax.random.split(key)
            batch = batch["vol"].squeeze(1)
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
                ema_params=ema_params,
                ema_state=ema_state,
                opt_state_1=opt_state[0],
                opt_state_2=opt_state[1][0],
                opt_state_3=opt_state[1][1],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )

    parser.add_argument(
        "--continue_training", type=bool, default=False, help="Continue training"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config["dataset"] == "wmh":
        train_loader = get_wmh_train_dataloader(config)

    elif config["dataset"] == "brats":
        train_loader = get_brats_train_dataloader(config)

    if args.continue_training:
        begin_epoch = max(
            [
                int(e.split(".")[0][4:])
                for e in os.listdir(config["save_path"])
                if e.endswith(".npz") and e[0] == "a"
            ]
        )

        config["begin_epoch"] = begin_epoch
        train(config, train_loader, continue_training=True)
    else:
        train(config, train_loader)
