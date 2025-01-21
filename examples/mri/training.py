import argparse
import os
import sys
from functools import partial

import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import optax
from optax import EmaState, EmptyState, ScaleByAdamState, ScaleByScheduleState

import numpy as np
import yaml
from tqdm import tqdm

from diffuse.diffusion.score_matching import score_match_loss
from diffuse.diffusion.sde import SDE, LinearSchedule
from diffuse.neural_network.unet import UNet
from examples.mri.brats.create_dataset import (
    get_dataloader as get_brats_dataloader,
)
from examples.mri.fastMRI.create_dataset import (
    get_dataloader as get_fastmri_dataloader,
)
from examples.mri.wmh.create_dataset import (
    get_dataloader as get_wmh_dataloader,
)

from examples.mri.utils import get_first_item

jax.config.update("jax_enable_x64", False)

dataloader_zoo = {
    "wmh": lambda cfg: get_wmh_dataloader(cfg, train=True),
    "brats": lambda cfg: get_brats_dataloader(cfg, train=True),
    "fastMRI": lambda cfg: get_fastmri_dataloader(cfg, train=True),
}

def train(config, train_loader, parallel=False, continue_training=False):
    key = jax.random.PRNGKey(0)

    beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
    sde = SDE(beta, tf=2.0)

    nn_unet = UNet(config["tf"] / config["n_t"], 64, upsampling="pixel_shuffle")

    if parallel:
        num_devices = jax.device_count()

        dummy_data = jnp.ones((num_devices, *get_first_item(train_loader).shape[1:]))
        dummy_labels = jnp.ones((num_devices,))

        keys = jax.random.split(key, num_devices)

        init_to_device = lambda key, dummy_data, dummy_labels: nn_unet.init(
            key, dummy_data, dummy_labels
        )

        init_params = jax.pmap(init_to_device)(keys, dummy_data, dummy_labels)
        init_params = jax.tree.map(lambda x: x[0], init_params)

    else:
        key, subkey = jax.random.split(key)
        init_params = nn_unet.init(
            subkey,
            jnp.ones(get_first_item(train_loader).shape),
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
        iterator_epoch = range(config["begin_epoch"], config["n_epochs"])

    else:
        params = init_params
        opt_state = optimizer.init(params)
        ema_state = ema_kernel.init(params)
        iterator_epoch = range(config["n_epochs"])

    if parallel:
        num_devices = jax.device_count()
        devices = mesh_utils.create_device_mesh((num_devices,))

        mesh = Mesh(devices, axis_names=("batch",))
        sharding = NamedSharding(
            mesh,
            P(
                "batch",
            ),
        )
        replicated_sharding = NamedSharding(mesh, P())

        params = jax.device_put(params, replicated_sharding)
        opt_state = jax.device_put(opt_state, replicated_sharding)
        ema_state = jax.device_put(ema_state, replicated_sharding)

    for epoch in iterator_epoch:
        list_loss = []
        iterator = tqdm(train_loader, desc="Training", file=sys.stdout)

        for batch in iterator:
            key, subkey = jax.random.split(key)

            batch = jax.device_put(batch, sharding)
            params, opt_state, ema_state, val_loss, ema_params = batch_update(
                subkey, params, opt_state, ema_state, batch
            )

            iterator.set_postfix({"loss": val_loss})
            list_loss.append(val_loss)

        print(
            f"Epoch {epoch}, loss {val_loss}, mean_loss {sum(list_loss) / len(list_loss)}"
        )

        # if epoch % 5 == 0:
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
        "--parallel", type=bool, default=False, help="Parallel training"
    )
    parser.add_argument(
        "--continue_training", type=bool, default=False, help="Continue training"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_loader = dataloader_zoo[config["dataset"]](config)

    if args.continue_training:
        begin_epoch = (
            max(
                [
                    int(e.split(".")[0][4:])
                    for e in os.listdir(config["save_path"])
                    if e.endswith(".npz") and e[0] == "a"
                ]
            )
            - 1
        )

        config["begin_epoch"] = begin_epoch
        train(config, train_loader, parallel=args.parallel, continue_training=True)
    else:
        train(config, train_loader, parallel=args.parallel)
