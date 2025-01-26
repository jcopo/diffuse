import argparse
import sys
import os
import yaml
from functools import partial

import jax
import jax.numpy as jnp

import optax

from envyaml import EnvYAML
from tqdm import tqdm

from diffuse.diffusion.score_matching import score_match_loss, weight_zoo
from diffuse.diffusion.sde import SDE, LinearSchedule
from diffuse.neural_network import model_zoo
from examples.mri.brats.create_dataset import (
    get_dataloader as get_brats_dataloader,
)
from examples.mri.fastMRI.create_dataset import (
    get_dataloader as get_fastmri_dataloader,
)
from examples.mri.wmh.create_dataset import (
    get_dataloader as get_wmh_dataloader,
)

from examples.mri.utils import (
    checkpoint_model,
    get_first_item,
    get_latest_model,
    get_sharding,
    load_checkpoint,
)

jax.config.update("jax_enable_x64", False)

dataloader_zoo = {
    "WMH": lambda cfg: get_wmh_dataloader(cfg, train=True),
    "BRATS": lambda cfg: get_brats_dataloader(cfg, train=True),
    "fastMRI": lambda cfg: get_fastmri_dataloader(cfg, train=True),
}


def train(config, train_val_loaders, parallel=False):
    key = jax.random.PRNGKey(0)

    # Unpack train and validation loaders
    train_loader, val_loader = train_val_loaders

    # Configuring the SDE
    beta = LinearSchedule(
        b_min=config["sde"]["beta_min"],
        b_max=config["sde"]["beta_max"],
        t0=config["sde"]["t0"],
        T=config["sde"]["tf"],
    )
    sde = SDE(beta, tf=config["sde"]["tf"])

    # Configuring the UNet
    if config["score_model"] == "UNet":
        score_net = model_zoo[config["score_model"]](
            dt=config["unet"]["dt_embedding"],
            dim=config["unet"]["embedding_dim"],
            upsampling=config["unet"]["upsampling"],
            dim_mults=config["unet"]["dim_mults"],
        )
    elif config["score_model"] == "UNett":
        score_net = model_zoo[config["score_model"]](
            dim=config["unet"]["embedding_dim"]
        )
    else:
        raise ValueError(f"Score model {config['score_model']} not found")

    # Initializing the UNet
    key, subkey = jax.random.split(key)
    init_params = score_net.init(
        subkey,
        jnp.ones((1, *get_first_item(train_loader).shape[1:])),
        jnp.ones((1,)),
    )

    # Configuring the loss function
    if config["training"]["loss_weight"] == "None":
        lmbda = None

    elif config["training"]["loss_weight"] == "weight_fun":
        lmbda = jax.vmap(partial(weight_zoo["weight_fun"], sde=sde))

    else:
        raise ValueError(f"Loss weight {config['training']['loss_weight']} not found")

    loss = partial(score_match_loss, network=score_net, lmbda=lmbda)

    # Defining the training step
    def step(key, params, opt_state, ema_state, data, optimizer, ema_kernel, sde, cfg):
        val_loss, g = jax.value_and_grad(loss)(
            params, key, data, sde, cfg["training"]["nt_samples"], cfg["sde"]["tf"]
        )
        updates, opt_state = optimizer.update(g, opt_state)
        params = optax.apply_updates(params, updates)
        ema_params, ema_state = ema_kernel.update(params, ema_state)
        return params, opt_state, ema_state, val_loss, ema_params

    def validate_step(key, params, data, sde, cfg):
        val_loss = loss(
            params, key, data, sde, cfg["training"]["nt_samples"], cfg["sde"]["tf"]
        )
        return val_loss

    # Configuring the optimizer
    until_steps = int(0.95 * config["training"]["n_epochs"]) * len(train_loader)
    schedule = optax.cosine_decay_schedule(
        init_value=config["training"]["lr"], decay_steps=until_steps, alpha=1e-2
    )

    optimizer = optax.adam(learning_rate=schedule)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optimizer)
    ema_kernel = optax.ema(0.99)

    # JIT the training step
    batch_update = jax.jit(
        partial(step, optimizer=optimizer, ema_kernel=ema_kernel, sde=sde, cfg=config)
    )
    batch_validate = jax.jit(partial(validate_step, sde=sde, cfg=config))

    # Loading the checkpoint if exists
    begin_epoch = get_latest_model(config)
    if begin_epoch != -1:
        params, _, ema_state, opt_state, begin_epoch, best_val_loss, old_best_epoch = (
            load_checkpoint(config)
        )
        iterator_epoch = range(begin_epoch, config["training"]["n_epochs"])

    # Initializing the parameters and optimizer states if continue_training is False
    else:
        # Initialize parameters and optimizer states
        params, opt_state, ema_state = (
            init_params,
            optimizer.init(init_params),
            ema_kernel.init(init_params),
        )
        iterator_epoch = range(config["training"]["n_epochs"])
        best_val_loss = float("inf")
        old_best_epoch = -1

    # Configuring the sharding for parallel training (can be removed)
    if parallel:
        sharding, replicated_sharding = get_sharding()

        params = jax.device_put(params, replicated_sharding)
        opt_state = jax.device_put(opt_state, replicated_sharding)
        ema_state = jax.device_put(ema_state, replicated_sharding)

    # Training loop
    for epoch in iterator_epoch:
        list_loss, list_val_loss = [], []
        train_iterator = tqdm(train_loader, desc="Training", file=sys.stdout)
        val_iterator = tqdm(val_loader, desc="Validation", file=sys.stdout)

        for batch in train_iterator:
            key, subkey = jax.random.split(key)
            batch = jax.device_put(batch, sharding) if parallel else batch
            params, opt_state, ema_state, val_loss, ema_params = batch_update(
                subkey, params, opt_state, ema_state, batch
            )

            train_iterator.set_postfix({"loss": val_loss})
            list_loss.append(val_loss)

        train_loss = sum(list_loss) / len(list_loss)
        print(f"Epoch {epoch}, training loss: {train_loss}")

        for batch in val_iterator:
            key, subkey = jax.random.split(key)
            batch = jax.device_put(batch, sharding) if parallel else batch
            val_loss = batch_validate(subkey, params, batch)
            val_iterator.set_postfix({"loss": val_loss})
            list_val_loss.append(val_loss)
        current_val_loss = sum(list_val_loss) / len(list_val_loss)
        print(f"Validation loss: {current_val_loss}")

        # Save checkpoint only if validation loss improves
        if current_val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss} to {current_val_loss}"
            )
            best_val_loss = current_val_loss
            checkpoint_model(
                config,
                params,
                opt_state,
                ema_state,
                ema_params,
                epoch,
                best_val_loss,
                old_best_epoch,
            )
            old_best_epoch = epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file"
    )

    parser.add_argument(
        "--parallel", type=bool, default=False, help="Parallel training"
    )
    args = parser.parse_args()

    config = EnvYAML(args.config)

    if not os.path.exists(config["model_dir"]):
        os.makedirs(config["model_dir"], exist_ok=True)
        os.system(
            f"cp {args.config} {config['model_dir']}/config_{config['dataset']}.yaml"
        )

    train_val_loaders = dataloader_zoo[config["dataset"]](config)
    train(config, train_val_loaders, parallel=args.parallel)
