import os
import numpy as np
from optax import EmaState, EmptyState, ScaleByAdamState, ScaleByScheduleState
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import yaml


def get_first_item(dataloader):
    return next(iter(dataloader))


def get_latest_model(config):
    try:
        model_files = [
            int(e.split(".")[0][4:])  # Get number after "ann_"
            for e in os.listdir(config["model_dir"])
            if e.startswith("ann_") and e.endswith(".npz") and e[4:-4].isdigit()
        ]
        if not model_files:
            return -1
        return max(model_files) - 1

    except:
        return -1

def checkpoint_model(config, params, opt_state, ema_state, ema_params, epoch, delete_old=True):
    np.savez(
        os.path.join(config["model_dir"], f"ann_{epoch}.npz"),
        params=params,
        ema_params=ema_params,
        ema_state=ema_state,
        opt_state_1=opt_state[0],
        opt_state_2=opt_state[1][0],
        opt_state_3=opt_state[1][1],
    )
    if delete_old:
        # Get all model files and sort them by epoch number
        model_files = [
            f for f in os.listdir(config["model_dir"]) 
            if f.startswith("ann_") and f.endswith(".npz") and f[4:-4].isdigit()
        ]
        model_files.sort(key=lambda x: int(x[4:-4]))
        
        # Remove all but the last 5 models
        if len(model_files) > 5:
            for f in model_files[:-5]:
                os.remove(os.path.join(config["model_dir"], f))


def load_checkpoint(config):
    begin_epoch = get_latest_model(config)
    best_model = None
    while best_model is None:
        try:
            checkpoint = np.load(
                os.path.join(config["model_dir"], f"ann_{begin_epoch}.npz"),
                allow_pickle=True,
            )
            best_model = begin_epoch
        except:
            begin_epoch -= 1

    params = checkpoint["params"].item()

    params = checkpoint["params"].item()
    ema_params = checkpoint["ema_params"].item()

    ema_state = EmaState(
        count=checkpoint["ema_state"][0], ema=checkpoint["ema_state"][1]
    )

    opt_state = (
        EmptyState(),
        (
            ScaleByAdamState(
                checkpoint["opt_state_2"][0],
                checkpoint["opt_state_2"][1],
                checkpoint["opt_state_2"][2],
            ),
            ScaleByScheduleState(checkpoint["opt_state_3"][0]),
        ),
    )
    return params, ema_params, ema_state, opt_state, begin_epoch


def get_sharding():
    num_devices = jax.device_count()
    devices = mesh_utils.create_device_mesh((num_devices,))

    mesh = Mesh(devices, axis_names=("batch",))
    sharding = NamedSharding(mesh, P("batch"))
    replicated_sharding = NamedSharding(mesh, P())

    return sharding, replicated_sharding
