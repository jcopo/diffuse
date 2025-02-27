from pathlib import Path
import numpy as np
from typing import Tuple, Any
from dataclasses import dataclass
from optax import EmaState, EmptyState, ScaleByAdamState, ScaleByScheduleState
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# Constants
CHECKPOINT_PREFIX = "ann"
LATENT_SUFFIX = "_latent"
BEST_MODEL_PREFIX = "best_model"
MAX_CHECKPOINTS = 5


@dataclass
class CheckpointPaths:
    """Helper class to manage checkpoint file paths."""

    def __init__(self, model_dir: str, latent: bool = False):
        self.base_dir = Path(model_dir)
        self.prefix = f"{CHECKPOINT_PREFIX}{LATENT_SUFFIX if latent else ''}"
        self.best_model = f"{BEST_MODEL_PREFIX}{LATENT_SUFFIX if latent else ''}"

    def checkpoint_path(self, epoch: int) -> Path:
        return self.base_dir / f"{self.prefix}_{epoch}.npz"

    def best_model_path(self, temp: bool = False) -> Path:
        return self.base_dir / f"{self.best_model}{'_' if temp else ''}.npz"


def get_latest_model(config: dict) -> int:
    """
    Find the latest model checkpoint number.
    Returns -1 if no checkpoints found.
    """
    try:
        model_dir = Path(config["model_dir"])
        model_files = [
            int(f.stem.split("_")[-1])
            for f in model_dir.glob(f"{CHECKPOINT_PREFIX}*.npz")
            if f.stem.split("_")[-1].isdigit()
        ]
        return max(model_files) - 1 if model_files else -1
    except Exception:
        return -1


def save_checkpoint_file(path: Path, data: dict) -> None:
    """Helper function to safely save checkpoint files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def checkpoint_best_model(
    config: dict,
    params: Any,
    ema_params: Any,
    best_val_loss: float,
    latent: bool = False,
) -> None:
    """Save the best model checkpoint."""
    paths = CheckpointPaths(config["model_dir"], latent)

    # Save to temporary file first
    save_checkpoint_file(
        paths.best_model_path(temp=True),
        {
            "params": params,
            "ema_params": ema_params,
            "best_val_loss": best_val_loss,
        },
    )

    # Atomic rename
    best_path = paths.best_model_path()
    try:
        best_path.unlink(missing_ok=True)
    finally:
        paths.best_model_path(temp=True).rename(best_path)


def checkpoint_model(
    config: dict,
    params: Any,
    opt_state: tuple,
    ema_state: EmaState,
    ema_params: Any,
    epoch: int,
    delete_old: bool = True,
    latent: bool = False,
) -> None:
    """Save a model checkpoint."""
    paths = CheckpointPaths(config["model_dir"], latent)

    save_checkpoint_file(
        paths.checkpoint_path(epoch),
        {
            "params": params,
            "ema_params": ema_params,
            "ema_state": ema_state,
            "opt_state_1": opt_state[0],
            "opt_state_2": opt_state[1][0],
            "opt_state_3": opt_state[1][1],
        },
    )

    if delete_old:
        # Keep only the latest MAX_CHECKPOINTS files
        checkpoints = sorted(
            paths.base_dir.glob(f"{paths.prefix}_*.npz"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        for checkpoint in checkpoints[:-MAX_CHECKPOINTS]:
            checkpoint.unlink()


def load_checkpoint(
    config: dict, verbose: bool = False, latent: bool = False
) -> Tuple[Any, Any, EmaState, tuple, int]:
    """Load a model checkpoint."""
    begin_epoch = get_latest_model(config)
    if begin_epoch == -1:
        raise ValueError("No checkpoint found")

    paths = CheckpointPaths(config["model_dir"], latent)

    while begin_epoch >= 0:
        try:
            checkpoint = np.load(paths.checkpoint_path(begin_epoch), allow_pickle=True)
            break
        except Exception:
            begin_epoch -= 1
    else:
        raise ValueError("No valid checkpoint found")

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

    if verbose:
        print(f"Loaded checkpoint from epoch {begin_epoch}")
    return params, ema_params, ema_state, opt_state, begin_epoch


def load_best_model_checkpoint(
    config: dict, latent: bool = False
) -> Tuple[Any, Any, float]:
    """Load the best model checkpoint."""
    paths = CheckpointPaths(config["model_dir"], latent)

    try:
        checkpoint = np.load(paths.best_model_path(), allow_pickle=True)
    except FileNotFoundError:
        raise ValueError("No best model checkpoint found")

    return (
        checkpoint["params"].item(),
        checkpoint["ema_params"].item(),
        checkpoint["best_val_loss"],
    )


def get_sharding() -> Tuple[NamedSharding, NamedSharding]:
    """Get sharding configuration for distributed training."""
    num_devices = jax.device_count()
    devices = mesh_utils.create_device_mesh((num_devices,))

    mesh = Mesh(devices, axis_names=("batch",))
    sharding = NamedSharding(mesh, P("batch"))
    replicated_sharding = NamedSharding(mesh, P())

    return sharding, replicated_sharding
