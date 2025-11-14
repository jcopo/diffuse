"""Orbax checkpoint helpers for FLUX models."""

from __future__ import annotations

import functools
import gc
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from ..models import FluxTransformer2D, FluxVAE
from ..models.clip_text import CLIPTextConfig, CLIPTextEncoder
from ..models.t5_encoder import T5Encoder, T5EncoderConfig


def _dtype_to_str(value: Any) -> str:
    return jnp.dtype(value).name


def _dtype_from_str(value: str | None) -> jnp.dtype:
    if value is None:
        return jnp.float32
    return jnp.dtype(value)


def _next_param_dtype(module: Any, attr: str) -> str:
    try:
        param = getattr(module, attr)
        return _dtype_to_str(param.value.dtype)
    except AttributeError:
        return "float32"


def _leaf_dtype(tree: Any) -> str:
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, nnx.Param):
            return _dtype_to_str(leaf.value.dtype)
        if hasattr(leaf, "dtype"):
            return _dtype_to_str(leaf.dtype)
    return "float32"


def _latest_checkpoint_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint directory {root} does not exist.")
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    numeric = sorted((p for p in subdirs if p.name.isdigit()), key=lambda p: int(p.name))
    if numeric:
        return numeric[-1]
    return root


@functools.cache
def _default_array_sharding() -> jax.sharding.NamedSharding:
    cpu_devices = jax.devices("cpu")
    devices = cpu_devices or jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available for checkpoint restoration.")
    mesh = jax.sharding.Mesh(devices, ("device",))
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def _array_sharding_for_leaf(leaf: Any) -> jax.sharding.Sharding | None:
    candidate = getattr(leaf, "value", leaf)
    if isinstance(candidate, jax.ShapeDtypeStruct):
        return _default_array_sharding()
    if isinstance(candidate, jax.Array):
        existing = getattr(candidate, "sharding", None)
        if isinstance(existing, jax.sharding.Sharding):
            return existing
        return _default_array_sharding()
    return None


def _restore_state_from_path(
    state_path: Path,
    state_template: nnx.State,
    *,
    sharding: jax.sharding.Sharding | None = None,
) -> nnx.State:
    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = _build_restore_args_tree(state_template, sharding=sharding)
    return checkpointer.restore(
        str(state_path),
        args=ocp.args.PyTreeRestore(item=state_template, restore_args=restore_args),
    )


def _build_restore_args_tree(
    state_template: nnx.State,
    *,
    sharding: jax.sharding.Sharding | None = None,
) -> Any:
    def make_restore_args(leaf: Any) -> ocp.RestoreArgs:
        target_sharding = sharding or _array_sharding_for_leaf(leaf)
        if target_sharding is not None:
            return ocp.ArrayRestoreArgs(restore_type=jax.Array, sharding=target_sharding)
        return ocp.RestoreArgs()

    return jax.tree_util.tree_map(make_restore_args, state_template)


def _tree_to_device(tree: Any, device: jax.Device) -> Any:
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, device) if isinstance(x, jax.Array) else x,
        tree,
    )


@dataclass
class FluxCheckpointBundle:
    """Serializable payload for FLUX checkpoints."""

    transformer_state: nnx.State | None = None
    transformer_config: dict[str, Any] | None = None
    vae_state: nnx.State | None = None
    vae_config: dict[str, Any] | None = None
    clip_state: nnx.State | None = None
    clip_config: dict[str, Any] | None = None
    t5_state: nnx.State | None = None
    t5_config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


def bundle_from_components(
    transformer: FluxTransformer2D | None = None,
    clip_text_encoder: CLIPTextEncoder | None = None,
    t5_text_encoder: T5Encoder | None = None,
    *,
    vae: FluxVAE | None = None,
    metadata: dict[str, Any] | None = None,
    transformer_config: dict[str, Any] | None = None,
    vae_config: dict[str, Any] | None = None,
) -> FluxCheckpointBundle:
    """Create a checkpoint bundle from loaded FLUX modules."""

    transformer_state = None
    if transformer is not None:
        transformer_config = transformer_config or {
            "in_channels": transformer.in_channels,
            "hidden_dim": transformer.hidden_dim,
            "num_heads": transformer.num_heads,
            "num_double_layers": transformer.num_double_layers,
            "num_single_layers": transformer.num_single_layers,
            "joint_attention_dim": transformer.joint_attention_dim,
            "pooled_projection_dim": transformer.pooled_projection_dim,
            "mlp_ratio": getattr(transformer, "mlp_ratio", 4.0),
            "guidance_embeds": transformer.guidance_embeds,
            "param_dtype": _next_param_dtype(transformer.img_in, "kernel"),
            "dtype": _next_param_dtype(transformer.img_in, "kernel"),
        }
        transformer_graph, transformer_state = nnx.split(transformer)
        del transformer_graph

    vae_state: nnx.State | None = None
    if vae is not None:
        vae_config = vae_config or {
            "in_channels": 3,
            "latent_channels": getattr(vae, "latent_channels", 16),
            "param_dtype": _next_param_dtype(vae.encoder.conv_in, "kernel"),
        }
        vae_graph, vae_state = nnx.split(vae)
        del vae_graph

    clip_state: nnx.State | None = None
    clip_config: dict[str, Any] | None = None
    if clip_text_encoder is not None:
        clip_config = clip_text_encoder.config.to_dict()
        clip_graph, clip_state = nnx.split(clip_text_encoder)
        del clip_graph
        clip_config.setdefault("dtype", _leaf_dtype(clip_state))
        clip_config.setdefault("param_dtype", clip_config["dtype"])

    t5_state: nnx.State | None = None
    t5_config: dict[str, Any] | None = None
    if t5_text_encoder is not None:
        t5_graph, t5_state = nnx.split(t5_text_encoder)
        del t5_graph
        t5_config = t5_text_encoder.config.to_dict()
        t5_config.setdefault("dtype", _leaf_dtype(t5_state))
        t5_config.setdefault("param_dtype", t5_config["dtype"])

    return FluxCheckpointBundle(
        transformer_state=transformer_state,
        transformer_config=transformer_config,
        vae_state=vae_state,
        vae_config=vae_config,
        clip_state=clip_state,
        clip_config=clip_config,
        t5_state=t5_state,
        t5_config=t5_config,
        metadata=metadata,
    )


def save_flux_checkpoint(
    bundle: FluxCheckpointBundle,
    checkpoint_dir: str | Path,
    *,
    step: int | None = None,
) -> Path:
    """Save FLUX components as individual Orbax checkpoints."""

    base = Path(checkpoint_dir).expanduser().resolve()
    target = base / f"{step:08d}" if step is not None else base
    target.mkdir(parents=True, exist_ok=True)

    def save_component(name: str, state: Any, config: dict[str, Any] | None):
        if state is None and config is None:
            return
        comp_dir = target / name
        comp_dir.mkdir(exist_ok=True)
        if state is not None:
            state_path = comp_dir / "state"
            if state_path.exists():
                shutil.rmtree(state_path)
            ocp.PyTreeCheckpointer().save(str(state_path), item=state)
        if config is not None:
            with (comp_dir / "config.json").open("w") as f:
                json.dump(config, f, indent=2)

    save_component("transformer", bundle.transformer_state, bundle.transformer_config)
    save_component("clip_text", bundle.clip_state, bundle.clip_config)
    save_component("t5_text", bundle.t5_state, bundle.t5_config)
    if bundle.vae_state is not None:
        save_component("vae", bundle.vae_state, bundle.vae_config)

    if bundle.metadata:
        with (target / "metadata.json").open("w") as f:
            json.dump(bundle.metadata, f, indent=2)

    return target


def load_flux_checkpoint(
    checkpoint_dir: str | Path,
    *,
    step: int | None = None,
    transformer_rng_seed: int = 0,
    vae_rng_seed: int = 0,
) -> dict[str, Any]:
    """Return lightweight metadata for FLUX checkpoint components."""

    base = Path(checkpoint_dir).expanduser().resolve()
    target = base if step is None else base / f"{step:08d}"
    if not target.exists():
        if step is None:
            target = _latest_checkpoint_dir(base)
        else:
            raise FileNotFoundError(f"Checkpoint step {step} not found under {base}")

    def component_info(name: str) -> tuple[Path | None, dict[str, Any] | None]:
        comp_dir = target / name
        if not comp_dir.exists():
            return None, None
        state_path = comp_dir / "state"
        config_path = comp_dir / "config.json"
        config = None
        if config_path.exists():
            with config_path.open("r") as f:
                config = json.load(f)
        return (state_path if state_path.exists() else None), config

    transformer_state_path, transformer_config = component_info("transformer")
    clip_state_path, clip_config = component_info("clip_text")
    t5_state_path, t5_config = component_info("t5_text")
    vae_state_path, vae_config = component_info("vae")

    if transformer_state_path is None or transformer_config is None:
        raise FileNotFoundError("Transformer checkpoint incomplete: missing state or config")
    metadata_path = target / "metadata.json"
    metadata = None
    if metadata_path.exists():
        with metadata_path.open("r") as f:
            metadata = json.load(f)

    components: dict[str, dict[str, Any]] = {
        "transformer": {
            "config": transformer_config,
            "state_path": transformer_state_path,
            "rng_seed": transformer_rng_seed,
        }
    }

    if vae_state_path is not None and vae_config is not None:
        components["vae"] = {
            "config": vae_config,
            "state_path": vae_state_path,
            "rng_seed": vae_rng_seed,
        }

    if clip_state_path is not None and clip_config is not None:
        components["clip_text"] = {
            "config": clip_config,
            "state_path": clip_state_path,
            "rng_seed": transformer_rng_seed,
        }

    if t5_state_path is not None and t5_config is not None:
        components["t5_text"] = {
            "config": t5_config,
            "state_path": t5_state_path,
            "rng_seed": transformer_rng_seed,
        }

    return {
        "components": components,
        "transformer_config": transformer_config,
        "metadata": metadata or {},
    }


def _component_param_dtype(config: dict[str, Any], default: str = "float32") -> jnp.dtype:
    return _dtype_from_str(config.get("param_dtype", config.get("dtype", default)))


def _state_nbytes(state: nnx.State) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(state):
        candidate = getattr(leaf, "value", leaf)
        if isinstance(candidate, jax.Array):
            total += candidate.size * candidate.dtype.itemsize
        elif hasattr(candidate, "nbytes"):
            total += int(candidate.nbytes)
        elif hasattr(candidate, "size") and hasattr(candidate, "dtype"):
            total += int(candidate.size) * candidate.dtype.itemsize
    return total


def instantiate_flux_component(
    *,
    name: str,
    config: dict[str, Any],
    state_path: Path,
    rng_seed: int,
    target_device: jax.Device,
) -> tuple[nnx.Module, int]:
    """Instantiate and restore a single FLUX component onto the requested device.

    Uses nnx.eval_shape to create abstract model without allocating memory,
    then restores checkpoint weights directly to target device.
    """

    if not state_path.exists():
        raise FileNotFoundError(f"Missing checkpoint state at {state_path}")

    # Use nnx.eval_shape to create abstract model without allocating memory
    if name == "transformer":
        abstract_model = nnx.eval_shape(
            lambda: FluxTransformer2D(
                in_channels=config.get("in_channels", 64),
                hidden_dim=config.get("hidden_dim", 3072),
                num_heads=config.get("num_heads", 24),
                num_double_layers=config.get("num_double_layers", 19),
                num_single_layers=config.get("num_single_layers", 38),
                joint_attention_dim=config.get("joint_attention_dim", 4096),
                pooled_projection_dim=config.get("pooled_projection_dim", 768),
                mlp_ratio=config.get("mlp_ratio", 4.0),
                guidance_embeds=config.get("guidance_embeds", True),
                axes_dims_rope=tuple(config.get("axes_dims_rope", (16, 56, 56))),
                param_dtype=_component_param_dtype(config),
                dtype=_dtype_from_str(config.get("dtype", config.get("param_dtype"))),
                rngs=nnx.Rngs(rng_seed),
            )
        )
    elif name == "vae":
        abstract_model = nnx.eval_shape(
            lambda: FluxVAE(
                in_channels=config.get("in_channels", 3),
                latent_channels=config.get("latent_channels", 16),
                param_dtype=_component_param_dtype(config),
                rngs=nnx.Rngs(rng_seed),
            )
        )
    elif name == "clip_text":
        clip_cfg = CLIPTextConfig.from_dict(config)
        abstract_model = nnx.eval_shape(
            lambda: CLIPTextEncoder(
                clip_cfg,
                param_dtype=_component_param_dtype(config),
                dtype=_dtype_from_str(config.get("dtype", config.get("torch_dtype"))),
                rngs=nnx.Rngs(rng_seed),
            )
        )
    elif name == "t5_text":
        t5_cfg = T5EncoderConfig.from_dict(config)
        abstract_model = nnx.eval_shape(
            lambda: T5Encoder(
                t5_cfg,
                param_dtype=_component_param_dtype(config),
                dtype=_dtype_from_str(config.get("dtype", config.get("param_dtype"))),
                rngs=nnx.Rngs(rng_seed),
            )
        )
    else:
        raise ValueError(f"Unknown FLUX component '{name}'")

    # Split abstract model to get graph definition and abstract state (no allocation)
    graph, state_template = nnx.split(abstract_model)

    # Restore checkpoint directly to target device
    sharding = jax.sharding.SingleDeviceSharding(target_device)
    restored_state = _restore_state_from_path(
        state_path,
        state_template,
        sharding=sharding,
    )
    bytes_loaded = _state_nbytes(restored_state)

    # Merge graph with restored weights
    merged_module = nnx.merge(graph, restored_state)
    del restored_state
    gc.collect()

    return merged_module, bytes_loaded
