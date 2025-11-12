#!/usr/bin/env python3
"""FLUX checkpoint loader wired into Diffuse sampling via argparse CLI."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece as spm
from einops import rearrange
from flax import nnx
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing

from diffuse.denoisers.denoiser import Denoiser
from diffuse.diffusion.sde import Flow
from diffuse.integrator.deterministic import DDIMIntegrator
from diffuse.predictor import Predictor

from .utils.flux_checkpoint import (
    instantiate_flux_component,
    load_flux_checkpoint,
)
from .utils.flux_timer import FluxTimer


# -----------------------------------------------------------------------------
# Helper structures
# -----------------------------------------------------------------------------
def _patchify_latents(latents: jax.Array) -> jax.Array:
    """Convert (B, H, W, 16) VAE latents into transformer patches (B, H/2, W/2, 64)."""
    batch, h, w, c = latents.shape
    if c != 16:
        raise ValueError(f"Expected 16 latent channels, got {c}")
    if h % 2 or w % 2:
        raise ValueError("Latent height/width must be divisible by 2 for patchify.")
    return rearrange(latents, "b (h p1) (w p2) c -> b h w (c p1 p2)", p1=2, p2=2)


def _unpatchify_latents(latents: jax.Array) -> jax.Array:
    """Inverse of ``_patchify_latents``."""
    batch, h, w, c = latents.shape
    if c % 4:
        raise ValueError("Transformer latent channels must be divisible by 4.")
    return rearrange(latents, "b h w (c p1 p2) -> b (h p1) (w p2) c", p1=2, p2=2)


@dataclass
class EncodedPrompt:
    text_embeddings: jax.Array
    pooled_embeddings: jax.Array
    text_ids: jax.Array


@dataclass
class FluxConditionedNetwork:
    """Container for the conditioned FLUX transformer network."""

    network_fn: Callable[[jax.Array, jax.Array], jax.Array]
    in_channels: int
    dtype: jnp.dtype
    text_ids: jax.Array


def _format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = 0
    while value >= 1024.0 and unit < len(units) - 1:
        value /= 1024.0
        unit += 1
    return f"{value:.2f}{units[unit]}"


class FluxModelLoader:
    """Load FLUX components and prepare text-conditioned transformer networks."""

    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        device: jax.Device | None = None,
        verbose: bool = False,
    ):
        self.cpu_device = jax.devices("cpu")[0]
        self.verbose = verbose
        if device is not None:
            self.device = device
        else:
            gpu_devices = jax.devices("gpu")
            self.device = gpu_devices[0] if gpu_devices else self.cpu_device

        self._log(f"[flux-loader] Host CPU device: {self.cpu_device.platform}:{self.cpu_device.id}")
        self._log(f"[flux-loader] Active compute device: {self.device.platform}:{self.device.id}")

        bundle = load_flux_checkpoint(checkpoint_dir)
        components = bundle.get("components", {})
        transformer_spec = components.get("transformer")
        vae_spec = components.get("vae")
        clip_spec = components.get("clip_text")
        t5_spec = components.get("t5_text")

        if transformer_spec is None:
            raise RuntimeError("Checkpoint is missing the transformer component.")
        if vae_spec is None:
            raise RuntimeError("Checkpoint is missing the VAE component.")
        if clip_spec is None or t5_spec is None:
            raise RuntimeError("Checkpoint must include CLIP and T5 text encoders.")

        self.checkpoint_dir = checkpoint_dir.expanduser().resolve()

        self._transformer_spec = transformer_spec
        self._vae_spec = vae_spec
        self._clip_spec = clip_spec
        self._t5_spec = t5_spec

        self._component_loaders: dict[str, Callable[[jax.Device], tuple[nnx.Module, int]]] = {}
        self._component_modules: dict[str, nnx.Module | None] = {}
        self._component_state_bytes: dict[str, int] = {}
        for component, spec in (
            ("transformer", transformer_spec),
            ("vae", vae_spec),
            ("clip_text", clip_spec),
            ("t5_text", t5_spec),
        ):
            self._register_component(component, spec)

        vae_config = vae_spec.get("config", {}) if vae_spec is not None else {}
        asset_vae_config = {}
        asset_config_path = self.checkpoint_dir / "vae" / "config.json"
        if asset_config_path.exists():
            with asset_config_path.open("r") as f:
                asset_vae_config = json.load(f)

        scaling_factor = vae_config.get("scaling_factor", asset_vae_config.get("scaling_factor"))
        if scaling_factor is None:
            raise RuntimeError("VAE config missing required `scaling_factor` value.")
        self.vae_scaling_factor = float(scaling_factor)

        shift_factor = vae_config.get("shift_factor", asset_vae_config.get("shift_factor"))
        if shift_factor is None:
            raise RuntimeError("VAE config missing required `shift_factor` value.")
        self.vae_shift_factor = float(shift_factor)

        config = transformer_spec.get("config", {})
        dtype_name = config.get("dtype") or config.get("param_dtype") or "float32"
        self.transformer_dtype = jnp.dtype(dtype_name)
        self.transformer_in_channels = int(config.get("in_channels", 64))

        self._image_id_cache: dict[tuple[int, int], jax.Array] = {}

        # Drop bulk checkpoint bundle to free CPU memory immediately.
        del bundle, components

        self.clip_tokenizer_dir = self.checkpoint_dir / "tokenizer"
        self.t5_tokenizer_dir = self.checkpoint_dir / "tokenizer_2"
        if not self.clip_tokenizer_dir.exists():
            raise FileNotFoundError(f"CLIP tokenizer directory missing: {self.clip_tokenizer_dir}")
        if not self.t5_tokenizer_dir.exists():
            raise FileNotFoundError(f"T5 tokenizer directory missing: {self.t5_tokenizer_dir}")

        with (self.clip_tokenizer_dir / "tokenizer_config.json").open("r") as f:
            clip_cfg = json.load(f)

        self.clip_max_length = clip_cfg.get("model_max_length", 77)
        self.clip_tokenizer = self._create_clip_tokenizer(clip_cfg)

        self.t5_sp = spm.SentencePieceProcessor()
        self.t5_sp.load(str(self.t5_tokenizer_dir / "spiece.model"))
        with (self.t5_tokenizer_dir / "tokenizer_config.json").open("r") as f:
            t5_cfg = json.load(f)

        self.t5_max_length = t5_cfg.get("model_max_length", 256)
        self.t5_pad_id = t5_cfg.get("pad_token_id", 0)
        self.t5_eos_id = t5_cfg.get("eos_token_id", 1)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _make_loader(self, component: str, spec: dict[str, Any]) -> Callable[[jax.Device], tuple[nnx.Module, int]]:
        state_path = spec.get("state_path")
        config = spec.get("config") or {}
        rng_seed = spec.get("rng_seed", 0)
        if state_path is None:
            raise RuntimeError(f"Missing state path for component '{component}'.")

        def loader(target_device: jax.Device) -> tuple[nnx.Module, int]:
            module, bytes_loaded = instantiate_flux_component(
                name=component,
                config=config,
                state_path=state_path,
                rng_seed=rng_seed,
                target_device=target_device,
            )
            return module, bytes_loaded

        return loader

    def _register_component(self, component: str, spec: dict[str, Any]) -> None:
        self._component_loaders[component] = self._make_loader(component, spec)
        self._component_modules[component] = None
        self._component_state_bytes[component] = 0

    def _get_component(self, component: str) -> nnx.Module:
        loader = self._component_loaders.get(component)
        if loader is None:
            raise RuntimeError("Requested module is not available in checkpoint bundle.")
        module = self._component_modules.get(component)
        if module is None:
            self._log(f"[flux-loader] Restoring {component} on {self.device.platform}:{self.device.id}")
            module, bytes_loaded = loader(self.device)
            self._component_modules[component] = module
            if bytes_loaded:
                self._component_state_bytes[component] = bytes_loaded
                self._log(f"[flux-loader] Loaded {component} parameters (~{_format_bytes(bytes_loaded)})")
        return self._component_modules[component]

    def _release_component(self, component: str) -> None:
        if self._component_modules.get(component) is not None:
            self._log(f"[flux-loader] Releasing {component} from {self.device.platform}:{self.device.id}")
            self._component_modules[component] = None
            gc.collect()

    def _create_clip_tokenizer(self, clip_cfg: dict[str, Any]) -> Tokenizer:
        vocab_path = self.clip_tokenizer_dir / "vocab.json"
        merges_path = self.clip_tokenizer_dir / "merges.txt"
        if not vocab_path.exists() or not merges_path.exists():
            raise FileNotFoundError(f"CLIP tokenizer requires vocab.json and merges.txt in {self.clip_tokenizer_dir}")

        unk_token = clip_cfg.get("unk_token", "<|endoftext|>")
        model = BPE.from_file(
            str(vocab_path),
            str(merges_path),
            unk_token=unk_token,
        )
        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        bos_token = clip_cfg.get("bos_token", clip_cfg.get("cls_token", "<|startoftext|>"))
        bos_id = clip_cfg.get("bos_token_id", clip_cfg.get("cls_token_id", 0))
        eos_token = clip_cfg.get("eos_token", "<|endoftext|>")
        eos_id = clip_cfg.get("eos_token_id", clip_cfg.get("sep_token_id", 2))
        pad_token = clip_cfg.get("pad_token", eos_token)
        pad_id = clip_cfg.get("pad_token_id", eos_id)

        tokenizer.post_processor = TemplateProcessing(
            single=f"{bos_token} $A {eos_token}",
            pair=f"{bos_token} $A {eos_token} {bos_token} $B {eos_token}",
            special_tokens=[(bos_token, bos_id), (eos_token, eos_id)],
        )
        tokenizer.enable_truncation(self.clip_max_length)
        tokenizer.enable_padding(
            length=self.clip_max_length,
            pad_id=pad_id,
            pad_token=pad_token,
        )
        return tokenizer

    def release_text_encoders(self) -> None:
        """Free GPU memory by offloading text encoders back to CPU."""
        self._log("[flux-loader] Releasing text encoders")
        self._release_component("clip_text")
        self._release_component("t5_text")

    def release_transformer(self) -> None:
        """Explicitly free transformer weights from the Accelerator."""
        self._log("[flux-loader] Releasing transformer")
        self._release_component("transformer")

    def release_vae(self) -> None:
        """Explicitly free VAE weights from the Accelerator."""
        self._log("[flux-loader] Releasing VAE")
        self._release_component("vae")

    # ------------------------------------------------------------------ #
    # Prompt encoding helpers                                           #
    # ------------------------------------------------------------------ #
    def _encode_prompts(self, prompts: Iterable[str]) -> EncodedPrompt:
        prompt_list = list(prompts)
        self._log(f"[flux-loader] Encoding {len(prompt_list)} prompt(s)")
        if not prompt_list:
            raise ValueError("Received empty prompt list for encoding.")
        self._log("[flux-loader] Tokenizing prompts for CLIP/T5")
        clip_encodings = [self.clip_tokenizer.encode(p) for p in prompt_list]
        clip_ids = jnp.array([enc.ids for enc in clip_encodings], dtype=jnp.int32)
        clip_mask = jnp.array([enc.attention_mask for enc in clip_encodings], dtype=jnp.int32)
        clip_encoder = self._get_component("clip_text")
        clip_outputs = clip_encoder(clip_ids, attention_mask=clip_mask)
        self._log(
            f"[flux-loader] CLIP pooled embedding shape {tuple(clip_outputs['pooler_output'].shape)}, dtype={clip_outputs['pooler_output'].dtype}"
        )

        t5_id_rows = []
        t5_mask_rows = []
        for prompt in prompt_list:
            ids = list(self.t5_sp.encode(prompt, out_type=int))
            if not ids or ids[-1] != self.t5_eos_id:
                ids.append(self.t5_eos_id)
            ids = ids[: self.t5_max_length]
            mask = [1] * len(ids)
            if len(ids) < self.t5_max_length:
                pad_len = self.t5_max_length - len(ids)
                ids.extend([self.t5_pad_id] * pad_len)
                mask.extend([0] * pad_len)
            t5_id_rows.append(ids)
            t5_mask_rows.append(mask)

        t5_ids = jnp.array(t5_id_rows, dtype=jnp.int32)
        t5_mask = jnp.array(t5_mask_rows, dtype=jnp.int32)
        t5_encoder = self._get_component("t5_text")
        t5_outputs = t5_encoder(t5_ids, attention_mask=t5_mask)
        self._log(
            f"[flux-loader] T5 hidden state shape {tuple(t5_outputs['last_hidden_state'].shape)}, dtype={t5_outputs['last_hidden_state'].dtype}"
        )

        seq_len = t5_outputs["last_hidden_state"].shape[1]
        text_ids = jnp.zeros((seq_len, 3), dtype=self.transformer_dtype)

        return EncodedPrompt(
            text_embeddings=t5_outputs["last_hidden_state"].astype(self.transformer_dtype),
            pooled_embeddings=clip_outputs["pooler_output"].astype(self.transformer_dtype),
            text_ids=text_ids,
        )

    def _encode_conditioning(
        self,
        prompt: str,
        negative_prompt: str | None,
    ) -> tuple[EncodedPrompt, EncodedPrompt]:
        negative_prompt = negative_prompt if negative_prompt is not None else ""
        positive = self._encode_prompts([prompt])
        negative = self._encode_prompts([negative_prompt])
        return positive, negative

    @staticmethod
    def _broadcast_embeddings(emb: jax.Array, batch: int) -> jax.Array:
        if emb.shape[0] == batch:
            return emb
        return jnp.repeat(emb, batch, axis=0)

    def _get_image_ids(self, height: int, width: int) -> jax.Array:
        key = (height, width)
        cached = self._image_id_cache.get(key)
        if cached is None:
            ids = jnp.zeros((height, width, 3), dtype=self.transformer_dtype)
            ids = ids.at[..., 1].set(jnp.arange(height, dtype=self.transformer_dtype)[:, None])
            ids = ids.at[..., 2].set(jnp.arange(width, dtype=self.transformer_dtype)[None, :])
            cached = ids.reshape(height * width, 3)
            self._image_id_cache[key] = cached
        return cached

    # ------------------------------------------------------------------ #
    # Conditioned network construction                                  #
    # ------------------------------------------------------------------ #
    def _make_network_fn(
        self,
        positive: EncodedPrompt,
        guidance_scale: float,
        height: int,
        width: int,
    ) -> Callable[[jax.Array, jax.Array], jax.Array]:
        """Wrap the FLUX transformer as a Diffuse-compatible network.

        Args:
            positive: Encoded text prompts
            guidance_scale: Classifier-free guidance scale
            height: Latent height (in transformer patch space)
            width: Latent width (in transformer patch space)
        """
        transformer = self._get_component("transformer")
        text_ids = positive.text_ids

        # Pre-compute img_ids ONCE, outside any JAX transformation
        # This avoids tracer leaks from caching inside jax.lax.scan
        img_ids = self._get_image_ids(height, width)

        def network_fn(latents: jax.Array, timesteps: jax.Array) -> jax.Array:
            # vmap passes per-sample latents without batch dim: (H, W, C)
            latents_batched = latents[None, ...] if latents.ndim == 3 else latents

            t = timesteps[None] if timesteps.ndim == 0 else timesteps

            pos_txt = positive.text_embeddings[None, ...] if positive.text_embeddings.ndim == 2 else positive.text_embeddings
            pos_pool = positive.pooled_embeddings[None, ...] if positive.pooled_embeddings.ndim == 1 else positive.pooled_embeddings

            batch = latents_batched.shape[0]
            pos_txt = self._broadcast_embeddings(pos_txt, batch)
            pos_pool = self._broadcast_embeddings(pos_pool, batch)

            t = t.reshape(-1)
            if t.shape[0] == 1 and batch > 1:
                t = jnp.repeat(t, batch, axis=0)
            elif t.shape[0] != batch:
                raise ValueError(f"Mismatched timestep batch {t.shape[0]} vs latent batch {batch}")

            guidance_values = None
            if guidance_scale is not None:
                guidance_values = jnp.full((batch,), guidance_scale, dtype=self.transformer_dtype)

            # Use pre-computed img_ids from closure (no cache lookup inside scan)
            outputs = transformer(
                img_latents=latents_batched,
                txt_embeddings=pos_txt,
                timesteps=t,
                pooled_txt_emb=pos_pool,
                guidance_scale=guidance_values,
                img_ids=img_ids,
                txt_ids=text_ids,
            ).output
            return outputs[0]

        return network_fn

    def prepare_conditioned_network(
        self,
        prompt: str,
        negative_prompt: str | None,
        guidance_scale: float,
        height: int,
        width: int,
    ) -> FluxConditionedNetwork:
        """Prepare a text-conditioned FLUX network for generation.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (ignored in embedded-guidance mode)
            guidance_scale: Classifier-free guidance scale
            height: Output image height in pixels (will be converted to latent space)
            width: Output image width in pixels (will be converted to latent space)
        """
        self._log("[flux-loader] Preparing conditioned network")
        positive = self._encode_prompts([prompt])
        if negative_prompt and negative_prompt.strip():
            self._log(
                "[flux-loader] NOTE: negative prompts are ignored in embedded-guidance mode; set --guidance-scale to 0 for faithful reconstruction."
            )

        # Offload text encoders from accelerator memory BEFORE loading transformer
        # to avoid OOM - embeddings are already materialized in positive/negative
        self.release_text_encoders()

        # Convert pixel dimensions to transformer patch space
        # VAE downsamples by 8x, then patchify by 2x -> 16x total
        _, transformer_hw = _latent_shapes(height, width)

        network_fn = self._make_network_fn(
            positive,
            guidance_scale,
            transformer_hw[0],
            transformer_hw[1],
        )

        return FluxConditionedNetwork(
            network_fn=network_fn,
            in_channels=self.transformer_in_channels,
            dtype=self.transformer_dtype,
            text_ids=positive.text_ids,
        )

    def encode_images(self, images: jax.Array) -> jax.Array:
        """Encode RGB images to FLUX transformer latent space.

        Args:
            images: RGB images of shape (batch, height, width, 3) in range [0, 1]

        Returns:
            Transformer latents of shape (batch, h/16, w/16, 64) (patchified)
        """
        self._log("[flux-loader] Encoding images through VAE")
        # Scale images from [0, 1] to [-1, 1]
        images_scaled = images * 2.0 - 1.0

        vae = self._get_component("vae")
        vae_latents = vae.encode(images_scaled)

        # Apply VAE shift and scaling
        if self.vae_shift_factor != 0.0:
            vae_latents = vae_latents - self.vae_shift_factor
        if self.vae_scaling_factor != 0.0:
            vae_latents = vae_latents * self.vae_scaling_factor

        # Convert to transformer space (patchify)
        transformer_latents = _patchify_latents(vae_latents)
        self.release_vae()
        return transformer_latents

    def decode_latents(self, latents: jax.Array) -> np.ndarray:
        self._log("[flux-loader] Decoding latents through VAE")
        vae_latents = _unpatchify_latents(latents)
        if self.vae_scaling_factor != 0.0:
            vae_latents = vae_latents / self.vae_scaling_factor
        if self.vae_shift_factor != 0.0:
            vae_latents = vae_latents + self.vae_shift_factor
        vae = self._get_component("vae")
        decoded = vae.decode(vae_latents)
        images = jnp.clip((decoded + 1.0) / 2.0, 0.0, 1.0)
        self.release_vae()
        return np.array(jax.device_get(images))


def _latent_shapes(height: int, width: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if height % 8 or width % 8:
        raise ValueError("Height and width must be divisible by 8 for the FLUX VAE.")
    vae_hw = (height // 8, width // 8)
    if vae_hw[0] % 2 or vae_hw[1] % 2:
        raise ValueError("Latent spatial size must be even (required for patchify).")
    transformer_hw = (vae_hw[0] // 2, vae_hw[1] // 2)
    return vae_hw, transformer_hw


def run_diffuse_generation(
    *,
    conditioned_network: FluxConditionedNetwork,
    height: int,
    width: int,
    num_steps: int,
    seed: int,
    batch_size: int,
    sigma_shift: float | None,
) -> jax.Array:
    """Create Diffuse components and sample latents."""

    _, transformer_hw = _latent_shapes(height, width)
    image_seq_len = transformer_hw[0] * transformer_hw[1]

    flow = Flow(tf=1.0)
    timer = FluxTimer(
        n_steps=num_steps,
        eps=1e-3,
        tf=1.0,
        shift=sigma_shift if sigma_shift is not None else 1.15,
        use_dynamic_shift=sigma_shift is None,
    )
    if sigma_shift is None:
        timer.set_image_seq_len(image_seq_len)
        print(f"[flux-loader] Dynamic sigma shift (mu={timer._mu:.3f}) for sequence length {image_seq_len}")

    predictor = Predictor(model=flow, network=conditioned_network.network_fn, prediction_type="velocity")
    integrator = DDIMIntegrator(model=flow, timer=timer)
    denoiser = Denoiser(
        integrator=integrator,
        model=flow,
        predictor=predictor,
        x0_shape=(transformer_hw[0], transformer_hw[1], conditioned_network.in_channels),
    )

    key = jax.random.PRNGKey(seed)
    state, _ = denoiser.generate(
        rng_key=key,
        n_steps=num_steps,
        n_particles=batch_size,
        keep_history=False,
    )
    latents = state.integrator_state.position.astype(conditioned_network.dtype)
    return latents


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def _save_images(array: np.ndarray, path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB images, got array with shape {array.shape}")

    images = [array[0]] if array.shape[0] == 1 else list(array)
    for idx, img in enumerate(images):
        target = path if len(images) == 1 else path.with_stem(f"{path.stem}_{idx:02d}")
        Image.fromarray((img * 255).astype(np.uint8)).save(str(target))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FLUX inference using Diffuse and an Orbax checkpoint bundle.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing transformer/vae/clip_text/t5_text and tokenizer/tokenizer_2 directories.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Positive prompt text.")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Optional negative prompt.")
    parser.add_argument("--height", type=int, default=1024, help="Output height (must be divisible by 16).")
    parser.add_argument("--width", type=int, default=1024, help="Output width (must be divisible by 16).")
    parser.add_argument("--num-steps", type=int, default=30, help="Number of DDIM integration steps.")
    parser.add_argument(
        "--sigma-shift",
        type=float,
        default=None,
        help=(
            "Flux Möbius shift parameter. Leave unset to use the resolution-dependent schedule (≈0.5 @ 256 tokens → 1.15 @ 4096 tokens)."
        ),
    )
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument(
        "--verbose-loading",
        action="store_true",
        help="Print detailed device placement and memory usage information while loading models.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("flux_output.png"),
        help="Output path. Multiple samples receive numbered suffixes.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    loader = FluxModelLoader(
        checkpoint_dir=args.checkpoint_dir,
        verbose=args.verbose_loading,
    )
    conditioned = loader.prepare_conditioned_network(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
    )
    latents = run_diffuse_generation(
        conditioned_network=conditioned,
        height=args.height,
        width=args.width,
        num_steps=args.num_steps,
        seed=args.seed,
        batch_size=args.batch_size,
        sigma_shift=args.sigma_shift,
    )
    loader.release_transformer()
    images = loader.decode_latents(latents)
    _save_images(images, args.output)
    count = images.shape[0]
    suffix = "image" if count == 1 else "images"
    print(f"✅ Saved {count} {suffix} to {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
