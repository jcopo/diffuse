# FLUX.1-dev Standalone Example

This is a **standalone** FLUX.1-dev implementation that demonstrates using pre-trained FLUX models with the Diffuse sampling framework - without requiring the `triax` training package as a dependency.

## What's Inside

This example contains everything needed to run FLUX inference:

- **models/** - FLUX model architectures (Transformer, VAE, CLIP, T5)
- **utils/** - Checkpoint loading and FLUX-specific timer
- **run_flux_inference.py** - CLI for text-to-image generation

## Installation

```bash
# From the diffusion package root
pip install -e .

# Additional requirements for FLUX
pip install sentencepiece tokenizers einops orbax-checkpoint
```

## Usage

### Command-Line Interface

```bash
python -m diffuse.examples.flux_dev.run_flux_inference \
    --checkpoint-dir /path/to/flux/checkpoint \
    --prompt "A serene landscape with mountains at sunset" \
    --height 1024 \
    --width 1024 \
    --num-steps 30 \
    --guidance-scale 4.0 \
    --output output.png
```

### Python API

```python
from pathlib import Path
from diffuse.examples.flux_dev.run_flux_inference import FluxModelLoader, run_diffuse_generation

# Load FLUX model
loader = FluxModelLoader(
    checkpoint_dir=Path("/path/to/flux/checkpoint"),
    verbose=True
)

# Prepare conditioned network
conditioned = loader.prepare_conditioned_network(
    prompt="A serene landscape with mountains at sunset",
    negative_prompt=None,
    guidance_scale=4.0,
    height=1024,
    width=1024,
)

# Generate latents
latents = run_diffuse_generation(
    conditioned_network=conditioned,
    height=1024,
    width=1024,
    num_steps=30,
    seed=42,
    batch_size=1,
    sigma_shift=None,  # Use dynamic shift
)

# Decode to images
images = loader.decode_latents(latents)
```

## Architecture

### Modular Design

The implementation follows Diffuse's philosophy of **separation of concerns**:

1. **Model Loading** - `FluxModelLoader` handles checkpoints and text encoders
2. **Text Conditioning** - Encodes prompts into velocity field
3. **Sampling** - Modular Timer + Integrator + Denoiser

### Key Components

**FluxTimer** - Resolution-adaptive time discretization with Möbius shift:
```python
from diffuse.examples.flux_dev.utils import FluxTimer

timer = FluxTimer(
    n_steps=30,
    shift=1.15,  # Static shift
    use_dynamic_shift=False
)

# Or use dynamic resolution-adaptive scheduling
timer = FluxTimer(
    n_steps=30,
    use_dynamic_shift=True  # Adaptive: 0.5 @ 256 tokens → 1.15 @ 4096 tokens
)
timer.set_image_seq_len(seq_len)
```

**FluxTransformer2D** - Dual-stream transformer with:
- Double blocks for joint image-text processing
- Single blocks for image-only refinement
- RoPE (Rotary Position Embeddings) for spatial awareness
- Embedded guidance for classifier-free guidance

## Checkpoint Format

Expected directory structure:
```
checkpoint_dir/
├── transformer/
│   ├── state/           # Orbax checkpoint
│   └── config.json
├── vae/
│   ├── state/
│   └── config.json
├── clip_text/
│   ├── state/
│   └── config.json
├── t5_text/
│   ├── state/
│   └── config.json
├── tokenizer/           # CLIP tokenizer
│   ├── vocab.json
│   ├── merges.txt
│   └── tokenizer_config.json
└── tokenizer_2/         # T5 tokenizer
    ├── spiece.model
    └── tokenizer_config.json
```

## Design Philosophy

This example demonstrates the **"no training tax"** research paradigm:

1. **Load Pre-trained Model** - Use FLUX weights without retraining
2. **Experiment with Sampling** - Swap Timers, Integrators, Denoisers
3. **Focus on Research** - Test ideas in hours, not weeks

### Why Standalone?

By including FLUX models directly in the diffusion package examples, we:
- ✅ Remove dependency on the `triax` training framework
- ✅ Make the diffusion package fully self-contained
- ✅ Provide a complete reference implementation
- ✅ Enable research on sampling algorithms without training complexity

## Performance Tips

### Memory Management

The loader automatically manages GPU memory:

```python
# Text encoders are offloaded after encoding
loader.prepare_conditioned_network(...)  # Auto-releases text encoders

# Explicit control
loader.release_transformer()
loader.release_vae()
loader.release_text_encoders()
```

### Dynamic Shift

For multi-resolution workflows, use dynamic shift:

```python
timer = FluxTimer(use_dynamic_shift=True)
# Automatically adjusts μ based on resolution:
# - 512×512 (16×16 patches = 256 tokens) → μ ≈ 0.5
# - 1024×1024 (32×32 patches = 1024 tokens) → μ ≈ 0.88
# - 2048×2048 (64×64 patches = 4096 tokens) → μ = 1.15
```

## Citation

If you use FLUX in your research, please cite:

```bibtex
@article{flux2024,
  title={FLUX.1: Open-Source Text-to-Image Models},
  author={Black Forest Labs},
  year={2024}
}
```

## License

Model architectures and utilities are provided under Apache 2.0 license.
Pre-trained FLUX weights follow Black Forest Labs' licensing terms.
