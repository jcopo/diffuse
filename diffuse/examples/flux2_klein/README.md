# FLUX.2-klein-4B on Apple Silicon (jax-metal + metallas)

512px text-to-image with the full FLUX.2-klein pipeline running on the GPU of
an Apple-silicon Mac: Qwen3 text encoder, DiT and VAE are exported to ONNX
once (torch), imported as Flax NNX modules with [tonnx], and sampled with
diffuse's `Denoiser`/`EulerIntegrator`. Attention runs on [metallas] flash
kernels and the VAE convs on the Metal-4 tensor units.

Warm timings on an M5 Pro (4 steps): **denoise 3.1 s (0.77 s/step) + VAE
decode 1.3 s**. Cold DiT compile ~34 s.

## Requirements

- macOS 26+ on M5 (Metal 4 / MetalPerformancePrimitives), Xcode toolchain
- the patched zml-xla metal PJRT plugin (see metallas README)
- `tonnx`, `metallas`; torch + diffusers + transformers for the one-time export

## Run

```bash
# one-time: export the three components to ONNX (~12GB total, fp16)
python export_onnx.py

export PJRT_NAMES_AND_LIBRARY_PATHS="metal:<path to libpjrt_c_api_gpu_plugin.dylib>"
export JAX_PLATFORMS="metal,cpu"
export METAL_TOOLCHAIN="$(dirname "$(xcrun -f air-as)")"
export JAX_CAPTURED_CONSTANTS_WARN_BYTES=-1

python encode.py --prompt "A serene mountain lake at sunset"
python denoise.py --steps 4
python decode.py            # -> klein_512.png
```

The stages are separate processes (each loads a 4B-param component); they
hand off through `klein_embeds.npy` / `klein_latents.npy`.

## Notes

- `denoise.py` rewrites the 25 torch-decomposed SDPA sites to `Attention`
  nodes (`tonnx.passes.fuse_sdpa`) and runs them on metallas flash; bf16
  latents differ from the exact reference by ~0.8% relative RMS.
- Weights travel as jit arguments (`frozen=False, min_param_bytes=1MB`)
  instead of XLA constants — compile drops from minutes to seconds. Keep the
  small tensors as constants: hundreds of parameter buffers crash the Metal
  shader compiler.
- Everything is cast to bf16: the metal plugin's f16 path miscompiles.

[tonnx]: https://github.com/jacopoiollo/tonnx
[metallas]: https://github.com/jacopoiollo/metallas
