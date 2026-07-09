"""Stage B: denoise 1024 img tokens with the klein DiT (diffuse Euler loop).

Attention runs on metallas flash kernels (torch-decomposed SDPA sites fused
via tonnx.passes.fuse_sdpa); weights travel as jit arguments
(min_param_bytes) so the 8GB graph compiles in seconds instead of minutes.
Saves klein_latents.npy for decode.py.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from flax import nnx

from diffuse.denoisers.denoiser import Denoiser
from diffuse.diffusion.sde import Flow
from diffuse.examples.flux_dev.utils import FluxTimer
from diffuse.integrator.deterministic import EulerIntegrator
from diffuse.predictor import Predictor
from metallas.flash import make_flash_attention
from tonnx import ops as tonnx_ops
from tonnx.module import OnnxModule
from tonnx.passes import fuse_sdpa

# The jax-metal plugin miscompiles f16; run everything bf16.
_orig_cast = tonnx_ops.OPS["Cast"]
tonnx_ops.OPS["Cast"] = lambda i, a: tuple(
    o.astype(jnp.bfloat16) if getattr(o, "dtype", None) == jnp.float16 else o
    for o in _orig_cast(i, a))

_flash = make_flash_attention()


def _flash_attention(inputs, attrs):
    q, k, v = inputs[:3]
    b, h, s, d = q.shape
    scale = attrs.get("scale", d ** -0.5)
    # The exported graph mixes f32/bf16 operands (jnp promotes silently, the
    # Pallas kernel is strict): run the whole site in bf16.
    q, k, v = (x.astype(jnp.bfloat16).reshape(b * h, s, d) for x in (q, k, v))
    return (_flash(q, k, v, scale=scale).reshape(b, h, s, d),)


tonnx_ops.OPS["Attention"] = _flash_attention

IMG_TOK, TXT_TOK, LAT = 1024, 128, 32


def mu_for(seq_len, steps):
    """compute_empirical_mu from Flux2KleinPipeline."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if seq_len > 4300:
        return a2 * seq_len + b2
    m200, m10 = a2 * seq_len + b2, a1 * seq_len + b1
    a = (m200 - m10) / 190.0
    return a * steps + (m200 - 200.0 * a)


ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default="flux2_klein.onnx")
ap.add_argument("--embeds", default="klein_embeds.npy")
ap.add_argument("--out", default="klein_latents.npy")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

dev = jax.devices("metal")[0]
embeds = jnp.asarray(np.load(args.embeds), jnp.bfloat16)

t0 = time.perf_counter()
model = onnx.load(args.onnx)
print(f"fused {fuse_sdpa(model.graph)} attention sites", flush=True)
module = OnnxModule(model.graph, rngs=nnx.Rngs(0), frozen=False,
                    min_param_bytes=1 << 20)
del model
for _, p in nnx.iter_graph(module):
    if isinstance(p, nnx.Param) and p.value.dtype == jnp.float16:
        p.value = p.value.astype(jnp.bfloat16)
gd, st = nnx.split(module)
st = jax.device_put(st, dev)
print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)

h, w = np.meshgrid(np.arange(LAT), np.arange(LAT), indexing="ij")
img_ids = jnp.asarray(np.stack(
    [np.zeros(IMG_TOK), h.ravel(), w.ravel(), np.zeros(IMG_TOK)], -1), jnp.float32)
txt_ids = jnp.asarray(np.stack(
    [np.zeros(TXT_TOK)] * 3 + [np.arange(TXT_TOK)], -1), jnp.float32)

timer = FluxTimer(n_steps=args.steps, eps=1e-3, tf=1.0,
                  use_dynamic_shift=True, shift_type="exponential")
timer.set_image_seq_len(IMG_TOK)
timer._mu = mu_for(IMG_TOK, args.steps)

with jax.default_device(dev):
    f = jax.jit(lambda s, *a: nnx.merge(gd, s)(*a))

    def network_fn(latents, t):
        lb = latents[None] if latents.ndim == 2 else latents
        tt = jnp.reshape(t, (-1,)).astype(jnp.bfloat16)
        v = f(st, lb.astype(jnp.bfloat16), embeds, tt, img_ids, txt_ids)
        v = v[0] if isinstance(v, tuple) else v
        return v[0].astype(jnp.float32) if latents.ndim == 2 else v.astype(jnp.float32)

    flow = Flow(tf=1.0)
    predictor = Predictor(model=flow, network=network_fn, prediction_type="velocity")
    integrator = EulerIntegrator(model=flow, timer=timer)
    denoiser = Denoiser(integrator=integrator, model=flow, predictor=predictor,
                        x0_shape=(IMG_TOK, 128))

    key = jax.random.key(args.seed)
    t0 = time.perf_counter()
    state, _ = denoiser.generate(rng_key=key, n_steps=args.steps, n_particles=1)
    lat = state.integrator_state.position
    lat.block_until_ready()
    print(f"generate compile+{args.steps} steps: {time.perf_counter() - t0:.1f}s", flush=True)

np.save(args.out, np.asarray(jnp.reshape(lat, (1, IMG_TOK, 128)).astype(jnp.float32)))
print(f"saved {args.out}", flush=True)
