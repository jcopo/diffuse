"""Stage C: VAE decode klein_latents.npy -> 512px PNG.

Stride-1 same-padded convs route to the Metal-4 tensor units via
metallas.conv (~87x the plugin's XLA conv lowering on these shapes).
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

from metallas.conv import conv2d_nchw
from tonnx import from_file
from tonnx import ops as tonnx_ops

# The jax-metal plugin miscompiles f16; run everything bf16.
_orig_cast = tonnx_ops.OPS["Cast"]
tonnx_ops.OPS["Cast"] = lambda i, a: tuple(
    o.astype(jnp.bfloat16) if getattr(o, "dtype", None) == jnp.float16 else o
    for o in _orig_cast(i, a))

_orig_conv = tonnx_ops.OPS["Conv"]


def _mpp_conv(inputs, attrs):
    x, w = inputs[0], inputs[1]
    b = inputs[2] if len(inputs) > 2 else None
    spatial = getattr(x, "ndim", 0) - 2
    strides = attrs.get("strides", [1, 1])
    dil = attrs.get("dilations", [1, 1])
    pads = attrs.get("pads", [0] * 4)
    kh, kw = (w.shape[2], w.shape[3]) if spatial == 2 else (0, 0)
    ok = (spatial == 2 and list(strides) == [1, 1] and list(dil) == [1, 1]
          and attrs.get("group", 1) == 1 and x.shape[0] == 1
          and kh % 2 == 1 and kw % 2 == 1
          and list(pads) == [kh // 2, kw // 2, kh // 2, kw // 2]
          and x.shape[2] % 8 == 0 and x.shape[3] % 16 == 0)
    if not ok:
        return _orig_conv(inputs, attrs)
    return (conv2d_nchw(x, w, b),)


tonnx_ops.OPS["Conv"] = _mpp_conv

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default="flux2_vae_decode.onnx")
ap.add_argument("--latents", default="klein_latents.npy")
ap.add_argument("--out", default="klein_512.png")
args = ap.parse_args()

dev = jax.devices("metal")[0]
lat = np.load(args.latents)  # (1, 1024, 128)
z = jnp.asarray(lat.transpose(0, 2, 1).reshape(1, 128, 32, 32), jnp.float32)

t0 = time.perf_counter()
module = from_file(args.onnx, rngs=nnx.Rngs(0))
gd, st = nnx.split(module)
st = jax.device_put(st, dev)
print(f"vae load: {time.perf_counter() - t0:.1f}s", flush=True)

with jax.default_device(dev):
    t0 = time.perf_counter()
    img = jax.jit(lambda s, z: nnx.merge(gd, s)(z))(st, z)
    img = (img[0] if isinstance(img, tuple) else img).block_until_ready()
    print(f"decode compile+run: {time.perf_counter() - t0:.1f}s", flush=True)

arr = np.asarray(img[0], np.float32)
arr = np.clip((arr.transpose(1, 2, 0) + 1) / 2 * 255, 0, 255).astype(np.uint8)
Image.fromarray(arr).save(args.out)
print(f"saved {args.out}", flush=True)
