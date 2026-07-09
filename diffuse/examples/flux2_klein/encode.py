"""Stage A: prompt -> Qwen3 features (1, 128, 7680), saved as klein_embeds.npy."""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from transformers import AutoTokenizer

from tonnx import from_file
from tonnx import ops as tonnx_ops

# The jax-metal plugin miscompiles f16; run everything bf16.
_orig_cast = tonnx_ops.OPS["Cast"]
tonnx_ops.OPS["Cast"] = lambda i, a: tuple(
    o.astype(jnp.bfloat16) if getattr(o, "dtype", None) == jnp.float16 else o
    for o in _orig_cast(i, a))

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", default="A serene mountain lake at sunset")
ap.add_argument("--onnx", default="qwen3_features.onnx")
ap.add_argument("--out", default="klein_embeds.npy")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained("black-forest-labs/FLUX.2-klein-4B",
                                    subfolder="tokenizer")
text = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                               tokenize=False, add_generation_prompt=True,
                               enable_thinking=False)
enc = tok(text, padding="max_length", truncation=True, max_length=128)
ids = jnp.asarray([enc["input_ids"]], jnp.int32)
mask = jnp.asarray([enc["attention_mask"]], jnp.int32)

t0 = time.perf_counter()
module = from_file(args.onnx, rngs=nnx.Rngs(0))
for _, p in nnx.iter_graph(module):
    if isinstance(p, nnx.Param) and p.value.dtype == jnp.float16:
        p.value = p.value.astype(jnp.bfloat16)
gd, st = nnx.split(module)
dev = jax.devices("metal")[0]
st = jax.device_put(st, dev)
print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)

t0 = time.perf_counter()
with jax.default_device(dev):
    emb = jax.jit(lambda s, i, m: nnx.merge(gd, s)(i, m))(st, ids, mask)
    emb = (emb[0] if isinstance(emb, tuple) else emb).block_until_ready()
print(f"encode compile+run: {time.perf_counter() - t0:.1f}s", flush=True)
np.save(args.out, np.asarray(emb.astype(jnp.float32)))
print(f"saved {args.out}", flush=True)
