"""Export the three FLUX.2-klein-4B components to fixed-shape ONNX (512px).

Torch runs once, offline; everything downstream is JAX. Shapes: 512px image
-> VAE latent 32x32 @ patch 2x2 -> 1024 img tokens; 128 text tokens @
joint_attention_dim 7680 (Qwen3 layers (9, 18, 27) stacked, matching
Flux2KleinPipeline._get_qwen3_prompt_embeds).

Usage: python export_onnx.py [--component text_encoder|transformer|vae|all]
"""

import argparse

import torch

REPO = "black-forest-labs/FLUX.2-klein-4B"
IMG_TOK, TXT_TOK = 1024, 128


def export_text_encoder(out="qwen3_features.onnx"):
    from transformers import Qwen3ForCausalLM

    enc = Qwen3ForCausalLM.from_pretrained(
        REPO, subfolder="text_encoder", torch_dtype=torch.float16,
        attn_implementation="eager").eval()

    class Features(torch.nn.Module):
        def __init__(self, m, layers=(9, 18, 27)):
            super().__init__()
            self.m = m
            self.layers = layers

        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True, use_cache=False)
            hs = torch.stack([out.hidden_states[k] for k in self.layers], dim=1)
            b, n, t, d = hs.shape
            return hs.permute(0, 2, 1, 3).reshape(b, t, n * d)

    ids = torch.zeros(1, TXT_TOK, dtype=torch.int64)
    mask = torch.ones(1, TXT_TOK, dtype=torch.int64)
    torch.onnx.export(Features(enc), (ids, mask), out,
                      input_names=["input_ids", "attention_mask"],
                      output_names=["embeds"], opset_version=18, dynamo=True)


def export_transformer(out="flux2_klein.onnx"):
    from diffusers import Flux2Transformer2DModel

    model = Flux2Transformer2DModel.from_pretrained(
        REPO, subfolder="transformer", torch_dtype=torch.float16).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, hs, ehs, t, iid, tid):
            return self.m(hs, ehs, t, iid, tid, return_dict=False)[0]

    args = (torch.zeros(1, IMG_TOK, 128, dtype=torch.float16),
            torch.zeros(1, TXT_TOK, 7680, dtype=torch.float16),
            torch.tensor([1.0], dtype=torch.float16),
            torch.zeros(IMG_TOK, 4, dtype=torch.float32),
            torch.zeros(TXT_TOK, 4, dtype=torch.float32))
    torch.onnx.export(
        Wrapper(model), args, out,
        input_names=["hidden_states", "encoder_hidden_states", "timestep",
                     "img_ids", "txt_ids"],
        output_names=["sample"], opset_version=18, dynamo=True)


def export_vae(out="flux2_vae_decode.onnx"):
    """BN denorm + unpatchify + decode: (1, 128, 32, 32) -> (1, 3, 512, 512)."""
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(
        REPO, subfolder="vae", torch_dtype=torch.float32).eval()

    class Decode(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
            self.register_buffer("mean", vae.bn.running_mean.view(1, -1, 1, 1))
            self.register_buffer("std", torch.sqrt(
                vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps))

        def forward(self, z):
            z = z * self.std + self.mean
            b, c, h, w = z.shape
            z = (z.reshape(b, c // 4, 2, 2, h, w)
                  .permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2))
            return self.vae.decode(z, return_dict=False)[0]

    z = torch.zeros(1, 128, 32, 32, dtype=torch.float32)
    torch.onnx.export(Decode(vae), (z,), out, input_names=["latents"],
                      output_names=["image"], opset_version=18, dynamo=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", default="all",
                    choices=["text_encoder", "transformer", "vae", "all"])
    a = ap.parse_args()
    if a.component in ("text_encoder", "all"):
        export_text_encoder()
    if a.component in ("transformer", "all"):
        export_transformer()
    if a.component in ("vae", "all"):
        export_vae()
