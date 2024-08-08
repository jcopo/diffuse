"""
This script is adapted from https://raw.githubusercontent.com/yiyixuxu/denoising-diffusion-flax/main/denoising_diffusion_flax/unet.py
under the Apache-2.0 license.

- You may find a copy of the license in https://github.com/yiyixuxu/denoising-diffusion-flax/tree/main?tab=Apache-2.0-1-ov-file#readme.
- The original author is https://github.com/yiyixuxu.
- Modifications:
    - Deleted the batch in time embedding.
    - Added pixel_shuffle upsampling.
    - Changed resize from 'nearest' to 'linear'.
    - Removed assert statements.
    - Removed "name".
    - Used in-house sinusoidal_embedding.
    - Some variable names refactorisation.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange
from typing import Any, Sequence, Tuple, Optional, Union

import math
from jaxtyping import ArrayLike


class PixelShuffle(nn.Module):
    scale: int

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return rearrange(
            x, "b h w (h2 w2 c) -> b (h h2) (w w2) c", h2=self.scale, w2=self.scale
        )


def sinusoidal_embedding(
    t: Union[ArrayLike, float], out_dim: int = 64, max_period: int = 10_000
) -> ArrayLike:
    """The so-called sinusoidal positional embedding.

    Parameters
    ----------
    t : FloatScalar
        A time variable. Note that this is the discrete time.
    out_dim : int
        The output dimension.
    max_period : int
        The maximum period.

    Returns
    -------
    JArray (..., out_dim)
        An array.

    Notes
    -----
    I have no idea what a sinusoidal positional embedding does. Perhaps, it means to find a function that maps a time
    scalar to a sequence. The implementation is based on
        - https://github.com/JTT94/diffusion_schrodinger_bridge/blob/1c82eba0a16aea3333ac738dde376b12a3f97f21/
        bridge/models/basic/time_embedding.py#L6
        - https://github.com/vdeborto/cdsb/blob/8fc9cc2a08daa083b84b5ddd38190bec931edeb0/
        bridge/models/unet/layers.py#L95
    """
    half = out_dim // 2

    fs = jnp.exp(-math.log(max_period) * jnp.arange(half) / (half - 1))
    embs = t * fs
    embs = jnp.concatenate([jnp.sin(embs), jnp.cos(embs)], axis=-1)
    if out_dim % 2 == 1:
        raise NotImplementedError(
            f"out_dim is implemented for even number only, while {out_dim} is given."
        )
    return embs


def l2norm(t, axis=1, eps=1e-12):
    """Performs L2 normalization of inputs over specified axis.

    Args:
      t: jnp.ndarray of any shape
      axis: the dimension to reduce, default -1
      eps: small value to avoid division by zero. Default 1e-12
    Returns:
      normalized array of same shape as t


    """
    denom = jnp.clip(jnp.linalg.norm(t, ord=2, axis=axis, keepdims=True), eps)
    out = t / denom
    return out


class Downsample(nn.Module):
    dim: Optional[int] = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        dim = self.dim if self.dim is not None else C
        x = nn.Conv(
            dim, kernel_size=(4, 4), strides=(2, 2), padding=1, dtype=self.dtype
        )(x)
        assert x.shape == (B, H // 2, W // 2, dim)
        return x


class Upsample(nn.Module):
    method: str = "resize"
    dim: Optional[int] = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        dim = self.dim if self.dim is not None else C
        if self.method == "resize":
            x = jax.image.resize(x, (B, H * 2, W * 2, C), "linear")
            x = nn.Conv(dim, kernel_size=(3, 3), padding=1, dtype=self.dtype)(x)
        elif self.method == "pixel_shuffle":
            x = nn.Conv(C * 4, kernel_size=(3, 3), padding=1, dtype=self.dtype)(x)
            x = PixelShuffle(scale=2)(x)
            x = nn.Conv(dim, kernel_size=(3, 3), padding=1, dtype=self.dtype)(x)
        else:
            raise ValueError(f"Unknown upsampling method: {self.method}")
        assert x.shape == (B, H * 2, W * 2, dim)
        return x


class WeightStandardizedConv(nn.Module):
    """
    apply weight standardization  https://arxiv.org/abs/1903.10520
    """

    features: int
    kernel_size: Sequence[int] = 3
    strides: Union[None, int, Sequence[int]] = 1
    padding: Any = 1
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        """
        Applies a weight standardized convolution to the inputs.

        Args:
          inputs: input data with dimensions (batch, spatial_dims..., features).

        Returns:
          The convolved data.
        """
        x = x.astype(self.dtype)

        conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            parent=None,
        )

        kernel_init = lambda rng, x: conv.init(rng, x)["params"]["kernel"]
        bias_init = lambda rng, x: conv.init(rng, x)["params"]["bias"]

        # standardize kernel
        kernel = self.param("kernel", kernel_init, x)
        eps = 1e-5 if self.dtype == jnp.float32 else 1e-3
        # reduce over dim_out
        redux = tuple(range(kernel.ndim - 1))
        mean = jnp.mean(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        var = jnp.var(kernel, axis=redux, dtype=self.dtype, keepdims=True)
        standardized_kernel = (kernel - mean) / jnp.sqrt(var + eps)

        bias = self.param("bias", bias_init, x)

        return conv.apply({"params": {"kernel": standardized_kernel, "bias": bias}}, x)


class ResnetBlock(nn.Module):
    """Convolutional residual block."""

    dim: int = None
    groups: Optional[int] = 8
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, time_emb):
        """
        Args:
          x: jnp.ndarray of shape [B, H, W, C]
          time_emb: jnp.ndarray of shape [B,D]
        Returns:
          x: jnp.ndarray of shape [B, H, W, C]
        """

        B, _, _, C = x.shape
        assert time_emb.shape[0] == B and len(time_emb.shape) == 2

        h = WeightStandardizedConv(
            features=self.dim, kernel_size=(3, 3), padding=1, name="conv_0"
        )(x)
        h = nn.GroupNorm(num_groups=self.groups, dtype=self.dtype, name="norm_0")(h)

        # add in timestep embedding
        time_emb = nn.Dense(
            features=2 * self.dim, dtype=self.dtype, name="time_mlp.dense_0"
        )(nn.swish(time_emb))
        time_emb = time_emb[:, jnp.newaxis, jnp.newaxis, :]  # [B, H, W, C]
        scale, shift = jnp.split(time_emb, 2, axis=-1)
        h = h * (1 + scale) + shift

        h = nn.swish(h)

        h = WeightStandardizedConv(
            features=self.dim, kernel_size=(3, 3), padding=1, name="conv_1"
        )(h)
        h = nn.swish(
            nn.GroupNorm(num_groups=self.groups, dtype=self.dtype, name="norm_1")(h)
        )

        if C != self.dim:
            x = nn.Conv(
                features=self.dim,
                kernel_size=(1, 1),
                dtype=self.dtype,
                name="res_conv_0",
            )(x)

        assert x.shape == h.shape

        return x + h


class Attention(nn.Module):
    heads: int = 4
    dim_head: int = 32
    scale: int = 10
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        dim = self.dim_head * self.heads

        qkv = nn.Conv(
            features=dim * 3,
            kernel_size=(1, 1),
            use_bias=False,
            dtype=self.dtype,
            name="to_qkv.conv_0",
        )(x)  # [B, H, W, dim *3]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # [B, H, W, dim]
        q, k, v = map(
            lambda t: rearrange(t, "b x y (h d) -> b (x y) h d", h=self.heads),
            (q, k, v),
        )

        assert q.shape == k.shape == v.shape == (B, H * W, self.heads, self.dim_head)

        q, k = map(l2norm, (q, k))

        sim = jnp.einsum("b i h d, b j h d -> b h i j", q, k) * self.scale
        attn = nn.softmax(sim, axis=-1)
        assert attn.shape == (B, self.heads, H * W, H * W)

        out = jnp.einsum("b h i j , b j h d  -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b x y (h d)", x=H)
        assert out.shape == (B, H, W, dim)

        out = nn.Conv(
            features=C, kernel_size=(1, 1), dtype=self.dtype, name="to_out.conv_0"
        )(out)
        return out


class LinearAttention(nn.Module):
    heads: int = 4
    dim_head: int = 32
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        dim = self.dim_head * self.heads

        qkv = nn.Conv(
            features=dim * 3,
            kernel_size=(1, 1),
            use_bias=False,
            dtype=self.dtype,
            name="to_qkv.conv_0",
        )(x)  # [B, H, W, dim *3]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # [B, H, W, dim]
        q, k, v = map(
            lambda t: rearrange(t, "b x y (h d) -> b (x y) h d", h=self.heads),
            (q, k, v),
        )
        assert q.shape == k.shape == v.shape == (B, H * W, self.heads, self.dim_head)
        # compute softmax for q along its embedding dimensions
        q = nn.softmax(q, axis=-1)
        # compute softmax for k along its spatial dimensions
        k = nn.softmax(k, axis=-3)

        q = q / jnp.sqrt(self.dim_head)
        v = v / (H * W)

        context = jnp.einsum("b n h d, b n h e -> b h d e", k, v)
        out = jnp.einsum("b h d e, b n h d -> b h e n", context, q)
        out = rearrange(out, "b h e (x y) -> b x y (h e)", x=H)
        assert out.shape == (B, H, W, dim)

        out = nn.Conv(
            features=C, kernel_size=(1, 1), dtype=self.dtype, name="to_out.conv_0"
        )(out)
        out = nn.LayerNorm(
            epsilon=1e-5, use_bias=False, dtype=self.dtype, name="to_out.norm_0"
        )(out)
        return out


class AttnBlock(nn.Module):
    heads: int = 4
    dim_head: int = 32
    use_linear_attention: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        normed_x = nn.LayerNorm(epsilon=1e-5, use_bias=False, dtype=self.dtype)(x)
        if self.use_linear_attention:
            attn = LinearAttention(self.heads, self.dim_head, dtype=self.dtype)
        else:
            attn = Attention(self.heads, self.dim_head, dtype=self.dtype)
        out = attn(normed_x)
        assert out.shape == (B, H, W, C)
        return out + x


class UNet(nn.Module):
    dt: float
    dim: int
    upsampling: str = "resize"
    init_dim: Optional[int] = None  # if None, same as dim
    out_dim: Optional[int] = None
    dim_mults: Tuple[int, int, int, int] = (1, 2, 4)
    resnet_block_groups: int = 8
    learnt_variance: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, time):
        if x.ndim < 4:
            x = jnp.expand_dims(x, 0)
        B, H, W, C = x.shape

        init_dim = self.dim if self.init_dim is None else self.init_dim
        hs = []
        h = nn.Conv(
            features=init_dim,
            kernel_size=(7, 7),
            padding=3,
            name="init.conv_0",
            dtype=self.dtype,
        )(x)

        hs.append(h)
        # use sinusoidal embeddings to encode timesteps
        if time.ndim < 1:
            time_emb = jnp.broadcast_to(
                sinusoidal_embedding(time / self.dt, out_dim=self.dim), (B, self.dim)
            )
        else:
            time_emb = jax.vmap(lambda z: sinusoidal_embedding(z, out_dim=self.dim))(
                time / self.dt
            )
        time_emb = nn.Dense(features=self.dim * 4, dtype=self.dtype)(time_emb)
        time_emb = nn.Dense(features=self.dim * 4, dtype=self.dtype)(
            nn.gelu(time_emb)
        )  # [B, 4*dim]

        # downsampling
        num_resolutions = len(self.dim_mults)
        for ind in range(num_resolutions):
            dim_in = h.shape[-1]
            h = ResnetBlock(
                dim=dim_in, groups=self.resnet_block_groups, dtype=self.dtype
            )(h, time_emb)
            hs.append(h)

            h = ResnetBlock(
                dim=dim_in, groups=self.resnet_block_groups, dtype=self.dtype
            )(h, time_emb)
            h = AttnBlock(dtype=self.dtype, name=f"down_{ind}.attnblock_0")(h)
            hs.append(h)

            if ind < num_resolutions - 1:
                h = Downsample(
                    dim=self.dim * self.dim_mults[ind],
                    dtype=self.dtype,
                    name=f"down_{ind}.downsample_0",
                )(h)

        mid_dim = self.dim * self.dim_mults[-1]
        h = nn.Conv(
            features=mid_dim,
            kernel_size=(3, 3),
            padding=1,
            dtype=self.dtype,
            name=f"down_{num_resolutions - 1}.conv_0",
        )(h)

        # middle
        h = ResnetBlock(
            dim=mid_dim,
            groups=self.resnet_block_groups,
            dtype=self.dtype,
            name="mid.resblock_0",
        )(h, time_emb)
        h = AttnBlock(
            use_linear_attention=False, dtype=self.dtype, name="mid.attenblock_0"
        )(h)
        h = ResnetBlock(
            dim=mid_dim,
            groups=self.resnet_block_groups,
            dtype=self.dtype,
            name="mid.resblock_1",
        )(h, time_emb)

        # upsampling
        for ind in reversed(range(num_resolutions)):
            dim_in = self.dim * self.dim_mults[ind]
            dim_out = self.dim * self.dim_mults[ind - 1] if ind > 0 else init_dim

            assert h.shape[-1] == dim_in
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            assert h.shape[-1] == dim_in + dim_out
            h = ResnetBlock(
                dim=dim_in,
                groups=self.resnet_block_groups,
                dtype=self.dtype,
                name=f"up_{ind}.resblock_0",
            )(h, time_emb)

            h = jnp.concatenate([h, hs.pop()], axis=-1)
            assert h.shape[-1] == dim_in + dim_out
            h = ResnetBlock(
                dim=dim_in,
                groups=self.resnet_block_groups,
                dtype=self.dtype,
                name=f"up_{ind}.resblock_1",
            )(h, time_emb)
            h = AttnBlock(dtype=self.dtype, name=f"up_{ind}.attnblock_0")(h)

            assert h.shape[-1] == dim_in
            if ind > 0:
                h = Upsample(
                    dim=dim_out,
                    method=self.upsampling,
                    dtype=self.dtype,
                    name=f"up_{ind}.upsample_0",
                )(h)

        h = nn.Conv(
            features=init_dim,
            kernel_size=(3, 3),
            padding=1,
            dtype=self.dtype,
            name=f"up_0.conv_0",
        )(h)

        # final
        h = jnp.concatenate([h, hs.pop()], axis=-1)
        assert h.shape[-1] == init_dim * 2

        out = ResnetBlock(
            dim=self.dim,
            groups=self.resnet_block_groups,
            dtype=self.dtype,
            name="final.resblock_0",
        )(h, time_emb)

        default_out_dim = C * (1 if not self.learnt_variance else 2)
        out_dim = default_out_dim if self.out_dim is None else self.out_dim

        out = nn.Conv(
            out_dim, kernel_size=(1, 1), dtype=self.dtype, name="final.conv_0"
        )(out)

        if B == 1:
            return out[0]
        else:
            return out
