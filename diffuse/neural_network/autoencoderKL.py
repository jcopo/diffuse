"""
This script is adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl.py#L37
under the Apache-2.0 license.
"""

import einops

import flax.linen as nn
import jax.numpy as jnp
import jax

from jaxtyping import ArrayLike, PyTreeDef, PRNGKeyArray
from typing import Callable, Tuple


from diffuse.neural_network.encoder import Encoder
from diffuse.neural_network.decoder import Decoder


class FlaxDiagonalGaussianDistribution(object):
    def __init__(self, parameters: ArrayLike, deterministic: bool = False):
        # Last axis to account for channels-last
        self.mean, self.logvar = jnp.split(parameters, 2, axis=-1)
        self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = jnp.exp(0.5 * self.logvar)
        self.var = jnp.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = jnp.zeros_like(self.mean)

    def sample(self, rng: PRNGKeyArray):
        return self.mean + self.std * jax.random.normal(rng, self.mean.shape)

    def kl(self):
        if self.deterministic:
            return jnp.array([0.0])

        return 0.5 * jnp.sum(
            self.mean**2 + self.var - 1.0 - self.logvar, axis=[1, 2, 3]
        )

    def mode(self):
        return self.mean


class AutoencoderKL(nn.Module):
    r"""
    Flax implementation of a VAE model with KL loss for decoding latent representations.

    This model inherits from [`FlaxModelMixin`]. Check the superclass documentation for it's generic methods
    implemented for all models (such as downloading or saving).

    This model is a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax Linen module and refer to the Flax documentation for all matter related to its
    general usage and behavior.

    Inherent JAX features such as the following are supported:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        in_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        out_channels (`int`, *optional*, defaults to 3):
            Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `(DownEncoderBlock2D)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `(UpDecoderBlock2D)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[str]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`):
            Number of ResNet layer for each block.
        act_fn (`str`, *optional*, defaults to `silu`):
            The activation function to use.
        latent_channels (`int`, *optional*, defaults to `4`):
            Number of channels in the latent space.
        norm_num_groups (`int`, *optional*, defaults to `32`):
            The number of groups for normalization.
        sample_size (`int`, *optional*, defaults to 32):
            Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):
            The `dtype` of the parameters.
    """

    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Tuple[str] = ("DownEncoderBlock2D",)
    up_block_types: Tuple[str] = ("UpDecoderBlock2D",)
    block_out_channels: Tuple[int] = (64,)
    layers_per_block: int = 1
    act_fn: str = "silu"
    latent_channels: int = 4
    norm_num_groups: int = 32
    sample_size: int = 32
    scaling_factor: float = 0.18215
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.encoder = Encoder(
            in_channels=self.in_channels,
            out_channels=self.latent_channels,
            down_block_types=self.down_block_types,
            block_out_channels=self.block_out_channels,
            layers_per_block=self.layers_per_block,
            act_fn=self.act_fn,
            norm_num_groups=self.norm_num_groups,
            double_z=True,
            dtype=self.dtype,
        )
        self.decoder = Decoder(
            in_channels=self.latent_channels,
            out_channels=self.out_channels,
            up_block_types=self.up_block_types,
            block_out_channels=self.block_out_channels,
            layers_per_block=self.layers_per_block,
            norm_num_groups=self.norm_num_groups,
            act_fn=self.act_fn,
            dtype=self.dtype,
        )
        self.quant_conv = nn.Conv(
            2 * self.latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )
        self.post_quant_conv = nn.Conv(
            self.latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )

    def init_weights(self, rng: PRNGKeyArray):
        # init input tensors
        sample_shape = (1, self.in_channels, self.sample_size, self.sample_size)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)

        params_rng, dropout_rng, gaussian_rng = jax.random.split(rng, 3)
        rngs = {"params": params_rng, "dropout": dropout_rng, "gaussian": gaussian_rng}

        return self.init(rngs, sample)

    def encode(self, sample: ArrayLike, deterministic: bool = True):
        sample = jnp.transpose(sample, (0, 2, 3, 1))
        hidden_states = self.encoder(sample, deterministic=deterministic)
        moments = self.quant_conv(hidden_states)
        posterior = FlaxDiagonalGaussianDistribution(moments)

        return posterior

    def decode(self, latents: ArrayLike, deterministic: bool = True):
        if latents.shape[-1] != self.latent_channels:
            latents = jnp.transpose(latents, (0, 2, 3, 1))

        hidden_states = self.post_quant_conv(latents)
        hidden_states = self.decoder(hidden_states, deterministic=deterministic)

        hidden_states = jnp.transpose(hidden_states, (0, 3, 1, 2))

        return hidden_states

    def __call__(
        self, sample: ArrayLike, deterministic: bool = True, training: bool = True
    ):
        posterior = self.encode(sample, deterministic=deterministic)
        rng = self.make_rng("gaussian")
        hidden_states = posterior.sample(rng)

        sample = self.decode(hidden_states, deterministic=deterministic)

        if training:
            return posterior, sample

        else:
            return sample


def ELBO(rngs: PRNGKeyArray, nn_params: PyTreeDef, x: ArrayLike, autoencoder: Callable):
    posterior, sample = autoencoder.apply(nn_params, x, rngs=rngs)

    kl_value = posterior.kl()
    mse_value = einops.reduce((sample - x) ** 2, "t ... -> t ", "mean")

    return jnp.mean(kl_value + mse_value)
