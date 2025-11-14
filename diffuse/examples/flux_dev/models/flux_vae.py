#!/usr/bin/env python3
"""
FLUX VAE Implementation
FluxVAE class matching real FLUX.1 AutoencoderKL architecture.
"""

import jax
import jax.numpy as jnp
from flax import nnx

#!/usr/bin/env python3
"""
FLUX VAE Implementation - Clean and Focused
Single FluxVAE class with proper NNX weight loading from SafeTensors.
"""


# Core imports

# Orbax imports

# Required imports - will fail if not available


class FluxVAE(nnx.Module):
    """FLUX.1 VAE model implementation matching real AutoencoderKL architecture."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        param_dtype=jnp.float32,
        rngs: nnx.Rngs = None,
    ):
        """Initialize FLUX.1 VAE to match real AutoencoderKL architecture."""
        if rngs is None:
            rngs = nnx.Rngs(42)

        self.latent_channels = latent_channels

        # Encoder - matching real FLUX.1 structure
        self.encoder = nnx.Module()

        # encoder.conv_in
        self.encoder.conv_in = nnx.Conv(
            in_features=in_channels,
            out_features=128,
            kernel_size=(3, 3),
            padding=(1, 1),
            param_dtype=param_dtype,
            rngs=rngs,
        )

        # encoder.down_blocks (4 blocks: 128->128, 128->256, 256->512, 512->512)
        encoder_down_blocks = []

        # Down block 0: 128 -> 128
        down_block_0 = nnx.Module()
        down_block_0_resnets = []
        for _j in range(2):
            resnet = nnx.Module()
            resnet.norm1 = nnx.GroupNorm(num_features=128, num_groups=32, param_dtype=param_dtype, rngs=rngs)
            resnet.conv1 = nnx.Conv(128, 128, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
            resnet.norm2 = nnx.GroupNorm(num_features=128, num_groups=32, param_dtype=param_dtype, rngs=rngs)
            resnet.conv2 = nnx.Conv(128, 128, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
            down_block_0_resnets.append(resnet)
        down_block_0.resnets = nnx.List(down_block_0_resnets)
        down_block_0.downsamplers = nnx.List(
            [nnx.Conv(128, 128, kernel_size=(3, 3), strides=(2, 2), padding="VALID", param_dtype=param_dtype, rngs=rngs)]
        )
        encoder_down_blocks.append(down_block_0)

        # Down block 1: 128 -> 256
        down_block_1 = nnx.Module()
        down_block_1_resnets = []
        # First resnet: 128 -> 256
        resnet_0 = nnx.Module()
        resnet_0.norm1 = nnx.GroupNorm(num_features=128, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_0.conv1 = nnx.Conv(128, 256, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        resnet_0.norm2 = nnx.GroupNorm(num_features=256, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_0.conv2 = nnx.Conv(256, 256, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        resnet_0.conv_shortcut = nnx.Conv(128, 256, kernel_size=(1, 1), param_dtype=param_dtype, rngs=rngs)
        down_block_1_resnets.append(resnet_0)
        # Second resnet: 256 -> 256
        resnet_1 = nnx.Module()
        resnet_1.norm1 = nnx.GroupNorm(num_features=256, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_1.conv1 = nnx.Conv(256, 256, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        resnet_1.norm2 = nnx.GroupNorm(num_features=256, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_1.conv2 = nnx.Conv(256, 256, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        down_block_1_resnets.append(resnet_1)
        down_block_1.resnets = nnx.List(down_block_1_resnets)
        down_block_1.downsamplers = nnx.List(
            [nnx.Conv(256, 256, kernel_size=(3, 3), strides=(2, 2), padding="VALID", param_dtype=param_dtype, rngs=rngs)]
        )
        encoder_down_blocks.append(down_block_1)

        # Down block 2: 256 -> 512
        down_block_2 = nnx.Module()
        down_block_2_resnets = []
        # First resnet: 256 -> 512
        resnet_0 = nnx.Module()
        resnet_0.norm1 = nnx.GroupNorm(num_features=256, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_0.conv1 = nnx.Conv(256, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        resnet_0.norm2 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_0.conv2 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        resnet_0.conv_shortcut = nnx.Conv(256, 512, kernel_size=(1, 1), param_dtype=param_dtype, rngs=rngs)
        down_block_2_resnets.append(resnet_0)
        # Second resnet: 512 -> 512
        resnet_1 = nnx.Module()
        resnet_1.norm1 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_1.conv1 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        resnet_1.norm2 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_1.conv2 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        down_block_2_resnets.append(resnet_1)
        down_block_2.resnets = nnx.List(down_block_2_resnets)
        down_block_2.downsamplers = nnx.List(
            [nnx.Conv(512, 512, kernel_size=(3, 3), strides=(2, 2), padding="VALID", param_dtype=param_dtype, rngs=rngs)]
        )
        encoder_down_blocks.append(down_block_2)

        # Down block 3: 512 -> 512 (final)
        down_block_3 = nnx.Module()
        down_block_3_resnets = []
        # First resnet: 512 -> 512
        resnet_0 = nnx.Module()
        resnet_0.norm1 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_0.conv1 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        resnet_0.norm2 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_0.conv2 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        down_block_3_resnets.append(resnet_0)
        # Second resnet: 512 -> 512
        resnet_1 = nnx.Module()
        resnet_1.norm1 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_1.conv1 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        resnet_1.norm2 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_1.conv2 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        down_block_3_resnets.append(resnet_1)
        down_block_3.resnets = nnx.List(down_block_3_resnets)
        # No downsampler for final block
        encoder_down_blocks.append(down_block_3)

        self.encoder.down_blocks = nnx.List(encoder_down_blocks)

        # encoder.mid_block
        self.encoder.mid_block = nnx.Module()
        encoder_mid_resnets = []

        # First resnet in mid block
        resnet_0 = nnx.Module()
        resnet_0.norm1 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_0.conv1 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        resnet_0.norm2 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_0.conv2 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        encoder_mid_resnets.append(resnet_0)

        # Attention in mid block (FLUX uses Conv2D with 1x1 kernels, not Linear)
        encoder_mid_attentions = []
        attention_0 = nnx.Module()
        attention_0.group_norm = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        attention_0.to_q = nnx.Conv(512, 512, kernel_size=(1, 1), param_dtype=param_dtype, rngs=rngs)
        attention_0.to_k = nnx.Conv(512, 512, kernel_size=(1, 1), param_dtype=param_dtype, rngs=rngs)
        attention_0.to_v = nnx.Conv(512, 512, kernel_size=(1, 1), param_dtype=param_dtype, rngs=rngs)
        attention_0.to_out = nnx.List([nnx.Conv(512, 512, kernel_size=(1, 1), param_dtype=param_dtype, rngs=rngs)])
        encoder_mid_attentions.append(attention_0)

        # Second resnet in mid block
        resnet_1 = nnx.Module()
        resnet_1.norm1 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_1.conv1 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        resnet_1.norm2 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_1.conv2 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        encoder_mid_resnets.append(resnet_1)
        self.encoder.mid_block.resnets = nnx.List(encoder_mid_resnets)
        self.encoder.mid_block.attentions = nnx.List(encoder_mid_attentions)

        # encoder.conv_norm_out and conv_out
        self.encoder.conv_norm_out = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        self.encoder.conv_out = nnx.Conv(
            in_features=512,
            out_features=2 * latent_channels,  # 32 for mean and logvar (double_z=True)
            kernel_size=(3, 3),
            padding=(1, 1),
            param_dtype=param_dtype,
            rngs=rngs,
        )

        # FLUX.1 VAE does NOT have quant_conv or post_quant_conv
        # These are AutoencoderKL specific layers that don't exist in FLUX

        # Decoder - matching real FLUX.1 structure
        self.decoder = nnx.Module()

        # decoder.conv_in
        self.decoder.conv_in = nnx.Conv(
            in_features=latent_channels,
            out_features=512,
            kernel_size=(3, 3),
            padding=(1, 1),
            param_dtype=param_dtype,
            rngs=rngs,
        )

        # decoder.mid_block (same structure as encoder mid_block)
        self.decoder.mid_block = nnx.Module()
        decoder_mid_resnets = []

        # First resnet in decoder mid block
        resnet_0 = nnx.Module()
        resnet_0.norm1 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_0.conv1 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        resnet_0.norm2 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_0.conv2 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        decoder_mid_resnets.append(resnet_0)

        # Attention in decoder mid block (FLUX uses Conv2D with 1x1 kernels, not Linear)
        decoder_mid_attentions = []
        attention_0 = nnx.Module()
        attention_0.group_norm = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        attention_0.to_q = nnx.Conv(512, 512, kernel_size=(1, 1), param_dtype=param_dtype, rngs=rngs)
        attention_0.to_k = nnx.Conv(512, 512, kernel_size=(1, 1), param_dtype=param_dtype, rngs=rngs)
        attention_0.to_v = nnx.Conv(512, 512, kernel_size=(1, 1), param_dtype=param_dtype, rngs=rngs)
        attention_0.to_out = nnx.List([nnx.Conv(512, 512, kernel_size=(1, 1), param_dtype=param_dtype, rngs=rngs)])
        decoder_mid_attentions.append(attention_0)

        # Second resnet in decoder mid block
        resnet_1 = nnx.Module()
        resnet_1.norm1 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_1.conv1 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        resnet_1.norm2 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        resnet_1.conv2 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
        decoder_mid_resnets.append(resnet_1)
        self.decoder.mid_block.resnets = nnx.List(decoder_mid_resnets)
        self.decoder.mid_block.attentions = nnx.List(decoder_mid_attentions)

        # decoder.up_blocks - FLUX exact channel progression: [512, 512, 256, 128]
        decoder_up_blocks = []

        # Up block 0: 512 channels throughout (FLUX ResNet structure)
        # FLUX decoder uses num_res_blocks + 1 = 3 resnets per up_block
        up_block_0 = nnx.Module()
        up_block_0_resnets = []
        for _j in range(3):  # num_res_blocks + 1
            resnet = nnx.Module()
            # All resnets in up_block_0 are 512->512
            resnet.norm1 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
            resnet.conv1 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
            resnet.norm2 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
            resnet.conv2 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
            up_block_0_resnets.append(resnet)
        up_block_0.resnets = nnx.List(up_block_0_resnets)
        # FLUX uses interpolation + Conv, not ConvTranspose
        up_block_0.upsamplers = nnx.List([nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)])
        decoder_up_blocks.append(up_block_0)

        # Up block 1: 512 channels throughout (FLUX ResNet structure)
        up_block_1 = nnx.Module()
        up_block_1_resnets = []
        for _j in range(3):  # num_res_blocks + 1
            resnet = nnx.Module()
            # All resnets in up_block_1 are 512->512
            resnet.norm1 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
            resnet.conv1 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
            resnet.norm2 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
            resnet.conv2 = nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
            up_block_1_resnets.append(resnet)
        up_block_1.resnets = nnx.List(up_block_1_resnets)
        # FLUX uses interpolation + Conv, not ConvTranspose
        up_block_1.upsamplers = nnx.List([nnx.Conv(512, 512, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)])
        decoder_up_blocks.append(up_block_1)

        # Up block 2: 512 -> 256 transition (FLUX ResNet structure)
        up_block_2 = nnx.Module()
        up_block_2_resnets = []
        for j in range(3):  # num_res_blocks + 1
            resnet = nnx.Module()
            if j == 0:
                # First resnet: 512 -> 256 (channel reduction)
                resnet.norm1 = nnx.GroupNorm(num_features=512, num_groups=32, param_dtype=param_dtype, rngs=rngs)
                resnet.conv1 = nnx.Conv(512, 256, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
                resnet.norm2 = nnx.GroupNorm(num_features=256, num_groups=32, param_dtype=param_dtype, rngs=rngs)
                resnet.conv2 = nnx.Conv(256, 256, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
                # Shortcut for channel dimension change
                resnet.conv_shortcut = nnx.Conv(512, 256, kernel_size=(1, 1), param_dtype=param_dtype, rngs=rngs)
            else:
                # Subsequent resnets: 256 -> 256
                resnet.norm1 = nnx.GroupNorm(num_features=256, num_groups=32, param_dtype=param_dtype, rngs=rngs)
                resnet.conv1 = nnx.Conv(256, 256, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
                resnet.norm2 = nnx.GroupNorm(num_features=256, num_groups=32, param_dtype=param_dtype, rngs=rngs)
                resnet.conv2 = nnx.Conv(256, 256, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
            up_block_2_resnets.append(resnet)
        up_block_2.resnets = nnx.List(up_block_2_resnets)
        # FLUX uses interpolation + Conv, not ConvTranspose
        up_block_2.upsamplers = nnx.List([nnx.Conv(256, 256, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)])
        decoder_up_blocks.append(up_block_2)

        # Up block 3: 256 -> 128 transition (final, FLUX ResNet structure)
        up_block_3 = nnx.Module()
        up_block_3_resnets = []
        for j in range(3):  # num_res_blocks + 1
            resnet = nnx.Module()
            if j == 0:
                # First resnet: 256 -> 128 (channel reduction)
                resnet.norm1 = nnx.GroupNorm(num_features=256, num_groups=32, param_dtype=param_dtype, rngs=rngs)
                resnet.conv1 = nnx.Conv(256, 128, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
                resnet.norm2 = nnx.GroupNorm(num_features=128, num_groups=32, param_dtype=param_dtype, rngs=rngs)
                resnet.conv2 = nnx.Conv(128, 128, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
                # Shortcut for channel dimension change
                resnet.conv_shortcut = nnx.Conv(256, 128, kernel_size=(1, 1), param_dtype=param_dtype, rngs=rngs)
            else:
                # Subsequent resnets: 128 -> 128
                resnet.norm1 = nnx.GroupNorm(num_features=128, num_groups=32, param_dtype=param_dtype, rngs=rngs)
                resnet.conv1 = nnx.Conv(128, 128, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
                resnet.norm2 = nnx.GroupNorm(num_features=128, num_groups=32, param_dtype=param_dtype, rngs=rngs)
                resnet.conv2 = nnx.Conv(128, 128, kernel_size=(3, 3), padding=(1, 1), param_dtype=param_dtype, rngs=rngs)
            up_block_3_resnets.append(resnet)
        up_block_3.resnets = nnx.List(up_block_3_resnets)
        # No upsampler for final block
        decoder_up_blocks.append(up_block_3)

        self.decoder.up_blocks = nnx.List(decoder_up_blocks)

        # decoder.conv_norm_out and conv_out
        self.decoder.conv_norm_out = nnx.GroupNorm(num_features=128, num_groups=32, param_dtype=param_dtype, rngs=rngs)
        self.decoder.conv_out = nnx.Conv(
            in_features=128,
            out_features=in_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _apply_attention(self, h, attention_module):
        """Apply self-attention in the mid-block using Conv2D with 1x1 kernels."""
        batch_size, height, width, channels = h.shape

        # Normalize input
        h_norm = attention_module.group_norm(h)

        # Apply 1x1 convolutions to compute Q, K, V (keeps spatial dimensions)
        q = attention_module.to_q(h_norm)  # (B, H, W, C)
        k = attention_module.to_k(h_norm)  # (B, H, W, C)
        v = attention_module.to_v(h_norm)  # (B, H, W, C)

        # Flatten spatial dimensions for attention computation
        q_flat = q.reshape(batch_size, height * width, channels)
        k_flat = k.reshape(batch_size, height * width, channels)
        v_flat = v.reshape(batch_size, height * width, channels)

        # Use built-in dot product attention (matches FLUX's scaled_dot_product_attention)
        attn_out = nnx.dot_product_attention(q_flat, k_flat, v_flat)

        # Reshape back to spatial dimensions before 1x1 conv
        attn_out = attn_out.reshape(batch_size, height, width, channels)

        # Project output using 1x1 conv
        attn_out = attention_module.to_out[0](attn_out)

        # Apply residual connection
        return h + attn_out

    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder.conv_in(x)
        skip_connections = []

        # Down blocks
        for down_block in self.encoder.down_blocks:
            # ResNet blocks
            for resnet in down_block.resnets:
                residual = h
                h = nnx.swish(resnet.norm1(h))
                h = resnet.conv1(h)
                h = nnx.swish(resnet.norm2(h))
                h = resnet.conv2(h)

                # Apply shortcut if present
                if hasattr(resnet, "conv_shortcut"):
                    residual = resnet.conv_shortcut(residual)
                h = h + residual

            # Store skip connection before downsampling
            skip_connections.append(h)

            # Downsample if present - apply FLUX-style asymmetric padding
            if hasattr(down_block, "downsamplers") and down_block.downsamplers:
                # FLUX asymmetric padding: (0, 1, 0, 1) = (left, right, top, bottom)
                # JAX format: ((top, bottom), (left, right))
                h = jnp.pad(h, ((0, 0), (0, 1), (0, 1), (0, 0)), mode="constant", constant_values=0)
                h = down_block.downsamplers[0](h)

        # Mid block
        for resnet in self.encoder.mid_block.resnets:
            residual = h
            h = nnx.swish(resnet.norm1(h))
            h = resnet.conv1(h)
            h = nnx.swish(resnet.norm2(h))
            h = resnet.conv2(h)
            h = h + residual

        # Apply attention in mid block
        for attention in self.encoder.mid_block.attentions:
            h = self._apply_attention(h, attention)

        # Final normalization and output
        h = nnx.swish(self.encoder.conv_norm_out(h))
        h = self.encoder.conv_out(h)

        # FLUX.1 VAE doesn't use quant_conv - direct split
        # Split into mean and logvar
        mean, logvar = jnp.split(h, 2, axis=-1)
        return mean, logvar, skip_connections

    def decode(self, z, skip_connections=None):
        """Decode latent to reconstruction."""
        # FLUX.1 VAE doesn't use post_quant_conv - direct input
        h = self.decoder.conv_in(z)

        # Mid block
        for resnet in self.decoder.mid_block.resnets:
            residual = h
            h = nnx.swish(resnet.norm1(h))
            h = resnet.conv1(h)
            h = nnx.swish(resnet.norm2(h))
            h = resnet.conv2(h)
            h = h + residual

        # Apply attention in decoder mid block
        for attention in self.decoder.mid_block.attentions:
            h = self._apply_attention(h, attention)

        # Up blocks - FLUX doesn't use skip connections in decoder
        for _i, up_block in enumerate(self.decoder.up_blocks):
            # ResNet blocks
            for _j, resnet in enumerate(up_block.resnets):
                residual = h
                h = nnx.swish(resnet.norm1(h))
                h = resnet.conv1(h)
                h = nnx.swish(resnet.norm2(h))
                h = resnet.conv2(h)

                # Apply shortcut if present (for channel dimension changes)
                if hasattr(resnet, "conv_shortcut"):
                    residual = resnet.conv_shortcut(residual)
                h = h + residual

            # Upsample if present - FLUX uses interpolation + conv
            if hasattr(up_block, "upsamplers") and up_block.upsamplers:
                # FLUX upsampling: interpolate then apply conv
                # JAX format is NHWC, so h.shape = (batch, height, width, channels)
                h = jax.image.resize(h, (h.shape[0], h.shape[1] * 2, h.shape[2] * 2, h.shape[3]), method="nearest")
                h = up_block.upsamplers[0](h)

        # Final normalization and output
        h = nnx.swish(self.decoder.conv_norm_out(h))
        h = self.decoder.conv_out(h)

        return h

    def __call__(self, x):
        """Full VAE forward pass."""
        mean, logvar, skip_connections = self.encode(x)

        # Reparameterization trick
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(jax.random.PRNGKey(0), mean.shape)
        z = mean + std * eps

        x_recon = self.decode(z, skip_connections)
        return x_recon, mean, logvar
