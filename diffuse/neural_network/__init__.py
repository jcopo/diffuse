from diffuse.neural_network.unet import UNet
from diffuse.neural_network.autoencoderKL import AutoencoderKL

model_zoo = {
    "UNet": UNet,
    "AutoencoderKL": AutoencoderKL
}