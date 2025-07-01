from diffuse.neural_network.SDvae import AutoEncoder, AutoEncoderParams
from diffuse.neural_network.unet import UNet, UNetParams

model_zoo = {
    "UNet": (UNet, UNetParams),
    "SDVAE": (AutoEncoder, AutoEncoderParams),
}
