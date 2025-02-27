# Standard library imports
from typing import Tuple, Callable
from functools import partial

# Third-party imports
import jax
import optax

# Local imports
from .base import BaseTrainer
from diffuse.neural_network import AutoencoderKL
from diffuse.neural_network.losses import kl_loss


class LatentModelTrainer(BaseTrainer):
    def _init_model(
        self,
    ) -> Tuple[dict, optax.GradientTransformation, optax.GradientTransformation]:
        self.model = AutoencoderKL(**self.config["neural_network"]["autoencoder"])

        self.key, subkey = jax.random.split(self.key)
        params = self.model.init_weights(subkey)

        optimizer, ema_kernel = self._configure_optimizer(latent=True)
        return params, optimizer, ema_kernel

    def _define_loss(self) -> Callable:
        return partial(
            kl_loss, beta=self.config["loss"]["latent"]["beta"], network=self.model
        )
