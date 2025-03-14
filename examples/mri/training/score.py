# Standard library imports
from typing import Tuple, Callable
from functools import partial

# Third-party imports
import jax
import optax

# Local imports
from .base import BaseTrainer
from diffuse.neural_network import UNet
from diffuse.neural_network.losses import score_match_loss, noise_match_loss, weight_fun
from diffuse.diffusion.sde import LinearSchedule, SDE


class ScoreModelTrainer(BaseTrainer):
    def _init_model(
        self,
    ) -> Tuple[dict, optax.GradientTransformation, optax.GradientTransformation]:
        self.model = UNet(**self.config["neural_network"]["unet"])

        self.key, subkey = jax.random.split(self.key)
        params = self.model.init_weights(subkey)

        noise_schedule = LinearSchedule(
            b_min=0.02,
            b_max=5.0,
            t0=self.config["sde"]["t0"], T=self.config["sde"]["tf"]
        )
        self.sde = SDE(beta=noise_schedule, tf=self.config["sde"]["tf"])

        optimizer, ema_kernel = self._configure_optimizer()
        return params, optimizer, ema_kernel

    def _define_loss(self) -> Callable:
        if self.config["loss"]["score"]["type"] == "score_matching":
            lmbda = jax.vmap(partial(weight_fun, sde=self.sde))
            return partial(
                score_match_loss, sde=self.sde, network=self.model, lmbda=lmbda
            )
        elif self.config["loss"]["score"]["type"] == "noise_matching":
            return partial(noise_match_loss, sde=self.sde, network=self.model)
