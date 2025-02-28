# Standard library imports
import sys
from typing import Any, Tuple, Callable, Optional, Iterator

# Third-party imports
import jax
from jax import numpy as jnp
import optax
from tqdm import tqdm

# Local imports
from .utils import (
    load_checkpoint,
    load_best_model_checkpoint,
    checkpoint_model,
    checkpoint_best_model,
    get_sharding,
)


class BaseTrainer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.key = jax.random.PRNGKey(0)
        self.sharding, self.replicated_sharding = get_sharding()
        self.latent = self.__class__.__name__ == "LatentModelTrainer"

    def _init_model(
        self,
    ) -> Tuple[dict, optax.GradientTransformation, optax.GradientTransformation]:
        """Initialize model and parameters - to be implemented by subclasses"""
        raise NotImplementedError

    def _define_loss(self) -> Callable:
        """Define loss function - to be implemented by subclasses"""
        raise NotImplementedError

    def _configure_optimizer(
        self, latent: bool = False
    ) -> Tuple[optax.GradientTransformation, optax.GradientTransformation]:
        """Configure optimizer with common settings"""
        schedule_params = (
            self.config["training"]["latent"]["cosine_schedule"]
            if latent
            else self.config["training"]["score"]["cosine_schedule"]
        )
        schedule = optax.cosine_decay_schedule(**schedule_params)

        optimizer = optax.adam(learning_rate=schedule)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optimizer)

        ema_params = (
            self.config["training"]["latent"]["ema"]
            if latent
            else self.config["training"]["score"]["ema"]
        )
        ema_kernel = optax.ema(**ema_params)
        return optimizer, ema_kernel

    def _load_checkpoint(
        self,
    ) -> Optional[Tuple[dict, optax.OptState, optax.EmaState, range, float]]:
        """Load checkpoint with common logic"""
        try:
            params, _, ema_state, opt_state, begin_epoch = load_checkpoint(self.config)
            n_epochs = (
                self.config["training"]["latent"]["n_epochs"]
                if self.latent
                else self.config["training"]["score"]["n_epochs"]
            )
            iterator_epoch = range(begin_epoch, n_epochs)
            _, _, best_val_loss = load_best_model_checkpoint(self.config)
            return params, opt_state, ema_state, iterator_epoch, best_val_loss

        except ValueError:
            return None

    def _configure_training_step(
        self,
        optimizer: optax.GradientTransformation,
        ema_kernel: optax.GradientTransformation,
        loss_fn: Callable,
    ) -> Callable:
        """Configure and JIT compile the training step"""

        def step(
            key: jnp.ndarray,
            params: dict,
            opt_state: optax.OptState,
            ema_state: optax.EmaState,
            data: Any,
        ) -> Tuple[dict, optax.OptState, optax.EmaState, float, dict]:
            val_loss, g = jax.value_and_grad(loss_fn)(params, key, data)
            updates, opt_state = optimizer.update(g, opt_state)
            params = optax.apply_updates(params, updates)
            ema_params, ema_state = ema_kernel.update(params, ema_state)
            return params, opt_state, ema_state, val_loss, ema_params

        return jax.jit(step)

    def _configure_validation_step(self, loss_fn: Callable) -> Callable:
        """Configure and JIT compile the validation step"""

        def validate_step(key: jnp.ndarray, params: dict, data: Any) -> float:
            val_loss = loss_fn(params, key, data)
            return val_loss

        return jax.jit(validate_step)

    def _train_epoch(
        self,
        train_loader: Iterator,
        params: dict,
        opt_state: optax.OptState,
        ema_state: optax.EmaState,
        batch_update: Callable,
    ) -> Tuple[dict, optax.OptState, optax.EmaState, dict]:
        """Run one training epoch"""
        list_loss = []
        train_iterator = tqdm(train_loader, desc="Training", file=sys.stdout)

        for batch in train_iterator:
            self.key, subkey = jax.random.split(self.key)
            batch = jax.device_put(batch, self.sharding)

            params, opt_state, ema_state, val_loss, ema_params = batch_update(
                subkey, params, opt_state, ema_state, batch
            )

            train_iterator.set_postfix({"loss": val_loss})
            list_loss.append(val_loss)

        print(
            f"Training epoch finished, training loss: {sum(list_loss) / len(list_loss)}"
        )

        return params, opt_state, ema_state, ema_params

    def _validate_epoch(
        self, val_loader: Iterator, params: dict, batch_validate: Callable
    ) -> float:
        """Run one validation epoch"""
        list_val_loss = []
        val_iterator = tqdm(val_loader, desc="Validation", file=sys.stdout)

        for batch in val_iterator:
            self.key, subkey = jax.random.split(self.key)
            batch = jax.device_put(batch, self.sharding)

            val_loss = batch_validate(subkey, params, batch)
            val_iterator.set_postfix({"loss": val_loss})
            list_val_loss.append(val_loss)

        print(
            f"Validation epoch finished, validation loss: {sum(list_val_loss) / len(list_val_loss)}"
        )
        return sum(list_val_loss) / len(list_val_loss)

    def _save_checkpoints(
        self,
        params: dict,
        opt_state: optax.OptState,
        ema_state: optax.EmaState,
        ema_params: dict,
        epoch: int,
        val_loss: float,
        best_val_loss: float,
    ) -> None:
        """Save model checkpoints"""
        # Save regular checkpoint
        checkpoint_model(
            self.config,
            params,
            opt_state,
            ema_state,
            ema_params,
            epoch,
            latent=self.latent,
        )

        # Save best model checkpoint if improved
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss} to {val_loss}")
            checkpoint_best_model(
                self.config,
                params,
                ema_params,
                val_loss,
                latent=self.latent,
            )

    def train(self, train_val_loaders: Tuple[Iterator, Iterator]) -> None:
        """Main training loop with common logic"""
        train_loader, val_loader = train_val_loaders

        # Initialize model and training components
        params, optimizer, ema_kernel = self._init_model()
        loss_fn = self._define_loss()

        # Configure training steps
        batch_update = self._configure_training_step(optimizer, ema_kernel, loss_fn)
        batch_validate = self._configure_validation_step(loss_fn)

        # Load checkpoint or initialize states
        checkpoint_data = self._load_checkpoint()
        if checkpoint_data:
            params, opt_state, ema_state, iterator_epoch, best_val_loss = (
                checkpoint_data
            )

        else:
            opt_state = optimizer.init(params)
            ema_state = ema_kernel.init(params)
            n_epochs = (
                self.config["training"]["latent"]["n_epochs"]
                if self.latent
                else self.config["training"]["score"]["n_epochs"]
            )
            iterator_epoch = range(n_epochs)
            best_val_loss = float("inf")

        params = jax.device_put(params, self.replicated_sharding)
        opt_state = jax.device_put(opt_state, self.replicated_sharding)
        ema_state = jax.device_put(ema_state, self.replicated_sharding)

        # Training loop
        for epoch in iterator_epoch:
            params, opt_state, ema_state, ema_params = self._train_epoch(
                train_loader, params, opt_state, ema_state, batch_update
            )
            val_loss = self._validate_epoch(val_loader, params, batch_validate)

            self._save_checkpoints(
                params, opt_state, ema_state, ema_params, epoch, val_loss, best_val_loss
            )
            best_val_loss = min(best_val_loss, val_loss)
