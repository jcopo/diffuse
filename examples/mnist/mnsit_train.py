import jax
import jax.numpy as jnp
import numpy as np
import optax
import einops
from tqdm import tqdm
from pathlib import Path

from diffuse.neural_network.unet import UNet
from diffuse.neural_network.losses import score_match_loss
from diffuse.diffusion.sde import SDE, LinearSchedule
from diffuse.timer.base import VpTimer

# Configuration
CONFIG = {
    "data": {
        "batch_size": 128,  # Reduced for better gradient estimates
        "data_path": "dataset/mnist.npz",
    },
    "model": {
        "channels": 128,  # Increased capacity
        "upsampling": "pixel_shuffle",
    },
    "sde": {
        "b_min": 0.1,  # Increased for better training stability
        "b_max": 20.0,  # Increased for stronger noise
        "t0": 0.001,  # Small but non-zero
        "T": 2.0,
    },
    "training": {
        "n_epochs": 5000,  # More epochs for better convergence
        "lr_init": 1e-4,  # Conservative learning rate
        "lr_final": 1e-6,  # Lower final LR
        "warmup_epochs": 100,  # Warmup for stability
        "ema_decay": 0.9999,  # Stronger EMA
        "grad_clip": 1.0,
        "save_every": 500,
    },
    "paths": {"weights_dir": "weights", "logs_dir": "logs"},
}


def setup_directories():
    """Create necessary directories"""
    Path(CONFIG["paths"]["weights_dir"]).mkdir(exist_ok=True)
    Path(CONFIG["paths"]["logs_dir"]).mkdir(exist_ok=True)


def load_and_preprocess_data(key):
    """Load and preprocess MNIST data"""
    data = jnp.load(CONFIG["data"]["data_path"])
    xs = data["X"]

    # Shuffle data
    xs = jax.random.permutation(key, xs, axis=0)
    # Add channel dimension and normalize
    xs = einops.rearrange(xs, "b h w -> b h w 1")
    xs = xs.astype(jnp.float32)

    print(f"Data shape: {xs.shape}")
    print(f"Data range: [{xs.min():.3f}, {xs.max():.3f}]")

    return xs


def create_model_and_sde():
    """Initialize the UNet model and SDE"""
    # SDE setup
    beta = LinearSchedule(
        b_min=CONFIG["sde"]["b_min"], b_max=CONFIG["sde"]["b_max"], t0=CONFIG["sde"]["t0"], T=CONFIG["sde"]["T"]
    )
    sde = SDE(beta=beta, tf=CONFIG["sde"]["T"])

    # Timer for training
    timer = VpTimer(n_steps=1000, eps=CONFIG["sde"]["t0"], tf=CONFIG["sde"]["T"])

    # Model setup
    model = UNet(
        dt=CONFIG["sde"]["T"] / 1000,  # Fine-grained time discretization
        channels=CONFIG["model"]["channels"],
        upsampling=CONFIG["model"]["upsampling"],
    )

    return model, sde, timer


def create_optimizer():
    """Create optimizer with warmup and cosine decay"""
    total_steps = CONFIG["training"]["n_epochs"] * (60000 // CONFIG["data"]["batch_size"])
    warmup_steps = CONFIG["training"]["warmup_epochs"] * (60000 // CONFIG["data"]["batch_size"])

    # Learning rate schedule with warmup
    warmup_schedule = optax.linear_schedule(
        init_value=0.0, end_value=CONFIG["training"]["lr_init"], transition_steps=warmup_steps
    )

    cosine_schedule = optax.cosine_decay_schedule(
        init_value=CONFIG["training"]["lr_init"],
        decay_steps=total_steps - warmup_steps,
        alpha=CONFIG["training"]["lr_final"] / CONFIG["training"]["lr_init"],
    )

    lr_schedule = optax.join_schedules(schedules=[warmup_schedule, cosine_schedule], boundaries=[warmup_steps])

    # Optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(CONFIG["training"]["grad_clip"]),
        optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999, eps=1e-8),
    )

    return optimizer


def weight_function(t, sde):
    """Improved weighting function for score matching loss"""
    alpha_t, _ = sde.alpha_beta(t)
    # Weight more heavily at higher noise levels
    return 1.0 / alpha_t


@jax.jit
def train_step(key, params, opt_state, ema_state, batch, model, sde, optimizer, ema_kernel):
    """Single training step"""

    def loss_fn(params):
        return score_match_loss(params, key, batch, sde, model, lmbda=lambda t: weight_function(t, sde))

    loss_val, grads = jax.value_and_grad(loss_fn)(params)

    # Apply gradients
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # Update EMA
    ema_params, ema_state = ema_kernel.update(params, ema_state)

    return params, opt_state, ema_state, loss_val, ema_params


def create_data_loader(data, batch_size, key):
    """Create batched data loader"""
    n_samples = data.shape[0]
    n_batches = n_samples // batch_size

    # Shuffle indices
    indices = jax.random.permutation(key, n_samples)
    indices = indices[: n_batches * batch_size]  # Drop remainder
    indices = indices.reshape(n_batches, batch_size)

    return indices


def main():
    """Main training loop"""
    setup_directories()

    # Initialize
    key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
    key_data, key_init, key = jax.random.split(key, 3)

    # Load data
    data = load_and_preprocess_data(key_data)

    # Create model and SDE
    model, sde, timer = create_model_and_sde()

    # Initialize model parameters
    sample_batch = data[: CONFIG["data"]["batch_size"]]
    sample_times = jnp.ones((CONFIG["data"]["batch_size"],))
    params = model.init(key_init, sample_batch, sample_times)

    print(f"Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(params))} parameters")

    # Create optimizer and EMA
    optimizer = create_optimizer()
    opt_state = optimizer.init(params)

    ema_kernel = optax.ema(CONFIG["training"]["ema_decay"])
    ema_state = ema_kernel.init(params)
    ema_params = params

    # Training loop
    n_batches_per_epoch = data.shape[0] // CONFIG["data"]["batch_size"]

    for epoch in range(CONFIG["training"]["n_epochs"]):
        # Create data loader for this epoch
        key_epoch, key = jax.random.split(key)
        batch_indices = create_data_loader(data, CONFIG["data"]["batch_size"], key_epoch)

        epoch_losses = []
        pbar = tqdm(range(n_batches_per_epoch), desc=f"Epoch {epoch + 1}/{CONFIG['training']['n_epochs']}")

        for batch_idx in pbar:
            key_step, key = jax.random.split(key)
            batch = data[batch_indices[batch_idx]]

            params, opt_state, ema_state, loss_val, ema_params = train_step(
                key_step, params, opt_state, ema_state, batch, model, sde, optimizer, ema_kernel
            )

            epoch_losses.append(float(loss_val))
            pbar.set_postfix({"loss": f"{loss_val:.6f}"})

        # Log epoch results
        mean_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1}: Mean Loss = {mean_loss:.6f}")

        # Save checkpoints
        if (epoch + 1) % CONFIG["training"]["save_every"] == 0:
            checkpoint_path = f"{CONFIG['paths']['weights_dir']}/ann_{epoch}.npz"
            np.savez(checkpoint_path, params=params, ema_params=ema_params, epoch=epoch, loss=mean_loss)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = f"{CONFIG['paths']['weights_dir']}/ann_final.npz"
    np.savez(final_path, params=params, ema_params=ema_params, epoch=CONFIG["training"]["n_epochs"], config=CONFIG)
    print(f"Training complete! Final model saved: {final_path}")


if __name__ == "__main__":
    main()
