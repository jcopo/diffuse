import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union

from diffuse.denoisers.denoiser import Denoiser
from diffuse.denoisers.cond import DPSDenoiser, TMPDenoiser, FPSDenoiser, CondDenoiser
from diffuse.integrator.base import Integrator
from diffuse.integrator.deterministic import EulerIntegrator, HeunIntegrator, DPMpp2sIntegrator, DDIMIntegrator
from diffuse.integrator.stochastic import EulerMaruyamaIntegrator
from diffuse.diffusion.sde import SDE, LinearSchedule
from diffuse.neural_network.unet import UNet
from diffuse.utils.plotting import plot_lines
from diffuse.timer.base import VpTimer
from examples.mnist.images import SquareMask  # Import the mask class

# Global configuration dictionary
CONFIG = {
    "schedules": {
        "LinearSchedule": {"params": {"b_min": 0.02, "b_max": 5.0, "t0": 0.0, "T": 2.0}, "class": LinearSchedule}
    },
    "integrators": {
        "euler": (EulerIntegrator, {"stochastic_churn_rate": 0.0}),
        "heun": (
            HeunIntegrator,
            {"stochastic_churn_rate": 0.0, "churn_min": 0.5, "churn_max": 2.0, "noise_inflation_factor": 1.0001},
        ),
        "dpmpp2s": (
            DPMpp2sIntegrator,
            {"stochastic_churn_rate": 0.0, "churn_min": 0.5, "churn_max": 2.0, "noise_inflation_factor": 1.0001},
        ),
        "ddim": (DDIMIntegrator, {}),
        "euler_maruyama": (EulerMaruyamaIntegrator, {}),
    },
    "denoisers": {
        "unconditional": (Denoiser, {"x0_shape": (28, 28, 1)}),
        "dps": (DPSDenoiser, {"forward_model": SquareMask(size=7, img_shape=(28, 28, 1))}),
        "tmp": (TMPDenoiser, {"forward_model": SquareMask(size=7, img_shape=(28, 28, 1))}),
        "fps": (FPSDenoiser, {"forward_model": SquareMask(size=7, img_shape=(28, 28, 1))}),
    },
    "space": {"t_init": 0.0, "t_final": 2.0, "n_samples": 3, "n_steps": 300},
}


def initialize_experiment(key: jax.random.PRNGKey, config_train: dict):
    """Initialize the experiment components"""
    # Load MNIST dataset
    data = np.load(os.path.join(config_train["path_dataset"], "mnist.npz"))
    xs = jnp.array(data["X"])
    xs = xs.reshape(xs.shape[0], xs.shape[1], xs.shape[2], 1)

    # Initialize SDE
    schedule_config = CONFIG["schedules"]["LinearSchedule"]
    params = schedule_config["params"].copy()
    params["T"] = config_train["sde"]["tf"]
    beta = schedule_config["class"](**params)
    sde = SDE(beta=beta, tf=config_train["sde"]["tf"])

    # Initialize ScoreNetwork
    score_net = UNet(
        dt=config_train["neural_network"]["unet"]["dt"],
        dim=config_train["neural_network"]["unet"]["dim"],
        upsampling=config_train["neural_network"]["unet"]["upsampling"],
        dim_mults=config_train["neural_network"]["unet"]["dim_mults"],
        sample_size=config_train["neural_network"]["unet"]["sample_size"],
        channel_size=config_train["neural_network"]["unet"]["channel_size"],
    )

    # Load trained model parameters
    model_path = f"{config_train['model_dir']}/best_model.npz"
    nn_trained = jnp.load(model_path, allow_pickle=True)
    params = nn_trained["ema_params"].item()  # Use EMA parameters for better stability

    def nn_score(x, t):
        return score_net.apply(params, x, t)

    # If using noise matching loss, convert to score
    if config_train.get("training", {}).get("loss") in ["noise_matching", "mae_noise_matching"]:
        nn_score = sde.noise_to_score(nn_score)

    return sde, nn_score, xs


def create_integrator(integrator_type: str, sde: SDE, timer: VpTimer) -> Integrator:
    """Create an integrator of the specified type"""
    if integrator_type not in CONFIG["integrators"]:
        raise ValueError(f"Unknown integrator type: {integrator_type}")

    integrator_class, integrator_params = CONFIG["integrators"][integrator_type]
    return integrator_class(sde=sde, timer=timer, **integrator_params)


def create_denoiser(
    integrator: Integrator, sde: SDE, score_fn: Callable, denoiser_type: str, **kwargs
) -> Union[Denoiser, CondDenoiser]:
    """Create a denoiser of the specified type with additional kwargs"""
    if denoiser_type not in CONFIG["denoisers"]:
        raise ValueError(f"Unknown denoiser type: {denoiser_type}")

    denoiser_class, denoiser_params = CONFIG["denoisers"][denoiser_type]
    # Merge default params with provided kwargs
    params = {**denoiser_params, **kwargs}

    return denoiser_class(integrator=integrator, sde=sde, score=score_fn, **params)


def run_unconditional_experiment(key: jax.random.PRNGKey, integrator_type: str = "dpmpp2s"):
    """Run unconditional sampling experiment"""
    # Initialize experiment components
    sde, nn_score, _ = initialize_experiment(key, config_train)  # Ignore ground truth data

    # Create timer
    timer = VpTimer(n_steps=CONFIG["space"]["n_steps"], eps=CONFIG["space"]["t_init"], tf=CONFIG["space"]["t_final"])

    # Create integrator
    integrator = create_integrator(integrator_type, sde, timer)

    # Create unconditional denoiser
    denoiser = create_denoiser(integrator, sde, nn_score, "unconditional")

    # Generate samples
    key_samples, _ = jax.random.split(key)

    state, hist = denoiser.generate(key_samples, CONFIG["space"]["n_steps"], CONFIG["space"]["n_samples"])

    return state, hist


def run_conditional_experiment(
    key: jax.random.PRNGKey, denoiser_type: str = "dps", integrator_type: str = "euler_maruyama", mask_size: int = 7
):
    """Run conditional sampling experiment with measurement"""
    # Initialize experiment components
    sde, nn_score, ground_truth_data = initialize_experiment(key, config_train)

    # Select ground truth image
    ground_truth = jax.random.choice(key, ground_truth_data)

    # Create timer
    timer = VpTimer(n_steps=CONFIG["space"]["n_steps"], eps=CONFIG["space"]["t_init"], tf=CONFIG["space"]["t_final"])

    # Create integrator
    integrator = create_integrator(integrator_type, sde, timer)

    # Create forward model (mask) with custom size if provided
    forward_model = SquareMask(size=mask_size, img_shape=(28, 28, 1))

    # Create conditional denoiser
    denoiser = create_denoiser(integrator, sde, nn_score, denoiser_type, forward_model=forward_model)

    # Create measurement state
    key_meas, key_samples = jax.random.split(key)
    measurement_state = forward_model.init(key_meas)
    xi = forward_model.init_design(key_meas)
    new_measurement = forward_model.measure(key_meas, ground_truth, xi)
    measurement_state = forward_model.update_measurement(measurement_state, new_measurement, xi)
    # Visualize measurement state
    visualize_measurement_state(measurement_state, forward_model, ground_truth)

    # Generate samples
    state, hist = denoiser.generate(
        key_samples, measurement_state, CONFIG["space"]["n_steps"], CONFIG["space"]["n_samples"]
    )

    return state, hist


def visualize_measurement_state(measurement_state, mask, ground_truth):
    """Visualize the measurement state with values"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot ground truth
    im1 = ax1.imshow(ground_truth[..., 0], cmap="gray")
    ax1.set_title("Ground Truth")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1)

    # Plot measurement mask
    mask_vis = mask.make(measurement_state.xi)[..., 0]
    im2 = ax2.imshow(mask_vis, cmap="gray")
    ax2.set_title("Measurement Mask")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2)

    # Plot measured values
    measured_values = measurement_state.y[..., 0]
    im3 = ax3.imshow(measured_values, cmap="gray")
    ax3.set_title("Measured Values")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage:
    key = jax.random.PRNGKey(42)

    # Example 1: Unconditional sampling with DPM++2S
    state, hist = run_unconditional_experiment(key, integrator_type="dpmpp2s")
    print("DPM++2S unconditional samples:")
    plot_lines(state.integrator_state.position)
    plot_lines(hist[-1])

    # Example 2: Conditional sampling with DPS
    state, hist = run_conditional_experiment(key, denoiser_type="dps", integrator_type="euler_maruyama", mask_size=7)
    print("DPS conditional samples:")
    plot_lines(state.integrator_state.position)
    plot_lines(hist[-1])
