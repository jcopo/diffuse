import jax
import jax.numpy as jnp
from envyaml import EnvYAML

from diffuse.denoisers.denoiser import Denoiser
from diffuse.diffusion.sde import SDE, LinearSchedule
from diffuse.integrator.deterministic import DPMpp2sIntegrator
from diffuse.neural_network.unet import UNet
from diffuse.neural_network.unett import UNet as Unet
from examples.mri.brats.create_dataset import get_dataloader as get_brats_dataloader
from examples.mri.wmh.create_dataset import get_dataloader as get_wmh_dataloader
from examples.mri.fastMRI.create_dataset import get_dataloader as get_fastmri_dataloader
from examples.mri.utils import get_first_item, load_checkpoint
from diffuse.utils.plotting import plot_lines
from diffuse.integrator.stochastic import EulerMaruyama
from diffuse.utils.plotting import sigle_plot

dataloader_zoo = {
    "WMH": lambda cfg: get_wmh_dataloader(cfg, train=False),
    "BRATS": lambda cfg: get_brats_dataloader(cfg, train=False),
    "fastMRI": lambda cfg: get_fastmri_dataloader(cfg, train=False),
}


def initialize_experiment(key, config):
    """Initialize all components needed for generation."""
    data_model = config["dataset"]

    dataloader = dataloader_zoo[data_model](config)
    xs = get_first_item(dataloader)

    n_t = config["inference"]["n_t"]
    tf = config["sde"]["tf"]
    key, subkey = jax.random.split(key)
    ground_truth = jax.random.choice(subkey, xs)

    beta = LinearSchedule(
        b_min=config["sde"]["beta_min"],
        b_max=config["sde"]["beta_max"],
        t0=config["sde"]["t0"],
        T=tf,
    )

    if config["score_model"] == "UNet":
        score_net = UNet(
            config["unet"]["dt_embedding"],
            config["unet"]["embedding_dim"],
            upsampling=config["unet"]["upsampling"],
            dim_mults=config["unet"]["dim_mults"],
        )
    elif config["score_model"] == "UNett":
        score_net = Unet(dim=config["unet"]["embedding_dim"])
    else:
        raise ValueError(f"Score model {config['score_model']} not found")

    _, params, _, _, _ = load_checkpoint(config)
    nn_score = lambda x, t: score_net.apply(params, x, t)

    sde = SDE(beta=beta, tf=tf)
    shape = xs[0].shape

    return sde, shape, n_t, nn_score, ground_truth


def main(
    key, n_samples=50, config_path="examples/mri/configs/config_fastMRI_inference.yaml"
):
    config = EnvYAML(config_path)

    # Get ground truth data
    sde, shape, n_t, nn_score, ground_truth = initialize_experiment(key, config)

    # Plot ground truth
    ground_truth_complex = ground_truth[..., 0] + 1j * ground_truth[..., 1]
    abs_ground_truth = jnp.abs(ground_truth_complex)
    sigle_plot(abs_ground_truth)

    n_t = 100

    # Initialize denoiser
    # integrator = DPMpp2sIntegrator(sde)
    integrator = EulerMaruyama(sde)
    denoiser = Denoiser(
        integrator=integrator,
        sde=sde,
        score=nn_score,
        n_steps=n_t,
        x0_shape=shape,
    )

    # Generate samples
    keys = jax.random.split(key, n_samples)
    vec_generator = jax.jit(jax.vmap(denoiser.generate))
    state, hist = vec_generator(keys)

    # Plot results
    samples = state.integrator_state.position
    abs_samples = jnp.abs(samples[..., 0] + 1j * samples[..., 1])
    plot_lines(abs_samples)

    return state, hist


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="examples/mri/configs/config_fastMRI_inference.yaml",
    )
    parser.add_argument("--n_samples", type=int, default=50)

    args = parser.parse_args()

    state, hist = main(n_samples=args.n_samples, config_path=args.config)
