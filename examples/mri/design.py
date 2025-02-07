import csv
import os

import jax
import jax.numpy as jnp
import optax
from envyaml import EnvYAML
from jaxtyping import PRNGKeyArray
import matplotlib.pyplot as plt

from diffuse.design.bayesian_design import ExperimentOptimizer, ExperimentRandom
from diffuse.denoisers.cond_denoiser import CondDenoiser
from diffuse.denoisers.cond_tweedie import CondTweedie
from diffuse.diffusion.sde import SDE, LinearSchedule
from diffuse.integrator.stochastic import EulerMaruyama
from diffuse.integrator.deterministic import DPMpp2sIntegrator, Euler, DDIMIntegrator, HeunIntegrator
from diffuse.neural_network.unet import UNet
from diffuse.neural_network.unett import UNet as Unet
from examples.mri.brats.create_dataset import get_dataloader as get_brats_dataloader
from examples.mri.wmh.create_dataset import get_dataloader as get_wmh_dataloader
from examples.mri.fastMRI.create_dataset import get_dataloader as get_fastmri_dataloader
from examples.mri.brainFastMRI.create_dataset import get_dataloader as get_brainfastmri_dataloader
from examples.mri.kneeFastMRI.create_dataset import get_dataloader as get_kneefastmri_dataloader
from examples.mri.evals import WMHExperiment
from examples.mri.forward_models import maskRadial, maskSpiral, maskVertical, maskRandom
from examples.mri.logger import MRILogger, ExperimentLogger
from examples.mri.utils import get_first_item, load_checkpoint, load_best_model_checkpoint, get_sharding
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


from lpips_j.lpips import LPIPS
from functools import partial


import sys

# get user from environment variable
USER = os.getenv("USER")
WORKDIR = os.getenv("WORK")

dataloader_zoo = {
    "WMH": lambda cfg: get_wmh_dataloader(cfg, train=False),
    "BRATS": lambda cfg: get_brats_dataloader(cfg, train=False),
    "fastMRI": lambda cfg: get_fastmri_dataloader(cfg, train=False),
    "brainFastMRI": lambda cfg: get_brainfastmri_dataloader(cfg, train=False),
    "kneeFastMRI": lambda cfg: get_kneefastmri_dataloader(cfg, train=False),
}


def initialize_experiment(key: PRNGKeyArray, config: dict):
    """Initialize all components needed for the experiment."""
    data_model = config['dataset']
    model_dir = config['model_dir']

    dataloader = dataloader_zoo[data_model](config)
    xs = get_first_item(dataloader)

    key, subkey = jax.random.split(key)
    ground_truth = jax.random.choice(key, xs)

    n_t = config['inference']['n_t']
    tf = config['sde']['tf']

    dt = tf / n_t

    beta = LinearSchedule(
        b_min=config['sde']['beta_min'],
        b_max=config['sde']['beta_max'],
        t0=config['sde']['t0'],
        T=tf
    )

    if config['score_model'] == "UNet":
        score_net = UNet(config["unet"]["dt_embedding"], config["unet"]["embedding_dim"], upsampling=config["unet"]["upsampling"], dim_mults=config["unet"]["dim_mults"])
    elif config['score_model'] == "UNett":
        score_net = Unet(dim=config['unet']['embedding_dim'])
    else:
        raise ValueError(f"Score model {config['score_model']} not found")

    _, params, _ = load_best_model_checkpoint(config)

    def nn_score(x, t):
        score_value = score_net.apply(params, x, t)
        return score_value
    sde = SDE(beta=beta, tf=tf)

    if config.get('training', {}).get('loss') in ["noise_matching", "mae_noise_matching"]:
        nn_score = sde.noise_to_score(nn_score)

    shape = ground_truth.shape

    # Get mask configuration
    if config['mask']['mask_type'] == 'spiral':
        mask = maskSpiral(img_shape=shape, task=config['task'], num_samples=8000, sigma=1., data_model=config['dataset'])

    elif config['mask']['mask_type'] == 'radial':
        mask = maskRadial(
            num_lines= config['mask']['num_lines'],
            img_shape=shape,
            task=config['task'],
            data_model=config['dataset']
        )

    elif config['mask']['mask_type'] == 'vertical':
        mask = maskVertical(num_lines=10, # config['mask']['num_lines'],
                            img_shape=shape,
                            task=config['task'],
                            data_model=config['dataset']
                        )

    # Initialize experiment
    experiment = WMHExperiment(mask=mask)

    return sde, mask, ground_truth, dt, n_t, nn_score, experiment


def main(
    num_measurements: int,
    key: PRNGKeyArray,
    config: dict,
    prefix: str = "",
    space: str = "runs",
    logging: bool = False,
    random: bool = False,
    save_plots: bool = True,
    experiment_name: str = "",
):
    print("Running with config: \n \
            path_dataset: ", config['path_dataset'], "\n \
            model_dir: ", config['model_dir'], "\n \
            dataset: ", config['dataset'], "\n \
            task: ", config['task'], "\n \
            mask_type: ", config['mask']['mask_type'], "\n \
            num_lines: ", config['mask']['num_lines'], "\n \
            n_t: ", config['inference']['n_t'], "\n \
            n_samples: ", config['inference']['n_samples'], "\n \
            n_samples_cntrst: ", config['inference']['n_samples_cntrst'], "\n \
            n_loop_opt: ", config['inference']['n_loop_opt'], "\n \
            ", file=sys.stdout, flush=True)

    # Initialize experiment components
    sde, mask, ground_truth, dt, n_t, nn_score, experiment = initialize_experiment(key, config)
    lpips = LPIPS()
    abs_ground_truth = jnp.abs(ground_truth[..., 0] + 1j * ground_truth[..., 1])[..., None]
    params_lpips = lpips.init(key, abs_ground_truth, abs_ground_truth)
    lpips_fn = partial(lpips.apply, params_lpips)

    # Initialize logger
    logger = MRILogger(
        config=config,
        rng_key=key,
        experiment=experiment,
        prefix=prefix,
        space=space,
        random=random,
        save_plots=save_plots,
        experiment_name=experiment_name,
        lpips_fn=lpips_fn
    ) if logging else None
    devices = jax.devices()
    n_samples = config['inference']['n_samples']
    n_samples_cntrst = config['inference']['n_samples_cntrst']
    n_loop_opt = config['inference']['n_loop_opt']
    n_opt_steps = n_t * n_loop_opt

    # Conditional Denoiser
    # integrator = EulerMaruyama(sde)
    # integrator = DDIMIntegrator(sde)
    # integrator = Euler(sde)
    integrator = DPMpp2sIntegrator(sde=sde, stochastic_churn_rate=1., churn_min=1., churn_max=2., noise_inflation_factor=1.001)
    # integrator = HeunIntegrator(sde=sde, stochastic_churn_rate=1., churn_min=1., churn_max=2., noise_inflation_factor=1.001)
    resample = True

    print(f'Using {integrator.__class__.__name__} as integrator') 

    # denoiser = CondDenoiser(integrator, sde, nn_score, mask, resample)
    denoiser = CondTweedie(integrator, sde, nn_score, mask, resample)

    # init design and measurement

    xi = mask.init_design(key)

    if config['mask']['mask_type'] == 'vertical':
        measurement_state = mask.init_measurement(ground_truth)
    else:
        measurement_state = mask.init_measurement(xi)

    schedule = optax.exponential_decay(
        init_value=config['inference']['lr'],
        transition_steps=n_opt_steps,
        decay_rate=0.98,
        transition_begin=int(n_opt_steps * 0.25),
        staircase=False,
    )

    # ExperimentOptimizer
    optimizer = optax.chain(
        optax.adam(learning_rate=schedule),
        optax.scale(-1)
    )
    experiment_optimizer = ExperimentOptimizer(
        denoiser, mask, optimizer, ground_truth.shape
    )
    if random:
        experiment_optimizer = ExperimentRandom(denoiser, mask, ground_truth.shape)

    exp_state = experiment_optimizer.init(key, n_samples, n_samples_cntrst, dt)

    def scan_step(carry, n_meas):
        exp_state, measurement_state, key = carry

        key, subkey = jax.random.split(key)

        # Minimal debug print just for step tracking
        jax.debug.print("Processing step {n}", n=n_meas)
        jax.debug.print("design start: {}", exp_state.design)

        optimal_state, history = experiment_optimizer.get_design(
            exp_state, subkey, measurement_state, n_steps=n_t, n_loop_opt=n_opt_steps
        )
        # import pdb; pdb.set_trace()
        jax.debug.print("design optimal: {}", optimal_state.design)
        if logger and len(devices) == 1:
            logger.log(
                ground_truth,
                optimal_state,
                measurement_state,
                history,
                n_meas
            )

        # make new measurement
        measurement_state = mask.update_measurement(
            measurement_state, ground_truth, optimal_state.design
        )

        exp_state = experiment_optimizer.init(subkey, n_samples, n_samples_cntrst, dt)
        # Return the full optimal_state instead of just denoiser_state
        return (exp_state, measurement_state, key), (optimal_state, measurement_state)

    init_carry = (exp_state, measurement_state, key)
    (exp_state, final_measurement_state, key), (optimal_states, measurement_states) = jax.lax.scan(
        scan_step, init_carry, jnp.arange(num_measurements)
    )

    # Do logging after scan completes
    if logging and len(devices) > 1:
        def log_all_steps(ground_truth, optimal_states, measurement_states):
            from IPython.display import display, clear_output
            try:
                print("\n==== Starting Post-Scan Logging ====")
                for step in range(num_measurements):
                    print(f"\nLogging step {step}")
                    # Get the states for this step
                    step_state = jax.tree.map(lambda x: x[step], optimal_states)
                    step_measurement = jax.tree.map(lambda x: x[step], measurement_states)
                    logger.log(ground_truth, step_state, step_measurement, step)
                    plt.close('all')
                print("\n==== Logging Complete ====")
                display(flush=True)
            except Exception as e:
                print(f"Error during logging: {str(e)}")
                import traceback
                traceback.print_exc()
            return None

        # Get data to host
        optimal_states_host = jax.device_get(optimal_states)
        measurement_states_host = jax.device_get(measurement_states)
        ground_truth_host = jax.device_get(ground_truth)

        # Do the logging on host
        log_all_steps(ground_truth_host, optimal_states_host, measurement_states_host)

    return ground_truth, optimal_states, final_measurement_state.y


if __name__ == "__main__":
    import argparse
    import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument("--rng_key", type=int, default=0)
    parser.add_argument("--num_meas", type=int, default=3)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--space", type=str, default="runs")
    parser.add_argument("--config", type=str, default="examples/mri/configs/config_fastMRI_inference.yaml")

    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    # Load configuration
    config = EnvYAML(args.config)

    key_int = args.rng_key
    plot = args.plot
    num_meas = args.num_meas
    rng_key = jax.random.PRNGKey(key_int)

    # Set up directory paths
    timestamp = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
    key_save = rng_key[1]
    experiment_name = f"{args.prefix}/{key_save}_{timestamp}"

    ground_truth, optimal_state, final_measurement = main(
        num_meas,
        rng_key,
        config,
        prefix=args.prefix,
        space=args.space,
        logging=args.plot,
        experiment_name=experiment_name
    )


    ground_truth_random, optimal_state_random, final_measurement_random = main(
        num_meas,
        rng_key,
        config,
        prefix=args.prefix,
        space=args.space,
        logging=args.plot,
        random=True,
        experiment_name=experiment_name
    )
