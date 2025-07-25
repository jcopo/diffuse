# Diffuse

<p align="center">
  <img src="docs/_static/logo.png" alt="Denoising Process" style="width: 36%;">
</p>

A Python package designed for research in diffusion-based generative modeling with modular components that can be easily swapped and combined for experimentation.

## Quick Start

```python
from diffuse.diffusion.sde import SDE, LinearSchedule
from diffuse.timer import VpTimer, HeunTimer
from diffuse.integrator import EulerIntegrator, DDIMIntegrator
from diffuse.denoisers.cond import DPSDenoiser

# Define SDE with noise schedule
sde = SDE(beta=LinearSchedule(b_min=0.1, b_max=20.0, T=1.0))
n_steps = 100

# Choose timer
timer = VpTimer(n_steps=n_steps, eps=0.001, tf=1.0)
# timer = HeunTimer(n_steps=n_steps, rho=7.0, sigma_min=0.002, sigma_max=1.0)

# Timer-aware integrator
#integrator = EulerIntegrator(sde=sde, timer=timer)
integrator = DDIMIntegrator(sde=sde, timer=timer)

# DPS with timer
dps = DPSDenoiser(
    sde=sde,
    score=score_fn,
    integrator=integrator,
    forward_model=forward_model
)

# Generate conditional samples
state, trajectory = dps.generate(key, measurement_state, n_steps, n_samples=10)

# Single step
next_state = dps.step(rng_key, state, measurement_state) # x_t -> x_{t-1}
```

## Features

- **Flexible Noising process**: Support for various noise schedules and diffusion processes
- **Timer-aware integration**: Advanced timing schemes for improved sampling
- **Conditional sampling**: DPS (Diffusion Posterior Sampling) and other conditional methods
- **Modular design**: Mix and match denoisers, integrators, timers, and forward models
- **Research-focused**: Built for experimentation with new diffusion techniques
- **Examples**: MNIST, Gaussian mixtures, and other applications

## Installation

```bash
pip install -e .
```

## Examples

See the `examples/` directory for implementations including:
- MNIST digit generation
- Gaussian mixture modeling
- Conditional sampling demonstrations
