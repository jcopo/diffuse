Quick Start Guide
=================

This guide will walk you through the basics of using Diffuse to build diffusion models.

Prerequisites
-------------

Make sure you have Python 3.8+ and the following packages installed:

.. code-block:: bash

   pip install jax jaxlib numpy

Basic Concepts
--------------

Diffuse follows a modular design with four main components:

1. **SDE (Stochastic Differential Equation)** - Defines the forward and reverse processes
2. **Timer** - Controls the time scheduling during sampling  
3. **Integrator** - Numerically solves the reverse SDE
4. **Denoiser** - Handles conditional generation and guidance

Your First Diffusion Model
---------------------------

Let's create a simple 2D diffusion model:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from diffuse.diffusion.sde import LinearSchedule, DiffusionModel

   # Initialize random key
   key = jax.random.PRNGKey(42)

   # Create noise schedule
   schedule = LinearSchedule(b_min=0.1, b_max=20.0, t0=0.0, T=1.0)

   # Generate sample data (2D Gaussian)
   n_samples = 1000
   data = jax.random.normal(key, (n_samples, 2))

   print(f"✓ Created {data.shape[0]} samples in {data.shape[1]}D")

Forward Diffusion Process
-------------------------

The forward process gradually adds noise to data:

.. code-block:: python

   from diffuse.diffusion.sde import DiffusionModel

   class SimpleDiffusion(DiffusionModel):
       def __init__(self, schedule):
           self.schedule = schedule
       
       def noise_level(self, t):
           """Compute noise level at time t"""
           return 1.0 - jnp.exp(-self.schedule.integrate(t, 0.0))

   # Create diffusion model
   diffusion = SimpleDiffusion(schedule)

   # Apply forward process at different times
   timesteps = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])

   for t in timesteps:
       noise_level = diffusion.noise_level(t)
       print(f"t={t:.2f}: noise_level={noise_level:.3f}")

.. note::

   At t=0, there's no noise (pure data). At t=1, it's pure noise.

Reverse Process (Sampling)
---------------------------

To generate new samples, we reverse the diffusion process:

.. code-block:: python

   from diffuse.integrator.deterministic import EulerIntegrator

   # Create integrator
   integrator = EulerIntegrator()

   # Simple score function (normally learned by a neural network)
   def score_fn(x, t):
       """Placeholder score function"""
       return -x / (diffusion.noise_level(t) + 1e-5)

   # Start from pure noise
   key, sample_key = jax.random.split(key)
   x_init = jax.random.normal(sample_key, (100, 2))

   print(f"✓ Starting sampling from noise: {x_init.shape}")

Complete Pipeline
-----------------

Here's how to combine all components:

.. code-block:: python

   from diffuse.timer.base import Timer

   # Custom timer for linear scheduling
   class LinearTimer(Timer):
       def __init__(self, num_steps=50):
           self.num_steps = num_steps
       
       def __call__(self, step):
           """Map step to time: 1.0 -> 0.01"""
           return 1.0 - step / self.num_steps * 0.99

   # Create complete pipeline
   timer = LinearTimer(num_steps=50)
   
   print(f"✓ Pipeline ready:")
   print(f"  - SDE: LinearSchedule")  
   print(f"  - Integrator: EulerIntegrator")
   print(f"  - Timer: {timer.num_steps} steps")
   print(f"  - Score function: Simple linear")

Advanced Features
-----------------

Diffuse supports many advanced features:

Different Integrators
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from diffuse.integrator.deterministic import DDIMIntegrator, DPMpp2sIntegrator

   # Fast deterministic sampling
   ddim = DDIMIntegrator(eta=0.0)
   
   # High-quality sampling  
   dpm = DPMpp2sIntegrator()

Conditional Generation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from diffuse.denoisers.cond.dps import DPSDenoiser

   # For inverse problems and conditional generation
   # (requires defining a forward model)

Custom Schedules
~~~~~~~~~~~~~~~~

.. code-block:: python

   from diffuse.diffusion.sde import CosineSchedule

   # Better for some applications
   cosine_schedule = CosineSchedule(b_min=0.1, b_max=20.0, t0=0.0, T=1.0)

Next Steps
----------

Now that you understand the basics:

1. **Experiment**: Try different integrators and schedules
2. **Train Models**: Replace the simple score function with a neural network

.. tip::

   For real applications, the score function should be learned by training 
   a neural network (like a U-Net) to predict the noise added at each timestep.

Common Issues
-------------

**Import Errors**
   Make sure JAX is properly installed: ``pip install jax jaxlib``

**GPU Issues**  
   For GPU support: ``pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html``

**Memory Issues**
   JAX pre-allocates GPU memory. Set: ``export XLA_PYTHON_CLIENT_PREALLOCATE=false``