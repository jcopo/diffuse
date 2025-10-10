Diffuse: JAX-based Diffusion Models
====================================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/JAX-powered-orange
   :target: https://jax.readthedocs.io/
   :alt: JAX Powered

**Diffuse** is a research-oriented Python package for diffusion-based generative modeling
built on JAX and Flax. It provides modular, swappable components for building and
experimenting with diffusion models.

.. grid:: 2

    .. grid-item-card:: ⚡ JAX-Native
        :text-align: center

        Built from the ground up with JAX for automatic differentiation,
        JIT compilation, and GPU acceleration.

    .. grid-item-card:: 🔧 Modular Design
        :text-align: center

        Mix and match components: SDE + Timer + Integrator + Denoiser
        = Complete pipeline.

    .. grid-item-card:: 🧪 Research-Ready
        :text-align: center

        Experiment with different noise schedules, integrators,
        and conditioning methods.

    .. grid-item-card:: 🎯 Conditional Generation
        :text-align: center

        Built-in support for DPS, FPS, and other guided generation methods.

Quick Installation
------------------

For development:

.. code-block:: bash

   git clone https://github.com/jcopo/diffuse.git
   cd diffuse
   pip install -e .

Quick Start
-----------

Here's a minimal pipeline example:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from diffuse.diffusion.sde import LinearSchedule, SDE
   from diffuse.timer import VpTimer
   from diffuse.integrator.deterministic import DDIMIntegrator
   from diffuse.denoisers.denoiser import Denoiser

   # 1. Define components
   beta = LinearSchedule(b_min=0.02, b_max=7.0, t0=0.0, T=1.0)
   sde = SDE(beta=beta)
   timer = VpTimer(eps=1e-5, tf=1.0, n_steps=50)
   integrator = DDIMIntegrator(sde=sde, timer=timer)

   # 2. Create pipeline
   denoiser = Denoiser(
       integrator=integrator,
       sde=sde,
       score=score_function,  # Learned score function
       x0_shape=data_dim      # Shape of data samples
   )

   # 3. Generate samples
   key = jax.random.PRNGKey(0)
   final_state, _ = denoiser.generate(key, n_steps=50, n_samples=100)
   samples = final_state.integrator_state.position

   print(f"✓ Generated {samples.shape} samples")

See the :doc:`quickstart` guide for a complete tutorial.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   diffusion_crash_course
   diffusion_tutorial
   new_mixtures
   mnist_tutorial

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/diffusion
   api/integrators
   api/denoiser
   api/cond_denoisers
   api/neural_networks
   api/timer
   api/forward_models



