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

Install Diffuse using pip:

.. code-block:: bash

   pip install diffuse

For development:

.. code-block:: bash

   git clone https://github.com/jcopo/diffuse.git
   cd diffuse
   pip install -e .

Quick Start
-----------

Here's a minimal example to get started:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from diffuse.diffusion.sde import LinearSchedule, DiffusionModel

   # Create components
   key = jax.random.PRNGKey(42)
   schedule = LinearSchedule(b_min=0.1, b_max=20.0, t0=0.0, T=1.0)

   # Generate sample data
   data = jax.random.normal(key, (1000, 2))
   print(f"Created {data.shape[0]} samples in {data.shape[1]}D")

See the :doc:`quickstart` guide for a complete tutorial.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   quickstart
   installation
   diffusion_crash_course
   diffusion_tutorial



