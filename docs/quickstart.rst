Quick Start Guide
=================

This guide introduces the core components of Diffuse and shows how to build diffusion pipelines.

Core Components
---------------

Diffuse follows a modular design with four main components that can be mixed and matched:

1. **SDE (Stochastic Differential Equation)** - Defines the forward and reverse diffusion processes
2. **Timer** - Controls time scheduling during sampling
3. **Integrator** - Numerically solves the reverse SDE
4. **Denoiser** - Orchestrates generation and handles conditional sampling

SDE: Forward and Reverse Processes
----------------------------------

The SDE defines how noise is added during training and removed during sampling. Diffusion models are described by a stochastic differential equation:

.. math::
   dx(t) = f(x, t)dt + g(t)dW(t)

This corresponds to slowly adding noise such that the noised signal can be written as:

.. math::
   x(t) = s(t)x(0) + \sigma(t)\varepsilon, \quad \varepsilon\sim\mathcal{N}(0,I)

where :math:`s(t)` and :math:`\sigma(t)` are given by:

.. math::
   s(t) = \exp\left(\int_0^t f(\xi) d\xi\right), \quad
   \sigma(t) = s(t)\left(\int_0^t \frac{g(\xi)^2}{s(\xi)^2} d\xi \right)^{1/2}

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from diffuse.diffusion.sde import LinearSchedule, SDE

   # Create noise schedule
   beta = LinearSchedule(b_min=0.02, b_max=7.0, t0=0.0, T=1.0)
   sde = SDE(beta=beta)

   # The SDE provides coefficients for the diffusion process
   t = 0.5
   coeffs = sde.coefficients(t)
   print(f"At t={t}: drift={coeffs.drift:.3f}, diffusion={coeffs.diffusion:.3f}")

Different schedules are available:

.. code-block:: python

   from diffuse.diffusion.sde import CosineSchedule

   # Alternative: cosine schedule (often better for images)
   cosine_beta = CosineSchedule(b_min=0.02, b_max=7.0, t0=0.0, T=1.0)

Timer: Scheduling Integration Steps
-----------------------------------

The timer maps discrete integration steps to continuous time :math:`t \in [0, T]`. It defines the time discretization used during the numerical integration of the reverse SDE:

.. tikz:: Time discretization strategies
   :libs: positioning

   % Uniform discretization
   \draw[thick] (0,0) -- (5,0);
   \node[below] at (0,-0.2) {$0$};
   \node[below] at (5,-0.2) {$T$};
   \foreach \i in {0,1,2,3,4,5} {
       \fill[red] (\i,0) circle (1.5pt);
   }
   \node[above] at (0,0.2) {$t_0$};
   \node[above] at (1,0.2) {} ;
   \node[above] at (2,0.2) {$t_i$};
   \node[above] at (3,0.2) {} ;
   \node[above] at (4,0.2) {} ;
   \node[above] at (5,0.2) {$t_N$};
   \node at (2.5,-0.8) {Uniform};

   % Non-uniform discretization (concentrated at end)
   \draw[thick] (7,0) -- (12,0);
   \node[below] at (7,-0.2) {$0$};
   \node[below] at (12,-0.2) {$T$};
   \foreach \x in {7, 8.5, 9.5, 10.2, 10.7, 11.1, 11.4, 11.7, 12} {
       \fill[red] (\x,0) circle (1.5pt);
   }
   \node[above] at (7,0.2) {$t_0$};
   \node[above] at (8.5,0.2) {$t_1$};
   \node[above] at (9.5,0.2) {};
   \node[above] at (10.2,0.2){$t_i$};
   \node[above] at (11.7,0.2) {};
   \node[above] at (12,0.2) {$t_N$};
   \node at (9.5,-0.8) {Dense at end};

.. code-block:: python

   from diffuse.timer import VpTimer

   # Create timer with 50 integration steps
   timer = VpTimer(eps=1e-5, tf=1.0, n_steps=50)

   # Timer maps step index to time
   step = 25
   time = timer(step)
   print(f"Step {step} corresponds to time {time:.3f}")

Integrator: Numerical Solvers
-----------------------------

Integrators solve the reverse SDE numerically to perform denoising. The reverse SDE is given by:

.. math::
   dx = [f(x,t) - g(t)^2\nabla_x\log p_t(x)]dt + g(t)d\bar{W}(t)

Different integrators offer trade-offs between speed and quality:

.. code-block:: python

   from diffuse.integrator.deterministic import EulerIntegrator, DDIMIntegrator, DPMpp2sIntegrator
   from diffuse.integrator.stochastic import EulerMaruyamaIntegrator

   # Fast but lower quality
   euler = EulerIntegrator(sde=sde, timer=timer)

   # Good balance of speed and quality
   ddim = DDIMIntegrator(sde=sde, timer=timer)

   # High quality, slower
   dpm = DPMpp2sIntegrator(sde=sde, timer=timer)

   # Stochastic (adds randomness)
   euler_maruyama = EulerMaruyamaIntegrator(sde=sde, timer=timer)

Score Function
--------------

The score function :math:`\nabla_x\log p_t(x)` predicts the gradient of the log-density of the noisy data distribution at time :math:`t`. This is the key component that enables the reverse diffusion process. In practice, this is learned by a neural network and can be loaded using the nnx library the following way:

.. code-block:: python

   graphdef, state = nnx.split(model)
   def nn_score(x, t):
      model = nnx.merge(graphdef, state)
      return model(x, t).output

Unconditional Generation
------------------------

To generate new samples :math:`x_0` from pure noise :math:`x_T`, we integrate the reverse SDE from :math:`t=T` to :math:`t=0`. Combine components to generate samples from pure noise:

.. code-block:: python

   from diffuse.denoisers.denoiser import Denoiser

   # Create denoiser pipeline
   denoiser = Denoiser(
       integrator=ddim,
       sde=sde,
       score=score_function,
       x0_shape=(data_dim,)  # Shape of data samples
   )

   # Generate samples
   key = jax.random.PRNGKey(42)
   n_samples = 100
   n_steps = 50

   final_state, history = denoiser.generate(
       key, n_steps, n_samples, keep_history=True
   )

   samples = final_state.integrator_state.position
   print(f"Generated {samples.shape[0]} samples of dimension {samples.shape[1]}")

Conditional Generation
----------------------

For conditional sampling :math:`x_0 \sim p(x_0|y)` given measurements :math:`y`, use conditional denoisers that incorporate the measurement information during the reverse process:

.. code-block:: python

   from diffuse.denoisers.cond import FPSDenoiser, TMPDenoiser
   from diffuse.base_forward_model import MeasurementState
   from diffuse.examples.gaussian_mixtures.forward_models.matrix_product import MatrixProduct

   # Create measurement
   A = jnp.array([[1.0, 0.0]])  # Observe first coordinate
   y_observed = jnp.array([1.5])
   forward_model = MatrixProduct(A, std=0.1)

   measurement_state = MeasurementState(y=y_observed, mask_history=A)

   # Create conditional denoiser
   fps_denoiser = FPSDenoiser(
       integrator=ddim,
       sde=sde,
       score=score_function,
       forward_model=forward_model,
       x0_shape=(data_dim,)
   )

   # Generate conditional samples
   cond_state, cond_history = fps_denoiser.generate(
       key, measurement_state, n_steps, n_samples, keep_history=True
   )

   conditional_samples = cond_state.integrator_state.position

Complete Pipeline Example
-------------------------

Here's a minimal working example:

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
       score=score_function, # Learned score function
       x0_shape=data_dim  # Shape of data samples
   )

   # 3. Generate samples
   key = jax.random.PRNGKey(0)
   final_state, _ = denoiser.generate(key, n_steps=50, n_samples=100)
   samples = final_state.integrator_state.position

   print(f"✓ Generated {samples.shape} samples")

Pytest
------------------
This packages comes with an extensive test suite that can be run using pytest. To visualize the results, you can add --plot and use pytest -k to select desired Denoisers and Integrators combinations:

.. code-block:: bash

   pytest --plot -k "DDIMIntegrator and DPSDenoiser"
