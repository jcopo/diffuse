Diffusion Crash Course
======================

Diffusion Models
----------------------

Score based generative models, Flows or denoising Diffusion Models all correpond to an iterative process over :math:`T` steps that transforms some noise :math:`x_T \sim \mathcal{N}(0,I)` into a sample :math:`x_0 \sim p(x_0)` where :math:`p(x_0)` is the distribution of the data we want to generate.

The interpolation from data to noise is obtained by corrupting the data by a gaussian convolution with noise. This is defined by the following equation:

.. math::
   :label: eq:noise_interpolation

   x_t = s(t)x_0 + \sigma(t)\varepsilon, \quad \varepsilon\sim\mathcal{N}(0,I)

where the choice of :math:`s(t)` and :math:`\sigma(t)` depends on the choosen formulation (Flows, Score based generative models, Denoising Diffusion Models ...). :math:`s(t)` describes how the original data :math:`x(0)` is attenuated or amplified over time as noise is added while :math:`\sigma(t)` controls how much noise has been injected into the system at that time step. For example, score based generative models can be described by a stochastic differential equation (SDE) of the form:

.. math::
   :label: eq:forward_sde

   dx_t=f(t)x_t dt + g(t)dW_t

with :math:`s(t)` and :math:`\sigma(t)` defined as:

.. math::
   :label: eq:s_sigma_definitions

   s(t) \;=\; \exp\!\left(\int_0^t f(\xi)\, d\xi\right),  \quad
   \sigma(t) \;=\; s(t)\left(\int_0^t \frac{g(\xi)^2}{s(\xi)^2}\, d\xi \right)^{1/2}

The interpolation :eq:`eq:noise_interpolation` defines a transition kernel :math:`p(x_t|x_0)` that is Gaussian and that allows to formulate a way to iteratively change the noise :math:`x_T` to the desired data :math:`x_0`.

In the context of score based generative models, this is defined by the backward SDE:

.. math::
   :label: eq:backward_sde

   dx_t=[f(t)x_t−g(t)^2\nabla_x\log p_t(x_t)]dt+g(t)d\bar{W}_t

Where the intractable score :math:`\nabla_x\log p_t(x)` is estimated using a neural network :math:`c_t^\theta(x_t)`. The task of generating data from noise corresponds to integrating :eq:`eq:backward_sde` backwards in time from :math:`x_T` to :math:`x_0`.

Using the formalism of stochastic processes via the Fokker-Planck equation, one can show that solving the following ODE

.. math::
   :label: eq:backward_ode

   \frac{d}{dt}x_t = f(t)x_t - \frac{1}{2} g(t)^2 \nabla_x\log p_t(x_t)

is equivalent to solving the backward SDE because both have the same time marginal distribution :math:`p(x_t)`.


Flow-based Generative Models
----------------------------

Similarly, a different parametrization is obtained by using the framework of Flows. A popular choise is :math:`\sigma(t) = t` and :math:`s(t) = 1 - t`:

.. math::
   :label: eq:flow_interpolation

   x_t = (1-t)x_0 + t\varepsilon, \quad \varepsilon\sim\mathcal{N}(0,I)

Flow-based models simplify the ODE sampling process by learning velocity field :math:`u_t(x_t)` from linear interpolation between data and noise. Simpler straight trajectories are more ameanable to ODE-based sampling because they require less discretization points to reduce discretization error. So we can increase step size and reduce the number of needed integration steps.

The flow-ODE becomes:

.. math::
   :label: eq:flow_ode

   \frac{d}{dt}x_t = u_t(x_t)

where the velocity field :math:`u_t(x_t)`  of the flow is learned using a neural network :math:`u_t^\theta(x_t)`.


Finally, the final formulation learns to predict the noise that was added :math:`D_t^\theta(x_t) \approx \varepsilon` where :math:`\varepsilon` is the noise that was added to the data at time :math:`t` :math:`x_t = s(t)x_0 + \sigma(t)\varepsilon`.

For a same :math:`s_t` and :math:`\sigma_t`, these parametrizations are equivalent and can be deduced from each other:

.. math::
   :label: eq:parametrization_equivalence

   u_t(x) = \frac{\dot{s}_t}{s_t} x - \frac{\dot{\sigma}_t \sigma_t s_t - \dot{s}_t \sigma_t^2}{s_t} \nabla \log p_t(x)

Which in turn can be written more simply with the SDE formulation :eq:`eq:forward_sde` as:

.. math::
   :label: eq:flow_sde

   u_t(x) = f(t) x - \frac{g(t)^2}{2} \nabla \log p_t(x)

In the same way, using tweedie's formula, one can link the score and the denoiser:

.. math::
   :label: eq:score_denoiser_link

   \nabla \log p_t(x) = - \frac{1}{\sigma_t} D_t^\theta(x_t)



Once a parametrization has been trained, the denoising process can be performed by different methods. Eg a learned velocity field :math:`u_t^\theta(x_t)` could be converted to a learned score :math:`c_t^\theta(x_t)` and used to perform score based sampling.

Loss functions
--------------
#TODO: add details for each loss function

.. math::
   :label: eq:flow_loss

   \mathcal{L}_{\text{flow}}(\theta) = \mathbb{E} \left[ \| u_\theta(x_t, t) - (\epsilon -x_0) \|^2 \right]

where :math:`t \sim \mathcal{T}`, :math:`x_0 \sim p(x_0)`, :math:`\epsilon \sim \mathcal{N}(0, I)` and :math:`x_t = s(t)x_0 + \sigma(t)\epsilon`.

By minimizing:

.. math::
   :label: eq:denoising_loss

   \mathcal{L}_{\text{denoise}}(\theta) = \mathbb{E} \left[ \| D_\theta(x_t, t) - \epsilon \|^2 \right]

where :math:`t \sim \mathcal{T}`, :math:`x_0 \sim p(x_0)`, and :math:`\epsilon \sim \mathcal{N}(0, I)` and :math:`x_t = s(t)x_0 + \sigma(t)\epsilon`.

Score loss:

.. math::
   :label: eq:score_loss

   \mathcal{L}_{\text{score}}(\theta) = \mathbb{E} \left[ \lambda(t) \| s_\theta(x_t, t) - \nabla_{x_t} \log p_t(x_t | x_0) \|^2 \right]

where :math:`t \sim \mathcal{T}`, :math:`x_0 \sim p(x_0)`, and :math:`x_t \sim p_t(x_t | x_0)`. Here :math:`\mathcal{T}` is the time distribution and :math:`\lambda(t)` is a weighting function often chosen to be related to the noise variance :math:`\sigma_t^2`.







Popular methods
----------------

EDM: Efficient Diffusion Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DDIM: Denoising Diffusion Implicit Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rectified Flows
~~~~~~~~~~~~~~~


Generative Models
-----------------

In order to generate new samples :math:`x_0` from pure noise :math:`x_T`, diffusion models leverage the mathematical description of the denoising process defined above. The Python class ``Denoiser`` is used to define the diffusion process starting from noise :math:`x_T` and denoising until new data :math:`x_0` is generated. It leverages the class ``Integrator`` to perform the numerical integration of the backward SDE. Possible choices of ``Integrator`` are: ``EulerIntegrator``, ``HeunIntegrator``, ``DPMpp2sIntegrator``, ``DDIMIntegrator``.

Most ``Integrator`` defined in the literature necessitate :math:`f` and :math:`g` or :math:`s` and :math:`\sigma` to be defined. These attributes are defined in a ``DiffusionModel`` class.

The time discretization used in the ``Denoiser`` is defined in the ``Timer`` class. Possible choices of ``Timer`` are: ``LinearTimer`` or ``CosineTimer``.

We also provide a ``CondDenoiser`` class to sample conditionally on a measurement :math:`y` to generate samples :math:`x_0 \sim p(x_0|y)`.