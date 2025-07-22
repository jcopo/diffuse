Diffusion Crash Course
======================

TLDR: Diffusion Models
----------------------

Typically, diffusion models can be described by a stochastic differential equation (SDE) of the form:

.. math::
   dx(t)=f(x, t)dt + g(t)dW(t)

It corresponds to slowly adding noise to the signal :math:`x(t)` over time such that the noised signal at time :math:`t` can be written as:

.. math::
   x(t) = s(t)x(0) + \sigma(t)\varepsilon, \quad \varepsilon\sim\mathcal{N}(0,I)

meaning that the transition kernel :math:`p(x(t)|x(0))` is Gaussian.

:math:`s(t)` describes how the original data :math:`x(0)` is attenuated or amplified over time as noise is added while :math:`\sigma(t)` controls how much noise has been injected into the system at that time step. Both are available in closed form given :math:`f(x, t)` and :math:`g(t)`.

.. math::
   \boxed{s(t) \;=\; \exp\!\left(\int_0^t f(\xi)\, d\xi\right),  \quad
   \sigma(t) \;=\; s(t)\left(\int_0^t \frac{g(\xi)^2}{s(\xi)^2}\, d\xi \right)^{1/2}}

The previous equation describes a way to add noise to a signal. It is possible to define a time reversed process that describes the denoising of the signal by a backward SDE:

.. math::
   dx=[f(x,t)−g(t)^2\nabla_x\log p_t(x)]dt+g(t)d\bar{W}(t)

Generative Models
-----------------

In order to generate new samples :math:`x_0` from pure noise :math:`x_T`, diffusion models leverage the mathematical description of the denoising process defined above. The Python class ``Denoiser`` is used to define the diffusion process starting from noise :math:`x_T` and denoising until new data :math:`x_0` is generated. It leverages the class ``Integrator`` to perform the numerical integration of the backward SDE. Possible choices of ``Integrator`` are: ``EulerIntegrator``, ``HeunIntegrator``, ``DPMpp2sIntegrator``, ``DDIMIntegrator``.

Most ``Integrator`` defined in the literature necessitate :math:`f` and :math:`g` or :math:`s` and :math:`\sigma` to be defined. These attributes are defined in a ``DiffusionModel`` class.

The time discretization used in the ``Denoiser`` is defined in the ``Timer`` class. Possible choices of ``Timer`` are: ``LinearTimer`` or ``CosineTimer``.

We also provide a ``CondDenoiser`` class to sample conditionally on a measurement :math:`y` to generate samples :math:`x_0 \sim p(x_0|y)`.