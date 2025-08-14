Diffusion Crash Course
======================

*A comprehensive introduction to diffusion models and score-based generative modeling*

**Author:** `Jacopo Iollo <https://jcopo.github.io>`_ (https://github.com/jcopo)

Diffusion Models
----------------------

Score-based generative models [Song2019]_ [Song2020]_, Flows [Liu2022]_ [Lipman2022]_ or denoising Diffusion Models [Ho2020]_ [Nichol2021]_ all correspond to an iterative process over :math:`T` steps that converts some noise :math:`x_T \sim \mathcal{N}(0,I)` into a sample :math:`x_0 \sim p(x_0)` where :math:`p(x_0)` is the distribution of the data we want to generate. To define a way to map noise into data, Generative models rely on the fact that it is easy to define the interpolation that modifies data into noise.
As we will see, from that interpolation, a neural network can be trained to define the reverse process that refines noise into data.

To define an interpolation from data to noise, the strategy is to continuisly add noise to the data :math:`x_0` until it becomes pure noise :math:`x_T`. A general interpolation [Albergo2023]_ is defined by the following equation:

.. math::
   :label: eq:noise_interpolation

   x_t = \alpha_t x_0 + \sigma_t\varepsilon, \quad \varepsilon\sim\mathcal{N}(0,I)

where the choice of :math:`\alpha_t` and :math:`\sigma_t` depends on the chosen formulation (Flows, Score-based generative models, Denoising Diffusion Models ...). The coefficient :math:`\alpha_t` describes how the original data :math:`x_0` is attenuated or amplified over time as noise is added while :math:`\sigma_t` controls how much noise has been injected into the system at that time step.

An example that fits into this formulation is score-based generative models [Song2020]_, which can be described by a stochastic differential equation (SDE) of the form:

.. math::
   :label: eq:forward_sde

   dx_t=f(t)x_t dt + g(t)dW_t

where :math:`W_t` is a standard Wiener process. The previous SDE can be solved analytically, giving :math:`\alpha_t` and :math:`\sigma_t` as:

.. math::
   :label: eq:s_sigma_definitions

   \alpha_t \;=\; \exp\!\left(\int_0^t f(\xi)\, d\xi\right),  \quad
   \sigma_t \;=\; \alpha_t\left(\int_0^t \frac{g(\xi)^2}{\alpha(\xi)^2}\, d\xi \right)^{1/2}

that defines a transition kernel :math:`p(x_t|x_0)` that is Gaussian and that allows to iteratively transform the data :math:`x_0` into noise :math:`x_T`.

As mentioned earlier, the SDE :eq:`eq:forward_sde` and the interpolation :eq:`eq:noise_interpolation` allow in turn to define a reverse process that maps noise back to data. This is the core of modern generative models.
In the context of score-based generative models, this is defined by the reverse-time SDE:

.. math::
   :label: eq:backward_sde

   dx_t=[f(t)x_t−g(t)^2\nabla_x\log p_t(x_t)]dt+g(t)d\bar{W}_t

This equation runs backwards in time, starting from noise :math:`x_T` and denoising it until it reaches data :math:`x_0`. The term :math:`\nabla_x\log p_t(x_t)` is the score function, which is intractable for most target distributions. The intractable score :math:`\nabla_x\log p_t(x)` will be estimated using a neural network :math:`s_\theta(x_t, t)`. The task of generating data from noise then corresponds to integrating :eq:`eq:backward_sde` backwards in time from :math:`x_T` to :math:`x_0`.

But it is not the only way to generate new data from noise. Using the formalism of stochastic processes via the Fokker-Planck equation, one can show that solving the following probability-flow ODE

.. math::
   :label: eq:backward_ode

   \frac{d}{dt}x_t = f(t)x_t - \frac{1}{2} g(t)^2 \nabla_x\log p_t(x_t)

is equivalent to solving the reverse-time SDE because both have the same time marginal distribution :math:`p_t(x_t)` that evolves following the Fokker-Planck equation.

Interestingly, this ODE can also be written using :math:`\alpha_t` and :math:`\sigma_t` only as:

.. math::
   :label: eq:backward_ode_alpha_sigma

   \frac{d}{dt}x_t = \frac{\dot{\alpha}_t}{\alpha_t} x_t - \left(\dot{\sigma}_t \sigma_t  - \frac{\dot{\alpha}_t \sigma_t^2}{\alpha_t}\right) \nabla \log p_t(x_t)

Flow-based Generative Models
----------------------------

A useful choice within the flows framework is the straight-line (rectified flow) path [Liu2022]_ :math:`\sigma(t) = t` and :math:`\alpha(t) = 1 - t`:

.. math::
   :label: eq:flow_interpolation

   x_t = (1-t)x_0 + t\varepsilon, \quad \varepsilon\sim\mathcal{N}(0,I)

Flow-based models [Lipman2022]_ [Albergo2023]_ simplify the ODE sampling process by learning velocity field :math:`u_t(x_t)` from linear interpolation between data and noise. Simpler straight trajectories are more amenable to ODE-based sampling because they require less discretization points to reduce discretization error. So we can increase step size and reduce the number of needed integration steps.

The flow-ODE becomes:

.. math::
   :label: eq:flow_ode

   \frac{d}{dt}x_t = u_t(x_t)

where the velocity field :math:`u_t(x_t)` of the flow is learned using a neural network :math:`u_\theta(x_t, t)`.

Denoising
-----------------

Finally, the final formulation learns to predict the noise that was added :math:`D_\theta(x_t, t) \approx \varepsilon` where :math:`\varepsilon` is the noise that was added to the data at time :math:`t`:  :math:`\varepsilon = \frac{x_t - \alpha_t x_0}{\sigma_t}`.

For a same :math:`\alpha_t` and :math:`\sigma_t`, these parametrizations are equivalent and can be deduced from each other:

.. math::
   :label: eq:parametrization_equivalence

   u_t(x) = \frac{\dot{\alpha}_t}{\alpha_t} x - \left(\dot{\sigma}_t \sigma_t  - \frac{\dot{\alpha}_t \sigma_t^2}{\alpha_t}\right) \nabla \log p_t(x)

Which in turn can be written more simply with the SDE formulation :eq:`eq:forward_sde` as:

.. math::
   :label: eq:flow_sde

   u_t(x) = f(t) x - \frac{g(t)^2}{2} \nabla \log p_t(x)

In the same way, using Tweedie's formula [Efron2011]_, one can link the score and the denoiser:

.. math::
   :label: eq:score_denoiser_link

   \nabla \log p_t(x_t) \;=\; -\,\frac{1}{\sigma_t}\, \mathbb{E}[\varepsilon \mid x_t]
   \;\approx\; -\,\frac{1}{\sigma_t}\, D_\theta(x_t, t)

The relationship between score and x0-prediction is given by:

.. math::

   \nabla \log p_t(x_t) \;=\; \frac{\alpha_t}{\sigma_t^2}\,\hat{x}_0(x_t,t) \;-\; \frac{1}{\sigma_t^2}\, x_t

where :math:`\hat{x}_0(x_t, t) = \mathbb{E}[x_0 \mid x_t]` is the predicted clean data. Using Tweedie's formula:

.. math::
   :label: eq:tweedies_formula

   \hat{x}_0(x_t, t) \;=\; \frac{1}{\alpha_t}\,\Big(x_t \;+\; \sigma_t^2 \nabla_x \log p_t(x_t)\Big)

Once a parametrization has been trained, the denoising process can be performed by different methods. Eg a learned velocity field :math:`u_\theta(x_t, t)` could be converted to a learned score :math:`s_\theta(x_t, t)` and used to perform score-based sampling.

Loss functions
--------------

Flow loss (for rectified flows):

.. math::
   :label: eq:flow_loss

   \mathcal{L}_{\text{flow}}(\theta) = \mathbb{E} \left[ w(t) \| u_\theta(x_t, t) - (\varepsilon -x_0) \|^2 \right]

where :math:`t \sim \mathcal{T}`, :math:`x_0 \sim p(x_0)`, :math:`\varepsilon \sim \mathcal{N}(0, I)`, :math:`x_t = \alpha_t x_0 + \sigma_t \varepsilon`, and :math:`w(t)` is an optional weighting function.

Denoising loss (ε-prediction):

.. math::
   :label: eq:denoising_loss

   \mathcal{L}_{\text{denoise}}(\theta) = \mathbb{E} \left[ \lambda(t) \| D_\theta(x_t, t) - \varepsilon \|^2 \right]

where :math:`t \sim \mathcal{T}`, :math:`x_0 \sim p(x_0)`, :math:`\varepsilon \sim \mathcal{N}(0, I)`, :math:`x_t = \alpha_t x_0 + \sigma_t \varepsilon`, and :math:`\lambda(t)` is a weighting function often chosen as :math:`\lambda(t) \propto \sigma_t^{-2}` to equalize SNR across time.

Score loss:

.. math::
   :label: eq:score_loss

   \mathcal{L}_{\text{score}}(\theta) = \mathbb{E} \left[ \lambda(t) \| s_\theta(x_t, t) - \nabla_{x_t} \log p_t(x_t | x_0) \|^2 \right]

where :math:`t \sim \mathcal{T}`, :math:`x_0 \sim p(x_0)`, and :math:`x_t \sim p_t(x_t | x_0)`. Here :math:`\mathcal{T}` is the time distribution and :math:`\lambda(t)` is a weighting function often chosen to be related to the noise variance :math:`\sigma_t^2`. The target score is :math:`\nabla_{x_t}\log p_t(x_t \mid x_0) \;=\; -\frac{1}{\sigma_t^2}\,(x_t - \alpha_t x_0).`

Popular methods
----------------

EDM: Efficient Diffusion Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The EDM [Karras2022]_ framework uses a specific interpolation obtained by setting :math:`\alpha_t = 1` and :math:`\sigma_t = t` in the general interpolation :eq:`eq:noise_interpolation`:

.. math::
   :label: eq:edm_interpolation

   x_t = x_0 + t\varepsilon, \quad \varepsilon\sim\mathcal{N}(0,I)

With this parameterization, the probability-flow ODE :eq:`eq:backward_ode_alpha_sigma` becomes:

.. math::
   :label: eq:edm_backward_ode

   \frac{d}{dt}x_t = - t \nabla_x\log p_t(x_t) = \frac{1}{t} \left( x_t - \mathbb{E}[x_0 \mid x_t] \right)

which is then solved using Heun's method for numerical integration.

**Neural Network Parameterization**

A key contribution of EDM is the careful parameterization of the neural network used to learn the denoiser. Following established practices for training neural networks, EDM maintains input and output signal magnitudes at unit variance and avoids large variations in gradient magnitudes on a per-sample basis.

The EDM framework introduces a specific parameterization of the neural network :math:`F_\theta` used to construct the denoiser :math:`D_\theta` for diffusion-based generative models.
The denoiser :math:`D_\theta(x; t)` is defined as:

.. math::

   D_\theta(x; t) = c_{\text{skip}}(\sigma_t) x + c_{\text{out}}(\sigma_t) F_\theta \left( c_{\text{in}}(\sigma_t) x; c_{\text{noise}}(\sigma_t) \right)

where :math:`F_\theta` represents the trainable neural network, and the preconditioning functions :math:`c_{\text{skip}}(\sigma_t)`, :math:`c_{\text{in}}(\sigma_t)`, :math:`c_{\text{out}}(\sigma_t)`, and :math:`c_{\text{noise}}(\sigma_t)` modulate the skip connection, input scaling, output scaling, and noise conditioning, respectively. These functions are derived to maintain unit variance for inputs and training targets while minimizing error amplification. Specifically, the preconditioning functions are:

.. math::

   c_{\text{skip}}(\sigma_t) = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}, \quad c_{\text{in}}(\sigma) = \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}, \quad c_{\text{out}}(\sigma) = \sigma \cdot \sigma_{\text{data}} \sqrt{\frac{1}{\sigma_t^2 + \sigma_{\text{data}}^2}}, \quad c_{\text{noise}}(\sigma_t) = \frac{1}{4} \ln(\sigma_t)

where :math:`\sigma_{\text{data}}^2` is the data distribution variance, and :math:`c_{\text{noise}}` is chosen empirically.

EDM uses the denoising loss :eq:`eq:denoising_loss` defined above with the appropriate parameterization. To ensure uniform weighting across noise levels, the loss weighting function is set as:

.. math::

   \lambda(\sigma_t) = \frac{\sigma_t^2 + \sigma_{\text{data}}^2}{(\sigma_t \cdot \sigma_{\text{data}})^2}

The noise level distribution is modeled as log-normal:

.. math::

   \ln(\sigma_t) \sim \mathcal{N}(P_{\text{mean}} = -1.2, P_{\text{std}} = 1.2)

The preconditioning keeps activations and targets near unit scale while :math:`F_\theta` focuses on predicting the small difference instead of reconstructing the whole denoised sample. Together this yields strong FIDs (e.g., CIFAR-10 ≈2, ImageNet-64 ≈1.36) without architectural changes.


DDIM: Denoising Diffusion Implicit Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DDIM [Song2020b]_ assumes the same latent noise :math:`\varepsilon` along the entire path so we can write:

.. math::
   :label: eq:ddim_interpolation

   x_t = \alpha_t x_0 + \sigma_t\varepsilon

   x_s = \alpha_s x_0 + \sigma_s\varepsilon

with the same :math:`\varepsilon` for all :math:`t` and :math:`s`.

By substituting :math:`\varepsilon` in equation for :math:`x_s` we get, for :math:`s < t`:

.. math::
   :label: eq:ddim_interpolation_substitution

   x_s = \alpha_s x_0 + \sigma_s \frac{x_t - \alpha_t x_0}{\sigma_t}

   = (\alpha_s - \alpha_t \frac{\sigma_s}{\sigma_t})x_0 + \frac{\sigma_s}{\sigma_t}x_t

The DDIM update is then deduced by approximation :math:`x_0 \approx \mathbb{E}[x_0 \mid x_t]` using Tweedie's formula :eq:`eq:tweedies_formula`:

.. math::
   :label: eq:ddim_update

   x_s = \frac{\alpha_s}{\alpha_t} x_t - (\frac{\alpha_s\sigma_t}{\alpha_t} - \sigma_s) \varepsilon_\theta(x_t, t)

By substituting :math:`\varepsilon_\theta` and using Tweedie's formula :eq:`eq:tweedies_formula` to obtain :math:`\hat{x}_0`, the same update can be written in x0-prediction form.

By taking :math:`s = t - dt` by doing a first order Taylor expansion as :math:`dt \to 0` we retrieve the probability-flow ODE :eq:`eq:backward_ode_alpha_sigma` showing that the DDIM update has the right time marginal distribution :math:`p_t(x_t)`.

Generative Models
-----------------

In order to generate new samples :math:`x_0` from pure noise :math:`x_T`, diffusion models leverage the mathematical description of the denoising process defined above. The Python class ``Denoiser`` is used to define the diffusion process starting from noise :math:`x_T` and denoising until new data :math:`x_0` is generated. It leverages the class ``Integrator`` to perform the numerical integration of the reverse-time SDE or probability-flow ODE. Possible choices of ``Integrator`` are: ``EulerIntegrator``, ``HeunIntegrator``, ``DPMpp2sIntegrator``, ``DDIMIntegrator``.

Most ``Integrator`` defined in the literature necessitate :math:`f` and :math:`g` or :math:`\alpha` and :math:`\sigma` to be defined. These attributes are defined in a ``DiffusionModel`` class.

The time discretization used in the ``Denoiser`` is defined in the ``Timer`` class. Possible choices of ``Timer`` are: ``LinearTimer`` or ``CosineTimer``.

We also provide a ``CondDenoiser`` class to sample conditionally on a measurement :math:`y` to generate samples :math:`x_0 \sim p(x_0|y)`.

References
----------

.. [Song2019] Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *Advances in Neural Information Processing Systems*, 32.

.. [Song2020] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). Score-based generative modeling through stochastic differential equations. *arXiv preprint arXiv:2011.13456*.

.. [Song2020b] Song, J., Meng, C., & Ermon, S. (2020). Denoising diffusion implicit models. *arXiv preprint arXiv:2010.02502*.

.. [Ho2020] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33, 6840-6851.

.. [Nichol2021] Nichol, A., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *International Conference on Machine Learning*, PMLR, 8162-8171.

.. [Karras2022] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. *Advances in Neural Information Processing Systems*, 35, 26565-26577.

.. [Liu2022] Liu, X., Gong, C., & Liu, Q. (2022). Flow straight and fast: Learning to generate and transfer data with rectified flow. *arXiv preprint arXiv:2209.03003*.

.. [Lipman2022] Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow matching for generative modeling. *arXiv preprint arXiv:2210.02747*.

.. [Albergo2023] Albergo, M. S., & Vanden-Eijnden, E. (2023). Stochastic Interpolants: A Unifying Framework for Flows and Diffusions. *arXiv preprint arXiv:2209.15571*.

.. [Efron2011] Efron, B. (2011). Tweedie's formula and selection bias. *Journal of the American Statistical Association*, 106(496), 1602-1614.