Integrators
===========

.. currentmodule:: diffuse.integrator

Deterministic Integrators
--------------------------

Euler Integrator
~~~~~~~~~~~~~~~~

.. autoclass:: EulerIntegrator
   :members: init, __call__
   :show-inheritance:

Heun Integrator
~~~~~~~~~~~~~~~

.. autoclass:: HeunIntegrator
   :members: init, __call__
   :show-inheritance:

DPM++ 2S Integrator
~~~~~~~~~~~~~~~~~~~

.. autoclass:: DPMpp2sIntegrator
   :members: init, __call__
   :show-inheritance:

DDIM Integrator
~~~~~~~~~~~~~~~

.. autoclass:: DDIMIntegrator
   :members: init, __call__
   :show-inheritance:

Stochastic Integrators
-----------------------

Euler-Maruyama Integrator
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: EulerMaruyamaIntegrator
   :members: init, __call__
   :show-inheritance:

Base Classes
------------

IntegratorState
~~~~~~~~~~~~~~~

.. autoclass:: IntegratorState
   :show-inheritance:

Integrator
~~~~~~~~~~

.. autoclass:: Integrator
   :members: init, __call__
   :show-inheritance:
   :exclude-members: model, timer

   Base class for all numerical integrators.

ChurnedIntegrator
~~~~~~~~~~~~~~~~~

.. autoclass:: ChurnedIntegrator
   :members: init, __call__
   :show-inheritance:
   :exclude-members: model, timer, stochastic_churn_rate, churn_min, churn_max, noise_inflation_factor

   Base class for integrators with optional stochastic churning.
