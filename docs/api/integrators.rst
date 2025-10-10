Integrators
===========

.. currentmodule:: diffuse.integrator

Deterministic Integrators
--------------------------

.. autoclass:: EulerIntegrator
   :members:
   :show-inheritance:
   :exclude-members: model, timer, stochastic_churn_rate, churn_min, churn_max, noise_inflation_factor

.. autoclass:: HeunIntegrator
   :members:
   :show-inheritance:
   :exclude-members: model, timer, stochastic_churn_rate, churn_min, churn_max, noise_inflation_factor

.. autoclass:: DPMpp2sIntegrator
   :members:
   :show-inheritance:
   :exclude-members: model, timer, stochastic_churn_rate, churn_min, churn_max, noise_inflation_factor

.. autoclass:: DDIMIntegrator
   :members:
   :show-inheritance:
   :exclude-members: model, timer, stochastic_churn_rate, churn_min, churn_max, noise_inflation_factor

Stochastic Integrators
-----------------------

.. autoclass:: EulerMaruyamaIntegrator
   :members:
   :show-inheritance:
   :exclude-members: model, timer, stochastic_churn_rate, churn_min, churn_max, noise_inflation_factor

Base Classes
------------

.. autoclass:: IntegratorState

.. autoclass:: Integrator
   :members:
   :show-inheritance:
   :exclude-members: model, timer

.. autoclass:: ChurnedIntegrator
   :members:
   :show-inheritance:
   :exclude-members: model, timer, stochastic_churn_rate, churn_min, churn_max, noise_inflation_factor
