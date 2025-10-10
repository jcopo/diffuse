Timer
=====

.. currentmodule:: diffuse.timer

Timer Classes
-------------

VpTimer
~~~~~~~

.. autoclass:: VpTimer
   :members: __call__
   :show-inheritance:
   :exclude-members: n_steps, eps, tf

HeunTimer
~~~~~~~~~

.. autoclass:: HeunTimer
   :members: __call__
   :show-inheritance:
   :exclude-members: n_steps, rho, sigma_min, sigma_max

DDIMTimer
~~~~~~~~~

.. autoclass:: DDIMTimer
   :members: __call__
   :show-inheritance:
   :exclude-members: n_steps, n_time_training, c_1, c_2, j0

Base Classes
------------

Timer
~~~~~

.. autoclass:: Timer
   :members: __call__
   :show-inheritance:
   :exclude-members: n_steps
