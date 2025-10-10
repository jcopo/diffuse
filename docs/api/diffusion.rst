Diffusion Models
================

.. currentmodule:: diffuse.diffusion

SDE (Stochastic Differential Equations)
----------------------------------------

.. autoclass:: SDE
   :members: noise_level, signal_level, sde_coefficients, snr, score, tweedie, path
   :show-inheritance:
   :exclude-members: beta, tf

Noise Schedules
---------------

Linear Schedule
~~~~~~~~~~~~~~~

.. autoclass:: diffuse.diffusion.sde.LinearSchedule
   :members: __call__, integrate
   :show-inheritance:
   :exclude-members: b_min, b_max, t0, T

Cosine Schedule
~~~~~~~~~~~~~~~

.. autoclass:: diffuse.diffusion.sde.CosineSchedule
   :members: __call__, integrate
   :show-inheritance:
   :exclude-members: b_min, b_max, t0, T, s
