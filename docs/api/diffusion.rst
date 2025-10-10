Diffusion Models
================

.. currentmodule:: diffuse.diffusion.sde

Diffusion Model Types
----------------------

.. autoclass:: SDE
   :members:
   :show-inheritance:
   :exclude-members: beta, tf

.. autoclass:: Flow
   :members:
   :show-inheritance:
   :exclude-members: tf

.. autoclass:: EDM
   :members:
   :show-inheritance:
   :exclude-members: tf

Noise Schedules
---------------

.. autoclass:: LinearSchedule
   :members:
   :show-inheritance:
   :exclude-members: b_min, b_max, t0, T

.. autoclass:: CosineSchedule
   :members:
   :show-inheritance:
   :exclude-members: b_min, b_max, t0, T, s
