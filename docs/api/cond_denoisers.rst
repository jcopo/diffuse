Conditional Denoisers
=====================

DPS (Diffusion Posterior Sampling)
-----------------------------------

.. currentmodule:: diffuse.denoisers.cond.dps

.. autoclass:: DPSDenoiser
   :members:
   :show-inheritance:
   :exclude-members: integrator, model, predictor, forward_model, epsilon, zeta

DPS-GSG (Gradient Surrogate)
----------------------------

.. currentmodule:: diffuse.denoisers.cond.dps_gsg

.. autoclass:: DPSGSGDenoiser
   :members:
   :show-inheritance:
   :exclude-members: integrator, model, predictor, forward_model

FPS (Filtered Posterior Sampling)
----------------------------------

.. currentmodule:: diffuse.denoisers.cond.fps

.. autoclass:: FPSDenoiser
   :members:
   :show-inheritance:
   :exclude-members: integrator, model, predictor, forward_model, resample, ess_low, ess_high

TMP (Tweedie Moment Projection)
--------------------------------

.. currentmodule:: diffuse.denoisers.cond.tmp

.. autoclass:: TMPDenoiser
   :members:
   :show-inheritance:
   :exclude-members: integrator, model, predictor, forward_model

PiGDM (Pseudoinverse-Guided Diffusion)
--------------------------------------

.. currentmodule:: diffuse.denoisers.cond.pigdm

.. autoclass:: PiGDMDenoiser
   :members:
   :show-inheritance:
   :exclude-members: integrator, model, predictor, forward_model

DAPS (Decoupled Annealed Posterior Sampling)
--------------------------------------------

.. currentmodule:: diffuse.denoisers.cond.daps

.. autoclass:: DAPSDenoiser
   :members:
   :show-inheritance:
   :exclude-members: integrator, model, predictor, forward_model

PnPDM (Plug-and-Play Diffusion Model)
-------------------------------------

.. currentmodule:: diffuse.denoisers.cond.pnpdm

.. autoclass:: PnPDMDenoiser
   :members:
   :show-inheritance:
   :exclude-members: integrator, model, predictor, forward_model

DiffPIR (Diffusion-based Plug-and-Play Image Restoration)
---------------------------------------------------------

.. currentmodule:: diffuse.denoisers.cond.diffpir

.. autoclass:: DiffPIRDenoiser
   :members:
   :show-inheritance:
   :exclude-members: integrator, model, predictor, forward_model

EnKG (Ensemble Kalman Guidance)
-------------------------------

.. currentmodule:: diffuse.denoisers.cond.enkg

.. autoclass:: EnKGDenoiser
   :members:
   :show-inheritance:
   :exclude-members: integrator, model, predictor, forward_model
