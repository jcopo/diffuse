Forward Models
==============

.. currentmodule:: diffuse

Predictor
---------

.. autoclass:: diffuse.predictor.Predictor
   :members: score, noise, velocity, x0
   :show-inheritance:
   :exclude-members: model, network, prediction_type

Forward Model Protocol
----------------------

ForwardModel
~~~~~~~~~~~~

.. autoclass:: diffuse.base_forward_model.ForwardModel
   :members: apply, restore
   :show-inheritance:
   :exclude-members: std

MeasurementState
~~~~~~~~~~~~~~~~

.. autoclass:: diffuse.base_forward_model.MeasurementState
   :show-inheritance:
   :exclude-members: y, mask_history
