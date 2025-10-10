from typing import Protocol, NamedTuple
from jaxtyping import Array, PRNGKeyArray


class MeasurementState(NamedTuple):
    y: Array
    mask_history: Array


class ForwardModel(Protocol):
    std: float

    def apply(self, img: Array, measurement_state: MeasurementState) -> Array: ...

    def restore(self, img: Array, measurement_state: MeasurementState) -> Array: ...
