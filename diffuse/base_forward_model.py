from typing import Protocol, NamedTuple
from jaxtyping import Array, PRNGKeyArray


class MeasurementState(NamedTuple):
    y: Array
    mask_history: Array


class ForwardModel(Protocol):
    def measure(self, design: Array, theta: Array) -> Array: ...

    def restore(self, design: Array, y: Array, new_measurement: Array) -> Array: ...

    def make(self, design: Array) -> Array: ...

    def init_design(self, rng_key: PRNGKeyArray) -> Array: ...

    def init_measurement(self, rng_key: PRNGKeyArray) -> MeasurementState: ...

    def update_measurement(
        self, measurement_state: MeasurementState, new_measurement: Array, design: Array
    ) -> MeasurementState: ...
