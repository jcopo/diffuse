from typing import Protocol, NamedTuple
from jaxtyping import Array, PRNGKeyArray


class IntegratorState(NamedTuple):
    position: Array
    rng_key: PRNGKeyArray
    step: int = 0


class Integrator(Protocol):
    """
    Protocol defining the interface for numerical integrators.

    This class defines the required methods that any integrator must implement.
    Classes don't need to explicitly inherit from this Protocol as long as they
    implement the required methods with matching signatures.
    """

    def __call__(
        self, integrator_state: IntegratorState, drift: Array, diffusion: Array
    ) -> IntegratorState:
        """
        Performs one step of numerical integration.

        Args:
            integrator_state (IntegratorState): Current state of the integrator
            drift (Array): The drift term to integrate
            diffusion (Array): The diffusion term to integrate

        Returns:
            IntegratorState: Updated integrator state after integration step
        """

    def init(self, *args, **kwargs) -> IntegratorState:
        """
        Initializes the integrator with given arguments.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            IntegratorState: Initial integrator state
        """
