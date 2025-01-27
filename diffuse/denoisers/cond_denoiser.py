from dataclasses import dataclass
from typing import Callable, Tuple, NamedTuple

import einops
import jax
import jax.experimental
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, PRNGKeyArray
from blackjax.smc.resampling import stratified
import chex

from diffuse.integrator.base import Integrator, IntegratorState
from diffuse.diffusion.sde import SDE, SDEState
from diffuse.base_forward_model import ForwardModel, MeasurementState
from diffuse.utils.plotting import sigle_plot, plot_lines

def _vmapper(fn, type):
    def _set_axes(path, value):
        # Vectorize only particles and rng_key fields
        if any(field in str(path) for field in ["position", "rng_key", "weights"]):
            return 0
        return None

    # Create tree with selective vectorization
    in_axes = jax.tree_util.tree_map_with_path(_set_axes, type)
    return jax.vmap(fn, in_axes=(in_axes, None))

def ess(log_weights: Array) -> float:
    return jnp.exp(log_ess(log_weights))


def log_ess(log_weights: Array) -> float:
    """Compute the effective sample size.

    Parameters
    ----------
    log_weights: 1D Array
        log-weights of the sample

    Returns
    -------
    log_ess: float
        The logarithm of the effective sample size

    """
    return 2 * jsp.special.logsumexp(log_weights) - jsp.special.logsumexp(
        2 * log_weights
    )


class CondDenoiserState(NamedTuple):
    integrator_state: IntegratorState
    weights: Array


@dataclass
class CondDenoiser:
    """Conditional denoiser for conditional diffusion"""

    integrator: Integrator
    # logpdf: Callable[[SDEState, Array], Array]  # x -> t -> logpdf(x, t)
    sde: SDE
    score: Callable[[Array, float], Array]  # x -> t -> score(x, t)
    forward_model: ForwardModel
    _resample: bool = False

    def init(
        self, position: Array, rng_key: PRNGKeyArray, dt: float
    ) -> CondDenoiserState:
        n_particles = position.shape[0]
        weights = jnp.log(jnp.ones(n_particles) / n_particles)
        keys = jax.random.split(rng_key, n_particles)
        integrator_state = self.integrator.init(
            position, keys, jnp.array(0.0), jnp.array(dt)
        )
        return CondDenoiserState(integrator_state, weights)

    def generate(self, rng_key: PRNGKeyArray, forward_model: ForwardModel, measurement_state: MeasurementState, design: Array, n_steps: int, n_particles: int):
        dt = self.sde.tf / n_steps

        key, subkey = jax.random.split(rng_key)
        cntrst_thetas = jax.random.normal(subkey, (n_particles, *measurement_state.y.shape))

        key, subkey = jax.random.split(key)
        state = self.init(cntrst_thetas, subkey, dt)


        def body_fun(state: CondDenoiserState, key: PRNGKeyArray):
            posterior = self.posterior_logpdf(
                key, measurement_state.y, measurement_state.mask_history
            )
            state_next = self.batch_step(key, state, posterior, measurement_state)
            return _fix_time(state_next), state_next.integrator_state.position

        keys = jax.random.split(key, n_steps)
        return jax.lax.scan(body_fun, state, keys)

    def step(
        self, state: CondDenoiserState, score: Callable[[Array, float], Array]
    ) -> CondDenoiserState:
        r"""
        sample p(\theta_t-1 | \theta_t, \y_t-1, \xi)
        """
        integrator_state, weights = state
        integrator_state_next = self.integrator(integrator_state, score)

        return CondDenoiserState(integrator_state_next, weights)

    def batch_step(
        self, rng_key: PRNGKeyArray, state: CondDenoiserState, score: Callable[[Array, float], Array], measurement_state: MeasurementState
    ) -> CondDenoiserState:
        r"""
        batch step for conditional diffusion
        """
        # vmap over position, rng_key, weights
        state_next = _vmapper(self.step, state)(state, score)
        integrator_state_next = state_next.integrator_state
        integrator_state, weights = state.integrator_state, state.weights

        mask = measurement_state.mask_history
        y = measurement_state.y
        t = integrator_state.t
        tf = self.sde.tf

        # resample if necessary
        if self._resample:
            position = integrator_state_next.position
            weights = state_next.weights
            y_noised = self.y_noiser(mask, rng_key, SDEState(y, 0), tf-t).position
            A_theta = self.forward_model.measure_from_mask(mask, position)

            # abs_A_theta = jnp.abs(A_theta[..., 0] + 1j * A_theta[..., 1])
            # # Only plot if t > 1.5
            # jax.lax.cond(
            #     t > 1.5,
            #     lambda x: jax.experimental.io_callback(plot_lines, None, x, t),
            #     lambda x: None,
            #     jnp.log(abs_A_theta)
            # )
            # #plot position
            # jax.lax.cond(
            #     t > 1.5,
            #     lambda x: jax.experimental.io_callback(plot_lines, None, x),
            #     lambda x: None,
            #     jnp.abs(position[..., 0] + 1j * position[..., 1])
            # )
            # # Plot y_noised similar to abs_A_theta
            # abs_y_noised = jnp.abs(y_noised[..., 0] + 1j * y_noised[..., 1])
            # jax.lax.cond(
            #     t > 1.5,
            #     lambda x: jax.experimental.io_callback(sigle_plot, None, x),
            #     lambda x: None,
            #     jnp.log(abs_y_noised)
            # )

            alpha_t = jnp.exp(self.sde.beta.integrate(0.0, t))
            #jax.experimental.io_callback(sigle_plot, None, y_noised)
            logsprobs = jax.scipy.stats.norm.logpdf(y_noised[..., :2], A_theta[..., :2], alpha_t)
            #jax.debug.print("logsprobs: {}", logsprobs)
            # logsprobs = self.forward_model.logprob_y(y_noised, A_theta, alpha_t)
            logsprobs1 = einops.einsum(logsprobs, measurement_state.mask_history, "t ... i, ... -> t ... i")
            #logsprobs = self.forward_model.measure_from_mask(mask, logsprobs)
            logsprobs = einops.reduce(logsprobs1, "t ... -> t ", "sum")
            # Find index of highest logprob
            max_idx = jnp.argmax(logsprobs)

            # Plot the highest logprob position and A_theta
            def plot_max_state(logprobs, logprob, pos, a_theta, y, t_val):
                import matplotlib.pyplot as plt

                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 4))

                # Plot position
                pos_mag = jnp.abs(pos[..., 0] + 1j * pos[..., 1])
                ax1.imshow(pos_mag, cmap='gray')
                ax1.set_title(f'Position (logprob={logprob:.2f}, t={t_val:.2f})')

                # Plot A_theta
                a_theta_mag = jnp.log(jnp.abs(a_theta[..., 0] + 1j * a_theta[..., 1]))
                ax2.imshow(a_theta_mag, cmap='gray')
                ax2.set_title('A_theta')

                # plot diff
                abs_y = jnp.abs(y[..., 0] + 1j * y[..., 1])
                diff = abs_y - a_theta_mag
                ax3.imshow(diff, cmap='gray')
                ax3.set_title('Difference')

                # Plot logprobs distribution
                ax4.imshow(logprobs[..., 0], cmap='gray')
                ax4.set_title('Logprobs Distribution')
                ax4.set_xlabel('Log Probability')
                ax4.set_ylabel('Count')
                ax4.legend()

                plt.tight_layout()
                plt.show()
                plt.close()

            jax.lax.cond(
                t > 1.5,
                lambda x: jax.experimental.io_callback(plot_max_state, None, *x),
                lambda x: None,
                (logsprobs1, logsprobs[max_idx], position[max_idx], A_theta[max_idx], y_noised, t)
            )

            # jax.debug.print("logsprobs: {}", logsprobs)


            position, weights = self._resampling(position, logsprobs, rng_key)
            # jax.debug.print("weights: {}", weights)

        return CondDenoiserState(integrator_state_next, weights)

    def posterior_logpdf(
        self, rng_key: PRNGKeyArray, y_meas: Array, design_mask: Array
    ):
        # will be called backward in time
        # with t = tf - t, t from 0 to tf
        def _posterior_logpdf(x, t):
            tf = self.sde.tf
            y_t = self.y_noiser(
                design_mask, rng_key, SDEState(y_meas, 0), t
            ).position
            alpha_t = jnp.exp(self.sde.beta.integrate(0.0, tf - t))
            #guidance = jax.grad(self.forward_model.logprob_y)(x, y_t, design) #/ alpha_t
            guidance = self.forward_model.grad_logprob_y(x, y_t, design_mask) / alpha_t
            return guidance + self.score(x, t)

        return _posterior_logpdf


    def pooled_posterior_logpdf(
        self,
        rng_key: PRNGKeyArray,
        y_cntrst: Array,
        y_past: Array,
        design: Array,
        mask_history: Array,
    ):
        rng_key1, rng_key2 = jax.random.split(rng_key)
        vec_noiser = jax.vmap(
            self.y_noiser, in_axes=(None, None, SDEState(0, None), None)
        )
        mask = self.forward_model.make(design)

        # will be called backward in time
        # with t = tf - t, t from 0 to tf
        def _pooled_posterior_logpdf(x, t):
            y_t = vec_noiser(
                mask, rng_key1, SDEState(y_cntrst, 0), t
            ).position
            tf = self.sde.tf
            alpha_t = jnp.exp(self.sde.beta.integrate(0.0, tf - t))
            guidance = jax.vmap(
                #jax.grad(self.forward_model.logprob_y), in_axes=(None, 0, None)
                self.forward_model.grad_logprob_y, in_axes=(None, 0, None)
            )(x, y_t, mask) / alpha_t
            past_contribution = self.posterior_logpdf(rng_key2, y_past, mask_history)
            # import pdb; pdb.set_trace()
            # jax.debug.print("guidance: {}", guidance)
            return guidance.mean(axis=0) + past_contribution(x, t)

        return _pooled_posterior_logpdf

    def y_noiser(
        self, mask: Array, key: PRNGKeyArray, state: SDEState, ts: float
    ) -> SDEState:
        r"""
        Generate y^{(t)} = \sqrt{\bar{\alpha}_t} y + \sqrt{1-\bar{\alpha}_t} A_\xi \epsilon
        """
        y, t = state

        int_b = self.sde.beta.integrate(ts, t)
        alpha, beta = jnp.exp(-0.5 * int_b), 1 - jnp.exp(-int_b)

        rndm = jax.random.normal(key, y.shape)
        res = alpha * y + jnp.sqrt(beta) * self.forward_model.measure_from_mask(
            mask, rndm
        )
        return SDEState(res, ts)

    def _resampling(
        self, position: Array, log_weights: Array, rng_key: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        """Resample particles based on weights if effective sample size is in target range"""
        _norm = jax.scipy.special.logsumexp(log_weights, axis=0)
        #weights = jnp.exp(log_weights - _norm)
        weights = jax.nn.softmax(log_weights, axis=0)

        ess_val = ess(log_weights)
        n_particles = position.shape[0]
        key_resample = jax.random.split(rng_key)[1]
        idx = stratified(key_resample, weights, n_particles)

        return jax.lax.cond(
            (ess_val < 0.4 * n_particles) & (ess_val > 0.2 * n_particles),
            #lambda x: (x[idx], _normalize_log_weights(log_weights[idx])),
            lambda x: (x, _normalize_log_weights(log_weights)),
            lambda x: (x, _normalize_log_weights(log_weights)),
            position,
        )


def _fix_time(denoiser_state: CondDenoiserState):
    # Create new integrator states with fixed time
    new_denoiser_integrator = denoiser_state.integrator_state._replace(
        t=denoiser_state.integrator_state.t[0],
        dt=denoiser_state.integrator_state.dt[0]
    )

    # Return new denoiser states with updated integrator states
    return denoiser_state._replace(integrator_state=new_denoiser_integrator)

def _normalize_log_weights(log_weights: Array) -> Array:
    return jax.nn.log_softmax(log_weights, axis=0)
