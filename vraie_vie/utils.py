from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array
from typing import Callable


def slice_fourier(mri_slice):
    return jnp.fft.fftshift(jnp.fft.fft2(mri_slice))


def slice_inverse_fourier(fourier_transform):
    return jnp.real(jnp.fft.ifft2(jnp.fft.ifftshift(fourier_transform)))


@jax.custom_vjp
def _make(w: Array, s: int, shape: tuple, key: PRNGKeyArray):
    normalized_vector = w / w.sum()
    uniform_vector = jax.random.uniform(key, shape=shape, minval=0, maxval=1)
    return jnp.where(s * normalized_vector < uniform_vector, 1, 0)


def make_fwd(w: Array, s: int, shape: tuple, key: PRNGKeyArray):
    normalized_vector = w / w.sum()
    uniform_vector = jax.random.uniform(key, shape=shape, minval=0, maxval=1)
    output = jnp.where(s * normalized_vector < uniform_vector, 1, 0)
    return output, (w, s, shape, key)


def make_bwd(_, grad_output):
    return (grad_output, None, None, None)

_make.defvjp(make_fwd, make_bwd)


@dataclass
class maskFourier:
    s: int
    img_shape: tuple
    key: PRNGKeyArray
    _make_func: Callable

    def make(self, w: Array):
        _, subkey = jax.random.split(self.key)
        return self._make_func(w, self.s, subkey)

    def measure(self, w: Array, x: Array):
        mask = self.make(w)
        fourier_x = mask * slice_fourier(x[..., 0])
        zero_channel = jnp.zeros_like(fourier_x)
        return jnp.stack([fourier_x, zero_channel], axis=-1)

    def restore(self, w: Array, x: Array, measured: Array):
        # On crée le masque inverse
        inv_mask = 1 - self.make(w)

        # On calcule la transformée de Fourier de l'image
        fourier_x = slice_fourier(x[..., 0])

        # On masque les éléments observés
        masked_inv_fourier_x = inv_mask * fourier_x

        # On retrouve l'image originale
        img = slice_inverse_fourier(masked_inv_fourier_x + measured[..., 0])

        anomaly_map = x[..., 1]

        final = jnp.stack([img, anomaly_map], axis=-1)

        return final
