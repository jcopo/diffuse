import jax
import jax.numpy as jnp


def slice_fourier(mri_slice):
    return jnp.fft.fftshift(jnp.fft.fft2(mri_slice))


def slice_inverse_fourier(fourier_transform):
    return jnp.fft.ifft2(jnp.fft.ifftshift(fourier_transform))


# w.max() / w.sum() * 10000
@jax.custom_vjp
def generate_mask(key, w, s):
    normalized_vector = w / w.sum()
    uniform_vector = jax.random.uniform(key, shape=w.shape, minval=0, maxval=1)
    return jnp.where(s * normalized_vector < uniform_vector, 1, 0)


def generate_mask_fwd(key, w, s):
    mask = generate_mask(key, w, s)
    return mask, None


def generate_mask_bwd(_, grad_output):
    return (None, grad_output, None)


generate_mask.defvjp(generate_mask_fwd, generate_mask_bwd)
