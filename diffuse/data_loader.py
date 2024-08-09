import jax
import jax.numpy as jnp
import pdb
import matplotlib.pyplot as plt
import einops
from diffuse.unet import UNet
from diffuse.score_matching import score_match_loss
from diffuse.sde import SDE, LinearSchedule
from functools import partial

data = jnp.load("dataset/mnist.npz")
key = jax.random.PRNGKey(0)
xs = data["X"]
batch_size = 2

xs = jax.random.permutation(key, xs, axis=0)
data = einops.rearrange(xs, "b h w -> b h w 1")
# plt.imshow(data[0], cmap='gray')
# plt.show()
dt = jnp.linspace(0, 2.0, 200)
dt = jnp.array([2.0 / 200] * batch_size)

beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
sde = SDE(beta)

nn_unet = UNet(dt, 64)
init_params = nn_unet.init(key, data[:batch_size], dt)

res = nn_unet.apply(init_params, data[:batch_size], dt)

loss = partial(score_match_loss, lmbda=lambda x: jnp.ones(x.shape).squeeze(), network=nn_unet)
res = loss(init_params, key, data[:batch_size], sde, 100, 2.0)
grad_res = jax.grad(loss)(init_params, key, data[:batch_size], sde, 100, 2.0)

pdb.set_trace()