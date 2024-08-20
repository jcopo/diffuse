import jax
import jax.numpy as jnp
import pdb
import matplotlib.pyplot as plt
import einops
from diffuse.unet import UNet
from diffuse.score_matching import score_match_loss
from diffuse.sde import SDE, LinearSchedule
from functools import partial
import numpy as np
import optax
from tqdm import tqdm

data = jnp.load("dataset/mnist.npz")
key = jax.random.PRNGKey(0)
xs = data["X"]
batch_size = 256
n_epochs = 3500
n_t = 256
tf = 2.0
dt = tf / n_t

xs = jax.random.permutation(key, xs, axis=0)
data = einops.rearrange(xs, "b h w -> b h w 1")
shape_sample = data.shape[1:]
# plt.imshow(data[0], cmap='gray')
# plt.show()
# dt = jnp.linspace(0, 2.0, n_t)
# dt = jnp.array([2.0 / n_t] * batch_size)

beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)
sde = SDE(beta)

nn_unet = UNet(dt, 64, upsampling="pixel_shuffle")
# init_params = nn_unet.init(key, data[:batch_size], dt)
init_params = nn_unet.init(
    key, jnp.ones((batch_size, *shape_sample)), jnp.ones((batch_size,))
)


def weight_fun(t):
    int_b = sde.beta.integrate(t, 0).squeeze()
    return 1 - jnp.exp(-int_b)


loss = partial(score_match_loss, lmbda=jax.vmap(weight_fun), network=nn_unet)

nsteps_per_epoch = data.shape[0] // batch_size
until_steps = int(0.95 * n_epochs) * nsteps_per_epoch
lr = 2e-4
schedule = optax.cosine_decay_schedule(
    init_value=lr, decay_steps=until_steps, alpha=1e-2
)
optimizer = optax.adam(learning_rate=schedule)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optimizer)
ema_kernel = optax.ema(0.99)


@jax.jit
def step(key, params, opt_state, ema_state, data):
    val_loss, g = jax.value_and_grad(loss)(params, key, data, sde, n_t, tf)
    updates, opt_state = optimizer.update(g, opt_state)
    params = optax.apply_updates(params, updates)
    ema_params, ema_state = ema_kernel.update(params, ema_state)
    return params, opt_state, ema_state, val_loss, ema_params


params = init_params
opt_state = optimizer.init(params)
ema_state = ema_kernel.init(params)

for epoch in range(n_epochs):
    subkey, key = jax.random.split(key)
    # data = jax.random.permutation(subkey, data, axis=0)
    idx = jax.random.choice(
        subkey, data.shape[0], (nsteps_per_epoch, batch_size), replace=False
    )
    p_bar = tqdm(range(nsteps_per_epoch))
    list_loss = []
    for i in p_bar:
        subkey, key = jax.random.split(key)
        params, opt_state, ema_state, val_loss, ema_params = step(
            subkey, params, opt_state, ema_state, data[idx[i]]
        )
        p_bar.set_postfix({"loss=": val_loss})
        list_loss.append(val_loss)
    print(f"epoch=: {epoch} | mean_loss=: {sum(list_loss) / nsteps_per_epoch}")

    if (epoch + 1) % 500 == 0:
        np.savez(f"ann_{epoch}.npz", params=params, ema_params=ema_params)

np.savez(f"ann_end.npz", params=params, ema_params=ema_params)
