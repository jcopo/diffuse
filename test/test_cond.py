import jax
import jax.numpy as jnp
from diffuse.sde import SDE, LinearSchedule
from diffuse.images import SquareMask, measure
from diffuse.conditional import CondSDE, generate_cond_sample
from diffuse.unet import UNet
import numpy as np

# Load MNIST dataset
data = np.load("dataset/mnist.npz")
xs = jnp.array(data["X"])
# xs = jnp.array(data["X"])[:, 9:19, 9:19]  # Select only 10x10 pixels from the center
xs = xs.reshape(xs.shape[0], xs.shape[1], xs.shape[2], 1)  # Add channel dimension
batch_size = 2
tf = 2.0
batch_size = 256
n_epochs = 3000
n_t = 256
dt = tf / n_t

# Initialize PRNGKey
key = jax.random.PRNGKey(0)

# Define beta schedule
beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)

# Initialize ScoreNetwork (assuming you have this defined elsewhere)
score_net = UNet(dt, 64, upsampling="pixel_shuffle")
params = score_net.init(
    key, jnp.ones((batch_size, xs.shape[1], xs.shape[2], 1)), jnp.ones((batch_size,))
)


# Define neural network score function
def nn_score(x, t):
    return score_net.apply(params, x, t)


# SDE setup
sde = SDE(beta=beta)


x = xs[0]
mask = SquareMask(10, x.shape)
# x = state_Ts.position[-1]
xi = jnp.array([10.0, 20.0])
y = measure(xi, x, mask)
cond_sde = CondSDE(beta=beta, mask=mask, tf=2.0, score=nn_score)

generate_cond_sample(y, xi, key, 100, cond_sde, x.shape)
