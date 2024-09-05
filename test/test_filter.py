import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pdb

from diffuse.conditional import CondSDE, CondState
from diffuse.filter import generate_cond_sample
from diffuse.images import SquareMask, measure
from diffuse.sde import SDE, LinearSchedule
from diffuse.unet import UNet
from diffuse.optimizer import impl_step
from diffuse.sde import SDEState


def plotter_line(array):
    total_frames = len(array)

    # Define the fractions
    fractions = [0., 0.01, 0.05, .1, 0.3, 0.5, 0.9, .95, 1.]
    n = len(fractions)
    # Create a figure with subplots
    fig, axs = plt.subplots(1, n, figsize=(n*3, n))
    
    for idx, fraction in enumerate(fractions):
        # Calculate the frame index
        frame_index = int(fraction * total_frames)
        
        # Plot the image
        axs[idx].imshow(array[frame_index], cmap="gray")
        axs[idx].set_title(f"Frame at {fraction*100}% of total")
        axs[idx].axis('off')  # Turn off axis labels
        
    plt.tight_layout()
    plt.show()



# Load MNIST dataset
data = np.load("dataset/mnist.npz")
xs = jnp.array(data["X"])
xs = xs.reshape(xs.shape[0], xs.shape[1], xs.shape[2], 1)  # Add channel dimension
batch_size = 2
tf = 2.0
batch_size = 256
n_epochs = 3000
n_t = 256
dt = tf / n_t
ts = jnp.linspace(0, tf, n_t)

# Initialize PRNGKey
key = jax.random.PRNGKey(0)

# Define beta schedule
beta = LinearSchedule(b_min=0.02, b_max=5.0, t0=0.0, T=2.0)

# Initialize ScoreNetwork (assuming you have this defined elsewhere)
score_net = UNet(dt, 64, upsampling="pixel_shuffle")
params = score_net.init(key, jnp.ones((batch_size, 28, 28, 1)), jnp.ones((batch_size,)))
# nn_trained = jnp.load("ann_2999.npz", allow_pickle=True)
# params = nn_trained["params"].item()


# Define neural network score function
def nn_score(x, t):
    return score_net.apply(params, x, t)


# SDE setup
sde = SDE(beta=beta)


x = xs[0]
mask = SquareMask(10, x.shape)
# x = state_Ts.position[-1]
xi = jnp.array([10.0, 20.0])
cond_sde = CondSDE(beta=beta, mask=mask, tf=2.0, score=nn_score)
#y = measure(xi, x, mask)
past_y = measure(xi, x, mask)

y = jax.vmap(measure, in_axes=(None, 0, None))(xi, xs[0:40], mask)

key_noise = jax.random.split(key, n_t)
state_0 = SDEState(past_y, jnp.zeros_like(past_y))
past_y = jax.vmap(sde.path, in_axes=(0, None, 0))(key_noise, state_0, ts)



#res = generate_cond_sample(y, xi, key, 500, cond_sde, x.shape)
#res = generate_cond_sample(y, xi, key, cond_sde, x.shape)
# thetas (n_t, n_particles)
# thetas = res[1][0]
# random generate the thetas
n_particles = 100
n_contrast = 50
thetas = jax.random.normal(key, (n_t, n_particles, *x.shape))
cntrst_thetas = jax.random.normal(key, (n_t, n_contrast, *x.shape))

# Import necessary modules
from diffuse.optimizer import ImplicitState, impl_step
import optax

design = xi

# Initialize optimizer
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(design)

# Create initial state
initial_state = ImplicitState(thetas, cntrst_thetas, design, opt_state)

# Set up parameters for impl_step
key_step = jax.random.PRNGKey(42)
ts = jnp.linspace(0, tf, n_t)
dt = tf / (n_t - 1)

# Run impl_step
new_state = impl_step(initial_state, key_step, past_y, cond_sde, optimizer, ts, dt)
