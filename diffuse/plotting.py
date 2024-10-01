import jax.numpy as jnp
import matplotlib.pyplot as plt


def metric_l2(ground_truth, state):
    thetas, weights = state.thetas, state.weights
    squared_norm = jnp.sum(jnp.square(thetas - ground_truth), axis=(1, 2, 3))
    return jnp.einsum("i, i ->", weights, squared_norm)


def plot_comparison(ground_truth, state_random, state, y_random, y, logging_path):

    thetas, weights = state
    thetas_random, weights_random = state_random[0].position, state_random[1]

    n = 20
    best_idx = jnp.argsort(weights)[-n:][::-1]
    best_idx_random = jnp.argsort(weights_random)[-n:][::-1]

    fig = plt.figure(figsize=(40, 10))
    fig.suptitle("High weight samples from optimized measurements (top) and random measurements (bottom)", fontsize=18, y=0.7, x=0.6)

    # Create grid spec for layout with reduced vertical spacing
    gs = fig.add_gridspec(4, n, hspace=.0001)  # Added hspace parameter to reduce vertical spacing

    # Add the larger subplot for the first 4 squares
    ax_large = fig.add_subplot(gs[:2, :2])
    ax_large.imshow(ground_truth, cmap="gray")
    ax_large.axis("off")
    ax_large.set_title("Ground truth ", fontsize=12)

    ax_large = fig.add_subplot(gs[:2, 2:4])
    ax_large.imshow(y, cmap="gray")
    ax_large.axis("off")
    ax_large.set_title("Measure Optimized $y$", fontsize=12)

    ax_large = fig.add_subplot(gs[:2, 4:6])
    ax_large.imshow(y_random, cmap="gray")
    ax_large.axis("off")
    ax_large.set_title("Measure Random $y$", fontsize=12)

    for idx in range(n-6):
        ax1 = fig.add_subplot(gs[0, idx+6])
        ax2 = fig.add_subplot(gs[1, idx+6])

        ax1.imshow(thetas[best_idx[idx]], cmap="gray")
        ax2.imshow(thetas_random[best_idx_random[idx]], cmap="gray")

        # set no axis labels
        ax1.axis("off")
        ax2.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust 'rect' to accommodate the suptitle

    plt.savefig(f"{logging_path}/comparison.png", bbox_inches="tight")
    plt.close()



def plotter_random(ground_truth, joint_y, design, thetas, weights, n_meas, logging_path, size):
    n = 20
    best_idx = jnp.argsort(weights)[-n:][::-1]
    worst_idx = jnp.argsort(weights)[:n]

    # Create a figure with subplots
    fig = plt.figure(figsize=(40, 10))  # Reduced height from 12 to 10
    fig.suptitle("High weight (top) and low weight (bottom) Samples", fontsize=18, y=0.67, x=0.6)
    #fig.text(0.2, 1., f'Measurement {n_meas}', va='center', rotation='vertical', fontsize=14)

    # reduce spacing between title and subplots
    plt.subplots_adjust(top=0.85)

    # Create grid spec for layout with reduced vertical spacing
    gs = fig.add_gridspec(4, n, hspace=.0001)  # Added hspace parameter to reduce vertical spacing

    # Add the larger subplot for the first 4 squares
    ax_large = fig.add_subplot(gs[:2, :2])
    ax_large.imshow(ground_truth, cmap="gray")
    ax_large.text(-2., 13., f'Measurement {n_meas}', ha='center', va='center', fontsize=14, fontweight='bold', rotation='vertical')
    ax_large.scatter(design[0], design[1], marker="o", c="red")

    ax_large.axis("off")
    ax_large.set_title("Ground Truth", fontsize=12)

    # Add another large subplot
    ax_large = fig.add_subplot(gs[:2, 2:4])
    ax_large.imshow(joint_y, cmap="gray")
    ax_large.axis("off")
    ax_large.set_title("Measure $y$", fontsize=12)
    ax_large.scatter(design[0], design[1], marker="o", c="red")
    # add a square above the image. Around the design and 5 pixels from it
    ax_large.add_patch(plt.Rectangle((design[0]-size/2, design[1]-size/2), size, size, fill=False, edgecolor='red', linewidth=2))
    # Add the remaining subplots
    for idx in range(n-4):
        ax1 = fig.add_subplot(gs[0, idx+4])
        ax2 = fig.add_subplot(gs[1, idx+4])

        ax1.imshow(thetas[best_idx[idx]], cmap="gray")
        ax2.imshow(thetas[worst_idx[idx]], cmap="gray")

        # set no axis labels
        ax1.axis("off")
        ax2.axis("off")

    #plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust 'rect' to accommodate the suptitle

    plt.savefig(f"{logging_path}/samples_{n_meas}.png", bbox_inches="tight")
    plt.close()



def sigle_plot(array):
    plt.imshow(array, cmap="gray")
    plt.axis("off")
    plt.show()


def plot_samples(thetas, cntrst_thetas, weights, weights_c, past_y, y_c):
    total_frames = len(thetas)

    # Define the fractions
    fractions = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    n = len(fractions)

    # Create a figure with subplots
    fig, axs = plt.subplots(2, n+1, figsize=(20, 6))
    fig.suptitle("Theta (top) and Contrastive Theta (bottom) Samples", fontsize=16)

    for idx, fraction in enumerate(fractions):
        # Calculate the frame index
        frame_index = int(fraction * total_frames)
        # plot past_y
        axs[0, 0].imshow(past_y, cmap="gray")
        axs[1, 0].imshow(y_c, cmap="gray")
        # Plot the image
        axs[0, 1+idx].imshow(thetas[frame_index], cmap="gray")
        axs[1, 1+idx].imshow(cntrst_thetas[frame_index], cmap="gray")

        # Set titles
        axs[0, 1+idx].set_title(f"Weight: {weights[frame_index]:.2f}", fontsize=10)
        axs[1, 1+idx].set_title(f"Weight: {weights_c[frame_index]:.2f}", fontsize=10)

        # Turn off axis labels
        axs[0, 1+idx].axis("off")
        axs[1, 1+idx].axis("off")

    # Adjust layout and display
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.1, hspace=0.3)
    plt.show()

def plot_top_samples(thetas, cntrst_thetas, weights, weights_c, past_y, y_c):
    n = 50

    best_idx = jnp.argsort(weights)[-n:][::-1]
    worst_idx = jnp.argsort(weights)[:n]

    best_idx_c = jnp.argsort(weights_c)[-n:][::-1]
    worst_idx_c = jnp.argsort(weights_c)[:n]
    # Create a figure with subplots
    fig, axs = plt.subplots(4, n, figsize=(40, 12))
    fig.suptitle("Theta (top) and Contrastive Theta (bottom) Samples", fontsize=18, y=0.67, x=0.6)

    for idx in range(n):
        axs[0, idx].imshow(thetas[best_idx[idx]], cmap="gray")
        axs[1, idx].imshow(thetas[worst_idx[idx]], cmap="gray")
        axs[2, idx].imshow(cntrst_thetas[best_idx_c[idx]], cmap="gray")
        axs[3, idx].imshow(cntrst_thetas[worst_idx_c[idx]], cmap="gray")
        # set no axis labels
        axs[0, idx].axis("off")
        axs[1, idx].axis("off")
        axs[2, idx].axis("off")
        axs[3, idx].axis("off")

    plt.tight_layout()
    #plt.subplots_adjust(top=0.85, wspace=0.1, hspace=0.3)
    plt.show()


def plot_lines(array):
    fractions = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    n = len(fractions)
    fig, axs = plt.subplots(1, n, figsize=(n*3, 3))
    fig.suptitle("array")

    for idx, fraction in enumerate(fractions):
        # Calculate the frame index
        frame_index = int(fraction * array.shape[0])
        axs[idx].imshow(array[frame_index], cmap="gray")
        axs[idx].axis("off")

        # fix colormap range

    plt.show()


def log_samples(opt_hist, ground_truth, joint_y, thetas, weights, n_meas, logging_path, size):
    n = 20
    best_idx = jnp.argsort(weights)[-n:][::-1]
    worst_idx = jnp.argsort(weights)[:n]

    # Create a figure with subplots
    fig = plt.figure(figsize=(40, 10))  # Reduced height from 12 to 10
    fig.suptitle("High weight (top) and low weight (bottom) Samples", fontsize=18, y=0.67, x=0.6)
    #fig.text(0.2, 1., f'Measurement {n_meas}', va='center', rotation='vertical', fontsize=14)

    # reduce spacing between title and subplots
    plt.subplots_adjust(top=0.85)

    # Create grid spec for layout with reduced vertical spacing
    gs = fig.add_gridspec(4, n, hspace=.0001)  # Added hspace parameter to reduce vertical spacing

    # Add the larger subplot for the first 4 squares
    ax_large = fig.add_subplot(gs[:2, :2])
    ax_large.imshow(ground_truth, cmap="gray")
    ax_large.text(-2., 13., f'Measurement {n_meas}', ha='center', va='center', fontsize=14, fontweight='bold', rotation='vertical')

    ax_large.axis("off")
    ax_large.set_title("Ground Truth", fontsize=12)
    #ax_large.scatter(opt_hist[:, 0], opt_hist[:, 1], marker="+")
    ax_large.scatter(opt_hist[-1, 0], opt_hist[-1, 1], marker="o", c="red")

    # Add another large subplot
    ax_large = fig.add_subplot(gs[:2, 2:4])
    ax_large.imshow(joint_y, cmap="gray")
    ax_large.axis("off")
    ax_large.set_title("Measure $y$", fontsize=12)
    ax_large.scatter(opt_hist[-1, 0], opt_hist[-1, 1], marker="o", c="red")
    # add a square above the image. Around the design and 5 pixels from it
    ax_large.add_patch(plt.Rectangle((opt_hist[-1, 0]-size/2, opt_hist[-1, 1]-size/2), size, size, fill=False, edgecolor='red', linewidth=2))
    # Add the remaining subplots
    for idx in range(n-4):
        ax1 = fig.add_subplot(gs[0, idx+4])
        ax2 = fig.add_subplot(gs[1, idx+4])

        ax1.imshow(thetas[best_idx[idx]], cmap="gray")
        ax2.imshow(thetas[worst_idx[idx]], cmap="gray")

        # set no axis labels
        ax1.axis("off")
        ax2.axis("off")

    #plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust 'rect' to accommodate the suptitle

    plt.savefig(f"{logging_path}/samples_{n_meas}.png", bbox_inches="tight")
    plt.close()

def plot_results(opt_hist, ground_truth, joint_y, mask_history, thetas, cntrst_thetas):
    total_frames = len(thetas)

    # Define the fractions
    fractions = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    n = len(fractions)
    # Create a figure with subplots
    fig, axs = plt.subplots(1, 3 + n, figsize=((3 + n) * 3, n + 3))
    for idx, fraction in enumerate(fractions):
        # Calculate the frame index
        frame_index = int(fraction * total_frames)

        # Plot the image
        axs[3 + idx].imshow(thetas[frame_index], cmap="gray")
        #axs[idx].set_title(f"Frame at {fraction*100}% of total")
        axs[3 + idx].axis("off")  # Turn off axis labels

    ax1, ax2, ax3 = axs[:3]
    ax1.axis("off")
    ax2.axis("off")
    ax3.axis("off")
    ax1.set_title("Ground truth")
    ax2.set_title("Mesure")
    ax3.set_title("Mask")

    ax1.scatter(opt_hist[:, 0], opt_hist[:, 1], marker="+")
    ax1.imshow(ground_truth, cmap="gray")
    #ax2.scatter(opt_hist[:, 0], opt_hist[:, 1], marker="+")
    ax2.imshow(joint_y, cmap="gray")
    ax3.imshow(mask_history, cmap="gray")
    plt.tight_layout()
    plt.show()

