Structure of the repository:

- `diffuse/`: contains the source code of the Diffuse tool with the following files:
    - `mixtures.py`: contains the implementation of the mixture models for the test of toy diffusion
    - `sde.py`: Implementation of the Lin stochastic differential equations and its reverse-time counterpart used for sampling
    - `image.py`: tools for image processing and masking for MNIST
    - `score_matching.py`: implementation of the score matching loss used to train Diffusion Model
    - `unet.py`: implementation of the U-Net architecture used for the Diffusion Model
    - `mnist_train.py`: script to train the Diffusion Model on MNIST 
    - `conditional.py`: implementation of the conditional sampling procedure

- `test`: contains the test scripts for the Diffuse tool with a test on a toy example with Gaussian Mixtures