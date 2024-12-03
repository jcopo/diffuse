Structure of the repository:

- `diffuse/`: contains the source code of the Diffuse tool with the following submodules:
    - `diffusion/`: core diffusion implementations
        - `sde.py`: Implementation of the Lin stochastic differential equations and its reverse-time counterpart
        - `score_matching.py`: implementation of the score matching loss for training
    - `neural_network/`: neural network architectures
        - `unet.py`: implementation of the U-Net architecture
    - `samplopt/`: sampling and optimization
        - `conditional.py`: implementation of conditional sampling procedures
        - `inference.py`: inference utilities
        - `optimizer.py`: optimization algorithms
    - `utils/`: utility functions
        - `logger.py`: logging utilities
        - `plotting.py`: visualization tools

- `examples/`: contains example implementations and applications
    - `gaussian_mixtures/`: toy examples with Gaussian mixtures
        - `mixture.py`: mixture model implementations
        - `mixture_evolution.py`: evolution of mixture distributions
    - `mnist/`: MNIST dataset examples
        - `mnist_train.py`: training script for MNIST
        - `images.py`: image processing utilities
        - `design_mnist.py`: MNIST-specific model design
    - `mri/`: medical imaging applications
        - `wmh/`: White Matter Hyperintensities dataset
            - `create_dataset.py`: dataset creation for WMH
            - `evaluation.py`: evaluation metrics
        - `brats/`: BraTS dataset utilities
        - `mni_coregistration.py`: MNI space coregistration
        - `training.py`: training procedures for MRI data

- `test/`: contains test scripts for the Diffuse tool
    - Various test files for SDE, image processing, and mixture models
