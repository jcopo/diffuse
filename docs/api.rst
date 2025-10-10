API Reference Overview
======================

The Diffuse API is organized into the following components:

Core Components
---------------

**Diffusion Models**
  SDE implementations and noise schedules (Linear, Cosine)

**Integrators**
  Numerical integrators for solving SDEs (Euler, DDIM, Heun, DPM++, etc.)

**Denoisers**
  Unconditional denoising algorithms for sampling

**Conditional Denoisers**
  Conditional sampling methods (DPS, FPS, TMP)

Supporting Components
---------------------

**Neural Networks**
  UNet, VAE, and network building blocks

**Timer**
  Time scheduling schemes for sampling

**Forward Models**
  Forward model protocols and predictors

**Utilities**
  Logging, mapping, and helper functions
