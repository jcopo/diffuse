# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from __future__ import annotations

from pathlib import Path

import h5py
import jax.numpy as jnp
from jaxtyping import Array


def load_brain_slice(path: Path | str, slice_idx: int = 0) -> Array:
    """Load a single brain MRI slice from an HDF5 file.

    Args:
        path: Path to the brain HDF5 file
        slice_idx: Index of the slice to load

    Returns:
        Brain slice as a JAX array
    """
    with h5py.File(path, "r") as f:
        # Assuming the data is stored in a dataset named "data" or similar
        # Adjust based on actual file structure
        data = f["reconstruction_rss"][slice_idx]
        return jnp.array(data)


__all__ = ["load_brain_slice"]
