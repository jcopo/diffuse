# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
import jax
from jaxtyping import PyTree
from functools import partial
from typing import TypeVar

T = TypeVar("T", bound=PyTree)


def make_in_axes_except(x: PyTree, except_path: str) -> PyTree:
    """
    Creates an in_axes PyTree where all leaves are 0 except for the specified path which gets None.

    Args:
        x: The PyTree to create in_axes for
        except_path: The path/field name to exclude (will get None instead of 0)

    Returns:
        A PyTree with the same structure as x but with 0s and one None

    Example:
        class State(NamedTuple):
            position: Array
            step: int

        state = State(position=jnp.array([1,2,3]), step=0)
        in_axes = make_in_axes_except(state, "step")
        # Returns: State(position=0, step=None)
    """

    def _set_axes(path, _):
        if except_path in str(path):
            return None
        return 0

    return jax.tree_util.tree_map_with_path(_set_axes, x)


def pmap_reshaping(x: PyTree) -> PyTree:
    num_devices = jax.device_count()
    return jax.tree_util.tree_map(
        lambda x: x.reshape((num_devices, -1, *x.shape[1:])) if len(x.shape) > 0 else x,
        x,
    )


def pmap_unshaping(x: PyTree):
    return jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])) if len(x.shape) > 0 else x, x)


def pmapper(fn, x: T, batch_size: int = None, **kwargs) -> T:
    fn = partial(fn, **kwargs)
    def mapped_fn(x_):
        return jax.lax.map(f=fn, xs=x_, batch_size=batch_size)
    in_axes = jax.tree_util.tree_map(lambda _: 0, x)

    in_axes = (in_axes,)
    pmapped_fn = jax.pmap(mapped_fn, axis_name="devices", in_axes=in_axes)

    pmap_x = pmap_reshaping(x)
    pmaped_y = pmapped_fn(pmap_x)

    return pmap_unshaping(pmaped_y)
