import jax
from jaxtyping import PyTree
from functools import partial
from typing import TypeVar

T = TypeVar('T', bound=PyTree)

def pmap_reshaping(x: PyTree) -> PyTree:
    num_devices = jax.device_count()
    return jax.tree_map(
        lambda x: x.reshape((num_devices, -1, *x.shape[1:])) if len(x.shape) > 0 else x,
        x,
    )


def pmap_unshaping(x: PyTree):
    return jax.tree_map(
        lambda x: x.reshape((-1, *x.shape[2:])) if len(x.shape) > 0 else x, x
    )


def pmapper(fn, x: T, batch_size: int = None, **kwargs) -> T:
    fn = partial(fn, **kwargs)
    mapped_fn = lambda x_: jax.lax.map(f=fn, xs=x_, batch_size=batch_size)
    pmapped_fn = jax.pmap(mapped_fn, axis_name="devices", in_axes=(0,))

    pmap_x = pmap_reshaping(x)
    pmaped_y = pmapped_fn(pmap_x)

    return pmap_unshaping(pmaped_y)
