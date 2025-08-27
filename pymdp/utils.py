#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Utility functions

__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import jax
import jax.numpy as jnp
import numpy as np

import io
import matplotlib.pyplot as plt

from typing import (
    Any,
    List,
    Sequence,
)

Tensor = Any  # maybe jnp.ndarray, but typing seems not to be well defined for jax
Vector = List[Tensor]
Shape = Sequence[int]
ShapeList = list[Shape]


def norm_dist(dist: Tensor) -> Tensor:
    """Normalizes a Categorical probability distribution"""
    return dist / dist.sum(0)


def validate_normalization(tensor: Tensor, axis: int = 1, tensor_name: str = "tensor") -> None:
    """
    Validates that a probability tensor has normalised distributions along specified axis.
    It raises a ValueError if tensor has zero-filled distributions or unnormalised distributions
    """
    # sum along the specified axis
    sums = jnp.sum(tensor, axis=axis)
    
    # check for zero-filled distributions
    zero_filled = jnp.any(jnp.isclose(sums, 0.0))
    if zero_filled:
        raise ValueError(f"Please ensure that none of the distributions along {tensor_name}'s {axis}-th axis sum to zero...")
    
    # check for unnormalised distributions (non-zero but not summing to 1)
    not_normalised = jnp.any(~jnp.isclose(sums, 1.0))
    if not_normalised:
        raise ValueError(f"Please ensure that all distributions along {tensor_name}'s {axis}-th axis are properly normalised and sum to 1...")


def list_array_uniform(shape_list: ShapeList) -> Vector:
    """
    Creates a list of jax arrays representing uniform Categorical
    distributions with shapes given by shape_list[i]. The shapes (elements of shape_list)
    can either be tuples or lists.
    """
    arr = []
    for shape in shape_list:
        arr.append(norm_dist(jnp.ones(shape)))
    return arr


def list_array_zeros(shape_list: ShapeList) -> Vector:
    """
    Creates a list of 1-D jax arrays filled with zeros, with shapes given by shape_list[i]
    """
    arr = []
    for shape in shape_list:
        arr.append(jnp.zeros(shape))
    return arr


def list_array_scaled(shape_list: ShapeList, scale: float = 1.0) -> Vector:
    """
    Creates a list of 1-D jax arrays filled with scale, with shapes given by shape_list[i]
    """
    arr = []
    for shape in shape_list:
        arr.append(scale * jnp.ones(shape))

    return arr


def get_combination_index(x, dims):
    """
    Find the index of an array of categorical values in an array of categorical dimensions

    Parameters
    ----------
    x: ``numpy.ndarray`` or ``jax.Array`` of shape `(batch_size, act_dims)`
        ``numpy.ndarray`` or ``jax.Array`` of categorical values to be converted into combination index
    dims: ``list`` of ``int``
        ``list`` of ``int`` of categorical dimensions used for conversion

    Returns
    ----------
    index: ``np.ndarray`` or `jax.Array` of shape `(batch_size)`
        ``np.ndarray`` or `jax.Array` index of the combination
    """
    assert isinstance(x, jax.Array) or isinstance(x, np.ndarray)
    assert x.shape[-1] == len(dims)

    index = 0
    product = 1
    for i in reversed(range(len(dims))):
        index += x[..., i] * product
        product *= dims[i]
    return index


def index_to_combination(index, dims):
    """
    Convert the combination index according to an array of categorical dimensions back to an array of categorical values

    Parameters
    ----------
    index: ``np.ndarray`` or `jax.Array` of shape `(batch_size)`
        ``np.ndarray`` or `jax.Array` index of the combination
    dims: ``list`` of ``int``
        ``list`` of ``int`` of categorical dimensions used for conversion

    Returns
    ----------
    x: ``numpy.ndarray`` or ``jax.Array`` of shape `(batch_size, act_dims)`
        ``numpy.ndarray`` or ``jax.Array`` of categorical values to be converted into combination index
    """
    x = []
    for base in reversed(dims):
        x.append(index % base)
        index = index // base

    x = np.flip(np.stack(x, axis=-1), axis=-1)
    return x


def fig2img(fig):
    """
    Utility function that converts a matplotlib figure to a numpy array
    """
    with io.BytesIO() as buff:
        fig.savefig(buff, facecolor="white", format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    plt.close(fig)
    return im[:, :, :3]
