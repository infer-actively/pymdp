#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Utility functions

__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import jax
from jax import numpy as jnp, random as jr
from jax import tree_util as jtu
import numpy as np
import equinox as eqx

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


def norm_dist(dist: Tensor, event_dim=0) -> Tensor:
    """Normalizes a Categorical probability distribution"""
    return dist / dist.sum(event_dim, keepdims=True)


def validate_normalization(tensor: Tensor, axis: int = 1, tensor_name: str = "tensor") -> None:
    """
    Validates that a probability tensor has normalised distributions along specified axis.
    It raises a ValueError if tensor has zero-filled distributions or unnormalised distributions
    """

    # sum along the specified axis
    sums = jnp.sum(tensor, axis=axis)

    # check for zero-filled distributions
    eqx.error_if(sums, jnp.any(jnp.isclose(sums,0.0)), f"Please ensure that none of the distributions along {tensor_name}'s {axis}-th axis sum to zero...")
    
    # check for unnormalized distributions (non-zero but not summing to 1)
    eqx.error_if(sums, jnp.any(~jnp.isclose(sums,1.0)), f"Please ensure that all distributions along {tensor_name}'s {axis}-th axis are properly normalised and sum to 1...")


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

def random_single_categoricals(key, dims_per_categorical):

    num_categoricals = len(dims_per_categorical)
    keys_per_categorical = jr.split(key, num=num_categoricals)

    return jtu.tree_map(lambda ns, f: jr.dirichlet(keys_per_categorical[f], alpha=jnp.ones(ns)), dims_per_categorical, list(range(num_categoricals)))
    

def random_A_matrix(key, num_obs, num_states, A_factor_list=None):
    num_obs    = [num_obs]    if isinstance(num_obs, int)    else num_obs
    num_states = [num_states] if isinstance(num_states, int) else num_states
    num_modalities = len(num_obs)

    if A_factor_list is None:
        num_factors = len(num_states)
        A_factor_list = [list(range(num_factors))] * num_modalities

    keys = jr.split(key, num_modalities)
    A = []
    for m, n_o in enumerate(num_obs):
        batch_shape = tuple(num_states[idx] for idx in A_factor_list[m])  # trailing axes
        # Sample Dirichlet with batch_shape; returns shape = batch_shape + (n_o,)
        A_m = jr.dirichlet(keys[m], alpha=jnp.ones(n_o), shape=batch_shape)
        # Move event/observation dim to the front -> (n_o, *batch_shape)
        A.append(jnp.moveaxis(A_m, -1, 0))
    return A

def random_B_matrix(key, num_states, num_controls, B_factor_list=None, B_factor_control_list=None):

    num_states = [num_states] if isinstance(num_states, int) else num_states
    num_controls = [num_controls] if isinstance(num_controls, int) else num_controls
    num_factors = len(num_states)

    if B_factor_list is None:
        B_factor_list = [[f] for f in range(num_factors)]

    if B_factor_control_list is None:
        assert len(num_controls) == len(num_states)
        B_factor_control_list = [[f] for f in range(num_factors)]
    else:
        unique_controls = list(set(sum(B_factor_control_list, [])))        
        assert unique_controls == list(range(len(num_controls)))

    keys = jr.split(key, num_factors)
    B = []
    for f, ns in enumerate(num_states):
        lagging_shape = [ns_f for i, ns_f in enumerate(num_states) if i in B_factor_list[f]]
        control_shape = [na_f for i, na_f in enumerate(num_controls) if i in B_factor_control_list[f]]
        B_f = jr.dirichlet(keys[f], alpha=jnp.ones(ns), shape=lagging_shape+control_shape)
        B.append(jnp.moveaxis(B_f, -1, 0))
    return B

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
