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


def norm_dist(dist: Tensor) -> Tensor:
    """Normalizes a Categorical probability distribution"""
    return dist / dist.sum(0)

def list_array_norm_dist(dist_list: List[Tensor]) -> List[Tensor]:
    """Normalizes a list of Categorical probability distributions"""
    return jtu.tree_map(lambda dist: norm_dist(dist), dist_list)

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

def random_factorized_categorical(key, dims_per_var: Sequence[int]) -> List[jax.Array]:
    """"
    Creates a list of jax arrays representing random Categorical distributions with dimensions
    given by dims_per_var[i]. In the context of observations or hidden state posteriors,
    this can seen as a factorized categorical distribution over multiple variables, i.e.
    P(X1, X2, ..., Xn) = P(X1)P(X2)...P(Xn)
    """

    num_vars = len(dims_per_var)
    keys = jr.split(key, num_vars)

    return jtu.tree_map(lambda dim, i: jr.dirichlet(keys[i], alpha=jnp.ones(dim)), dims_per_var, list(range(num_vars)))

def random_A_array(key, num_obs, num_states, A_dependencies=None) -> List[jax.Array]:
    """"
    Creates a list of jax arrays representing observation likelihoods (A tensors or A matrices) with shapes
    determined by num_obs and num_states, and factorized according to A_factor_list.

    The storage of each A tensor in a separate list element represents the factorized assumption over modalities, namely:
    P(o^{1:M} | s^{1:F}) = P(o^1 | s^{factors for o^1}) * ... * P(o^M | s^{factors for o^M})
    """
    num_obs    = [num_obs]    if isinstance(num_obs, int)    else num_obs
    num_states = [num_states] if isinstance(num_states, int) else num_states
    num_modalities = len(num_obs)

    if A_dependencies is None:
        num_factors = len(num_states)
        A_dependencies = [list(range(num_factors))] * num_modalities

    keys = jr.split(key, num_modalities)
    A = []
    for m, n_o in enumerate(num_obs):
        lagging_dimensions = tuple(num_states[idx] for idx in A_dependencies[m])  # trailing axes
        # Sample Dirichlet with batch_shape=lagging_dimensions; returns shape = lagging_dimensions + (n_o,)
        A_m = jr.dirichlet(keys[m], alpha=jnp.ones(n_o), shape=lagging_dimensions)
        # Move event/observation dim to the front -> (n_o, *lagging_dimensions)
        A.append(jnp.moveaxis(A_m, -1, 0))
    return A

def random_B_array(key, num_states, num_controls, B_dependencies=None, B_action_dependencies=None):
    """"
    Creates a list of jax arrays representing state transition likelihoods (B tensors or B matrices) with shapes
    determined by num_states and num_controls, and factorized according to B_dependencies and B_action_dependencies.
    
    The storage of each B tensor in a separate list element represents the factorized assumption over hidden state factors, namely:
    P(s^{1:F} | s^{1:F}, a^{1:A}) = \prod_{f=1}^{F} P(s^f | s^{factors for s^f}, a^{controls for s^f})
    """

    num_states = [num_states] if isinstance(num_states, int) else num_states
    num_controls = [num_controls] if isinstance(num_controls, int) else num_controls
    num_factors = len(num_states)

    if B_dependencies is None:
        B_dependencies = [[f] for f in range(num_factors)]

    if B_action_dependencies is None:
        assert len(num_controls) == len(num_states)
        B_action_dependencies = [[f] for f in range(num_factors)]
    else:
        unique_controls = list(set(sum(B_action_dependencies, [])))        
        assert unique_controls == list(range(len(num_controls)))

    keys = jr.split(key, num_factors)
    B = []
    for f, ns in enumerate(num_states):
        lagging_shape = [ns_f for i, ns_f in enumerate(num_states) if i in B_dependencies[f]]
        control_shape = [na_f for i, na_f in enumerate(num_controls) if i in B_action_dependencies[f]]
        # Sample Dirichlet with batch_shape=lagging_shape+control_shape; returns shape = lagging_shape + control_shape + (ns,)
        B_f = jr.dirichlet(keys[f], alpha=jnp.ones(ns), shape=lagging_shape+control_shape)
        # Move event/state dim to the front -> (ns, *lagging_shape, *control_shape)
        B.append(jnp.moveaxis(B_f, -1, 0))
    return B


def create_controllable_B(num_states, num_controls) -> Vector:
    """
    JAX equivalent of ``pymdp.legacy.utils.construct_controllable_B``.
    
    Generates a fully controllable transition likelihood array, where each 
    action (control state) corresponds to a move to the n-th state from any 
    other state, for each control factor
    """

    num_states = [num_states] if isinstance(num_states, int) else list(num_states)
    num_controls = [num_controls] if isinstance(num_controls, int) else list(num_controls)

    if len(num_states) != len(num_controls):
        raise ValueError("`num_states` and `num_controls` must have the same length")

    B = []
    for ns, nc in zip(num_states, num_controls):
        # Deterministic tensor with shape (ns, 1, nc) encoding next-state selection per action
        state_axis = jnp.arange(ns, dtype=jnp.int32).reshape(ns, 1, 1)
        action_axis = jnp.arange(nc, dtype=jnp.int32).reshape(1, 1, nc)
        deterministic = (state_axis == action_axis).astype(jnp.float32)
        B.append(jnp.broadcast_to(deterministic, (ns, ns, nc)))

    return B

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

def make_A_full(A_reduced: List[jax.Array], A_dependencies: List[List[int]], num_obs: List[int], num_states: List[int]) -> List[jax.Array]:
    """ 
    Given a reduced A matrix, `A_reduced`, and a list of dependencies between hidden state factors and observation modalities, `A_dependencies`,
    return a full A matrix, `A_full`, where `A_full[m]` is the full A matrix for modality `m`. This means all redundant conditional independencies
    between observation modalities `m` and all hidden state factors (i.e. `range(len(num_states))`) are represented as lagging dimensions in `A_full`.
    """
    A_shape_list = [ [no] + num_states for no in num_obs]
    A_full = list_array_zeros(A_shape_list) # initialize the full likelihood tensor (ALL modalities might depend on ALL factors)
    all_factors = range(len(num_states)) # indices of all hidden state factors
    for m, _ in enumerate(A_full):

        # Step 1. Extract the list of the factors that modality `m` does NOT depend on
        non_dependent_factors = list(set(all_factors) - set(A_dependencies[m])) 

        # Step 2. broadcast or tile the reduced A matrix (`A_reduced`) along the dimensions of corresponding to `non_dependent_factors`, to give it the full shape of `(num_obs[m], *num_states)`
        expanded_dims = [num_obs[m]] + [1 if f in non_dependent_factors else ns for (f, ns) in enumerate(num_states)]
        tile_dims = [1] + [ns if f in non_dependent_factors else 1 for (f, ns) in enumerate(num_states)]
        A_full[m] = jnp.tile(A_reduced[m].reshape(expanded_dims), tile_dims)
    
    return A_full


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
