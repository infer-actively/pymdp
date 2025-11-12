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
import math
import itertools

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


# Infer states optimized methods

def apply_padding_batched(xs):
    '''xs: list of arrays'''
    
    max_rank = max(x.ndim for x in xs)
    max_dims = [sum(x.shape[0] for x in xs)] + [max(x.shape[i] for x in xs if i < x.ndim) for i in range(1, max_rank)]
    
    xs_padded = jnp.zeros(max_dims, dtype=xs[0].dtype)

    i = 0
    for x in xs:
        slices = (slice(i, i+x.shape[0]),)
        for j in range(1, max_rank):
            if j < x.ndim:
                slices += (slice(0, x.shape[j]),)
            else:
                slices += (0,)
        xs_padded = xs_padded.at[slices].set(x)
        i += x.shape[0]
        
    return xs_padded

def get_sample_obs(num_obs, batch_size=1):
    obs = [np.random.randint(0, obs_dim, (batch_size, 1)) for obs_dim in num_obs]
    return [jnp.array(o) for o in obs]

def init_agent_from_spec(num_obs, num_states, A_dependencies, A_sparsity_level=None, batch_size=1):

    A = []
    
    for i, obs_dim_i in enumerate(num_obs):
        
        lagging_dims = [num_states[f_j] for f_j in A_dependencies[i]]
        full_dimension = [obs_dim_i] + lagging_dims

        if A_sparsity_level is not None:

            # artificially controlling the amount of zeros in A matrices
        
            A_m = np.zeros([batch_size] + full_dimension)
            total_elements = np.prod(lagging_dims)
            nonzero_per_distribution = max(1, int((1.0 - A_sparsity_level) * obs_dim_i))

            for batch in range(batch_size):
        
                it = np.nditer(np.empty(lagging_dims), flags=['multi_index'])
                for _ in it:
                    idx = it.multi_index
                    nonzero_indices = np.random.choice(obs_dim_i, size=nonzero_per_distribution, replace=False)
                    values = np.random.rand(nonzero_per_distribution)
                    values /= values.sum()
                    for i, val in zip(nonzero_indices, values):
                        A_m[(batch, i) + idx] = val

        else:
            A_m = [np.random.rand(*full_dimension) for batch in range(batch_size)]
            A_m = [a / a.sum(axis=0, keepdims=True) for a in A_m]
            A_m = np.concatenate([a[np.newaxis, ...] for a in A_m], axis=0)
        
        A.append(A_m)
    
    D = [(1.0 / ns) * np.ones((batch_size, ns)) for ns in num_states]
    
    A = [jnp.array(a) for a in A]
    D = [jnp.array(d) for d in D]
    
    # A = [jnp.expand_dims(a, axis=0) for a in A]
    # D = [jnp.expand_dims(d, axis=0) for d in D]

    return A, D    

# Block diagonal approach functions
def build_block_diag_A(A_list: List[jnp.ndarray]):
    """Return block‑diagonal matrix and per‑modality state shapes."""
    A_flat = [a.reshape(a.shape[0], -1, a.shape[-1]) for a in A_list]
    state_shapes = tuple(tuple(int(d) for d in a.shape[1:-1]) for a in A_list)  # hashable tuples
    
    sizes = tuple(math.prod(shape) for shape in state_shapes)   # rows per modality
    cuts  = tuple(itertools.accumulate(sizes))[:-1]
    
    row_off = jnp.cumsum(jnp.array([0] + [a.shape[1] for a in A_flat]))
    col_off = jnp.cumsum(jnp.array([0] + [a.shape[-1] for a in A_flat]))
    bs, rows, cols = A_list[0].shape[0], int(row_off[-1]), int(col_off[-1])

    A_big = jnp.zeros((bs, rows, cols), dtype=A_list[0].dtype)
    for m, a in enumerate(A_flat):
        r0, r1 = int(row_off[m]),   int(row_off[m + 1])
        c0, c1 = int(col_off[m]),   int(col_off[m + 1])
        A_big = A_big.at[:, r0:r1, c0:c1].set(a)
    return A_big, state_shapes, cuts

# Preprocessing functions for block diagonal approach
def preprocess_A_for_block_diag(A):
    """Preprocess A matrices for block diagonal approach."""
    A_big, state_shapes, cuts = build_block_diag_A(A)
    return A_big, state_shapes, cuts

def prepare_obs_for_block_diag(obs, num_obs):
    """Prepare observation vectors for block diagonal approach."""
    # Convert to one-hot if needed
    o_vec = [nn.one_hot(o, num_obs[m]) for m, o in enumerate(obs)]
    obs_tmp = jtu.tree_map(lambda x: x[-1], o_vec)
    return obs_tmp

def concatenate_observations_block_diag(obs_list: List[jnp.ndarray]):
    """Step 0: Concatenate observations for block diagonal approach."""
    return jnp.concatenate(obs_list, axis=1)

def apply_A_end2end_padding_batched(A):
    
    max_rank = max(a.ndim for a in A)
    max_dims = [len(A), A[0].shape[0], max(a.shape[1] for a in A)] + [max(max(a.shape[2:]) for a in A)]*(max_rank - 2)
    
    A_padded = jnp.zeros(max_dims, dtype=A[0].dtype)

    for i, a in enumerate(A):
        slices = (i, slice(0, a.shape[0]))
        for j in range(1, max_rank):
            if j < a.ndim:
                slices += (slice(0, a.shape[j]),)
            else:
                slices += (0,)
        A_padded = A_padded.at[slices].set(a)
        
    return A_padded

def apply_obs_end2end_padding_batched(obs, max_obs_dim):
    
    full_shape = [len(obs), obs[0].shape[0], max_obs_dim]
    
    obs_padded = jnp.zeros(full_shape, dtype=obs[0].dtype)

    for i, o in enumerate(obs):
        obs_padded = obs_padded.at[i, :, slice(0, o.shape[1])].set(o)
        
    return obs_padded