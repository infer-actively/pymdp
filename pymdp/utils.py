#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Utility functions

__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import jax
from jax import numpy as jnp, random as jr
from jax import tree_util as jtu
from jax import nn
import numpy as np
import equinox as eqx
import math
import itertools
import json

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
    """
    Creates a list of jax arrays representing random Categorical distributions with dimensions
    given by dims_per_var[i]. In the context of observations or hidden state posteriors,
    this can seen as a factorized categorical distribution over multiple variables, i.e.
    P(X1, X2, ..., Xn) = P(X1)P(X2)...P(Xn)
    """

    num_vars = len(dims_per_var)
    keys = jr.split(key, num_vars)

    return jtu.tree_map(lambda dim, i: jr.dirichlet(keys[i], alpha=jnp.ones(dim)), dims_per_var, list(range(num_vars)))

def random_A_array(key, num_obs, num_states, A_dependencies=None) -> List[jax.Array]:
    """
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
    """
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


# Infer states optimized methods


# Generate random agent specs

def A_dep_factors_dist(num_states, A_dep_len):
    '''
    Probability distribution over hidden states to sample from when randomly
    generating A_dependencies. Ensures a negative correlation between lengths
    of A dependency lists and dimensionalities (numbers of levels) of hidden
    states (e.g. the longer a list is, the more we favor lower dimensional
    hidden states).

    num_states: np.array of integers, representing dimensionalities of
                hidden states
    A_dep_len:  int, representing the desired length of an A dependency
                list

    returns:    np.array of probabilities, with the same shape as num_states
    '''
    if A_dep_len == 1:
        # in the shortest possible case, use a uniform distribution
        return np.ones_like(num_states) / len(num_states)
    else:
        # otherwise, use something similar to a softmax distribution
        tmp = np.exp(-num_states /
                     (np.max(num_states) / (A_dep_len - 1)))
        return tmp / np.sum(tmp)


def A_dep_len_dist(choices, curr_sf_dim, max_sf_dim):
    '''
    In case a hidden state has already been assigned to an A dependency list,
    getting a probability distribution over potential lengths of that list.
    Again ensuring a negative correlation, the higher the dimensionality of
    the already assigned hidden state, the higher the probabilities of choosing
    a small length of the list.

    choices:     np.array of integers, representing the possibilities for the
                 length of an A dependency list
    curr_sf_dim: int, representing the number of levels of the hidden state that
                 has already been assigned to an A dependency list.
    max_sf_dim:  int, representing the largest possible number of levels of any
                 hidden state

    returns:     np.array of probabilities, with the same shape as choices
    '''
    tmp = np.exp(-choices * curr_sf_dim / max_sf_dim)
    return tmp / np.sum(tmp)


def A_dep_len_dist_unconditional(choices):
    '''
    An unconditional exponential distribution over possible lengths of A
    dependency lists, to favor shorter lists and only occasionally sample
    longer ones.

    choices:     np.array of integers, representing the possibilities for the
                 length of an A dependency list

    returns:     np.array of probabilities, with the same shape as choices
    '''
    tmp = np.exp(-choices / 3)
    return tmp / np.sum(tmp)


def generate_agent_spec(num_factors,
                        num_modalities,
                        state_dim_limits,
                        obs_dim_limits,
                        A_dep_len_limits,
                        dim_sampling_type,
                        A_dep_len_prior='uniform'):

    '''
    The main function - generating an agent specification given some basic information.

    num_factors:       int, representing the total number of hidden state factors
    num_modalities:    int, representing the total number of observation modalities
    state_dim_limits:  (int, int), the lower and upper limits on the numbers of
                       levels for each hidden state factor
    obs_dim_limits:    (int, int), the lower and upper limits on the numbers of
                       levels for each observation modality
    A_dep_len_limits:  (int, int), the lower and upper limits on the lengths of
                       A dependency lists
    dim_sampling_type: 'uniform' | 'with_gap_uniform' | 'with_gap_skewed' |
                       'with_large_gap_uniform' | 'with_large_gap_skewed' - which
                       kind of sampling to use for dimensionalities of hidden
                       states and observation modalities (described further below)
    A_dep_len_prior:   'uniform'| 'exponential' - whether to sample lengths of A
                       dependency lists uniformly or to favor shorter ones, in cases
                       where no hidden states have already been assigned to a list
    '''
    
    if dim_sampling_type == 'uniform':

        # when using uniform dimensionality sampling, no constraints are enforced
        num_states = [np.random.randint(*state_dim_limits) for i in range(num_factors)]
        num_obs = [np.random.randint(*obs_dim_limits) for i in range(num_modalities)]

    else:

        if dim_sampling_type == 'with_gap_uniform':
            # half the hidden states / observation modalities will have their dimensions
            # sampled from the top 30% of the entire interval [a, b] of possibilities,
            # and the remaining half will be sampled from the bottom 30%
            m = 0.3
            h = 0.5
            
        elif dim_sampling_type == 'with_gap_skewed':
            # only one fifth of hidden states / observation modalities will have their
            # dimensions sampled from the top 30% of the entire interval [a, b] of
            # possibilities, and the rest will be sampled from the bottom 30%
            m = 0.3
            h = 0.2
            
        elif dim_sampling_type == 'with_large_gap_uniform':
            # half the hidden states / observation modalities will have their dimensions
            # sampled from the top 10% of the entire interval [a, b] of possibilities,
            # and the remaining half will be sampled from the bottom 10%
            m = 0.1
            h = 0.5
            
        elif dim_sampling_type == 'with_large_gap_skewed':
            # only one fifth of hidden states / observation modalities will have their
            # dimensions sampled from the top 10% of the entire interval [a, b] of
            # possibilities, and the rest will be sampled from the bottom 10%
            m = 0.1
            h = 0.2
            
        else:
            raise ValueError(f'Unsupported dim_sampling_type: {dim_sampling_type}')

        # upon establishing the dimensionality sampling type, consider the interval
        # of possible dimensionalities of hidden states and generate random values
        
        s_lower, s_upper = state_dim_limits
        s_range = s_upper - s_lower
        
        num_states = [
            np.random.randint(s_lower, int(np.ceil(s_lower + m*s_range)))
            for i in range(num_factors - int(h * num_factors))
        ] + [
            np.random.randint(int(np.floor(s_upper - m*s_range)), s_upper)
            for i in range(int(h * num_factors))
        ]

        # doing the same for observation modalities
        
        o_lower, o_upper = obs_dim_limits
        o_range = o_upper - o_lower
        
        num_obs = [
            np.random.randint(o_lower, int(np.ceil(o_lower + m*o_range)))
            for i in range(num_modalities - int(h * num_modalities))
        ] + [
            np.random.randint(int(np.floor(o_upper - m*o_range)), o_upper)
            for i in range(int(h * num_modalities))
        ]

    # ensure that each hidden state factor influences at least one observation
    # modality (i.e. is used in at least one A dependency list)
    tmp = np.random.choice(num_modalities, num_factors, replace=False)
    A_dependencies = [[] for m in range(num_modalities)]
    for sf, om in enumerate(tmp):
        A_dependencies[om].append(sf)
        
    # generating the full graph of dependencies between observation modalities
    # and hidden state factors
    for om in range(num_modalities):

        # determine the possibile choices for the length of a dependency list
        A_dep_len_min, A_dep_len_max = A_dep_len_limits
        if A_dep_len_max > num_factors:
            A_dep_len_max = num_factors
        A_dep_len_choices = np.arange(A_dep_len_min, A_dep_len_max)
        
        if len(A_dependencies[om]) == 0:

            # if no hidden states have already been assigned to an A dependency list,
            # sample its length from an a-priori distribution (either uniform or exponential)
            A_dep_len = np.random.choice(
                A_dep_len_choices,
                p=A_dep_len_dist_unconditional(A_dep_len_choices) if A_dep_len_prior == 'exponential' else None
            )

            # after sampling the length of a list, randomly choose that many hidden states
            # as its elements (without repetition), and ensure a negative correlation
            # between the hidden state dimensions and the length of the list
            A_dependencies[om] = np.random.choice(
                num_factors,
                A_dep_len,
                replace=False,
                p=A_dep_factors_dist(np.array(num_states), A_dep_len)
            ).tolist()
            
        else:

            # if a hidden state has already been assigned to an A dependency list, choose
            # its final length so that it is negatively correlated with the dimensionality
            # of the hidden state (additionally, scale the probabilities according to the
            # prior distribution of lengths to avoid awkward edge cases that would skew
            # the statistics)
            p = np.ones_like(A_dep_len_choices, dtype=np.float64)
            if A_dep_len_prior == 'exponential':
                p = A_dep_len_dist_unconditional(A_dep_len_choices)
            p *= A_dep_len_dist(A_dep_len_choices, A_dependencies[om][0], state_dim_limits[1])
            p /= np.sum(p)
            A_dep_len = np.random.choice(A_dep_len_choices, p=p)

            # fill out the rest of the A dependency lists with hidden states other than
            # the one that is already there, and ensure a negative correlation between
            # the hidden state dimensions and the length of the list
            A_dependencies[om] += np.random.choice(
                [sf for sf in range(num_factors) if sf != A_dependencies[om][0]],
                A_dep_len - 1,
                replace=False,
                p=A_dep_factors_dist(
                    np.array([ns for sf, ns in enumerate(num_states) if sf != A_dependencies[om][0]]),
                    A_dep_len
                )
            ).tolist()
    
    A_dependencies = [sorted(A_dep) for A_dep in A_dependencies]

    return num_states, num_obs, A_dependencies


def generate_agent_specs_from_parameter_sets(parameter_sets, num_agents_per_set=1, output_file='agent_specs.json'):
    '''
    Generate agent specifications from coordinated parameter sets.
    
    parameter_sets:       list of tuples, where each tuple contains
                         (num_factors, num_modalities, state_dim_upper_limit, 
                          obs_dim_upper_limit, dim_sampling_type, label)
    num_agents_per_set:   int, number of random agents to generate per parameter set
    output_file:          str, path to save the JSON file with agent specifications
    
    returns:              dict with agent specifications
    '''
    
    agent_specs = {
        'arbitrary dependencies': []
        # it might make sense to sometimes constrain the A dependencies additionally
        # e.g. '2 dependencies per modality': [],
    }

    # iterate over the coordinated parameter sets
    for num_factors, num_modalities, state_dim_upper_limit, obs_dim_upper_limit, dim_sampling_type, label in parameter_sets:
        
        # it usually makes sense to have num_factors <= num_modalities, so we skip all other
        # cases (they would require a different random generation scheme and constraints)
        if num_factors > num_modalities:
            continue

        # sample random agent specifications from each parameter set
        for _ in range(num_agents_per_set):
        
            num_states, num_obs, A_dependencies = generate_agent_spec(
                num_factors,
                num_modalities,
                (2, state_dim_upper_limit),
                (2, obs_dim_upper_limit),
                (1, 11), # allow A dependency lists of lengths 1 to 10
                dim_sampling_type,
                A_dep_len_prior='exponential' # favoring shorter dependency lists in general
            )
        
            agent_specs['arbitrary dependencies'].append({
                'num_factors': int(num_factors),
                'num_modalities': int(num_modalities),
                'num_states': [int(n) for n in num_states],
                'num_obs': [int(n) for n in num_obs],
                'A_dependencies': [[int(dep) for dep in deps] for deps in A_dependencies],
                # having descriptive metadata can be useful for querying agent specifications
                # (e.g. selecting agents with a low number of hidden states and a high number
                # of observation modalities etc.)
                'metadata': {
                    'num_factors': label,
                    'num_modalities': label,
                    'state_dim_upper_limit': label,
                    'obs_dim_upper_limit': label,
                    'dim_sampling_type': dim_sampling_type
                }
            })

    # save to file if output_file is specified
    if output_file is not None:
        with open(output_file, 'w') as f:
            json.dump(agent_specs, f)
    
    return agent_specs






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

def init_A_and_D_from_spec(num_obs, num_states, A_dependencies, A_sparsity_level=None, batch_size=1):

    A = []
    
    for i, obs_dim_i in enumerate(num_obs):
        
        lagging_dims = [num_states[f_j] for f_j in A_dependencies[i]]
        full_dimension = [obs_dim_i] + lagging_dims

        if A_sparsity_level is not None:

            # artificially controlling the amount of zeros in A matrices
        
            A_m = np.zeros([batch_size] + full_dimension)
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