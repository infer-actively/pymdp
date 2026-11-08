#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions for model construction, data shaping, and sampling."""

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

from jaxtyping import Array

from typing import (
    Any,
    Sequence,
)
from dataclasses import dataclass
from heapq import heappush, heappop
from random import choices as _rnd_choices


def to_nested_tuple(x):
    """Recursively convert nested lists/tuples to nested tuples, for hashable-by-value storage."""
    if isinstance(x, (list, tuple)):
        return tuple(to_nested_tuple(v) for v in x)
    return x


def norm_dist(dist: Array) -> Array:
    """Normalizes a Categorical probability distribution.

    Parameters
    ----------
    dist: Array
        Unnormalized Categorical distribution.

    Returns
    -------
    Array
        Normalized distribution.
    """
    return dist / dist.sum(0)

def list_array_norm_dist(dist_list: list[Array]) -> list[Array]:
    """Normalizes a list of Categorical probability distributions.

    Parameters
    ----------
    dist_list: list[Array]
        List of unnormalized Categorical distributions.

    Returns
    -------
    list[Array]
        List of normalized distributions.
    """
    return jtu.tree_map(lambda dist: norm_dist(dist), dist_list)


def resolve_a_dependencies(
    num_factors: int,
    num_modalities: int,
    A_dependencies: list[list[int]] | None = None,
) -> list[list[int]]:
    """Return modality-to-factor dependencies, filling in the fully connected default."""
    if A_dependencies is None:
        return [list(range(num_factors)) for _ in range(num_modalities)]
    return A_dependencies


def resolve_b_dependencies(
    num_factors: int,
    B_dependencies: list[list[int]] | None = None,
) -> list[list[int]]:
    """Return factor-to-factor transition dependencies, filling in local defaults."""
    if B_dependencies is None:
        return [[f] for f in range(num_factors)]
    return B_dependencies


def resolve_b_action_dependencies(
    num_factors: int,
    B_action_dependencies: list[list[int]] | None = None,
) -> list[list[int]]:
    """Return control-factor dependencies, filling in the per-factor default."""
    if B_action_dependencies is None:
        return [[f] for f in range(num_factors)]
    return B_action_dependencies

def validate_normalization(tensor: Array, axis: int = 1, tensor_name: str = "tensor") -> None:
    """
    Validate that a probability tensor has normalized distributions along a given axis.

    Parameters
    ----------
    tensor: Array
        Tensor to validate.
    axis: int, default=1
        Axis that should contain normalized probability distributions.
    tensor_name: str, default="tensor"
        Human-readable name used in error messages.

    Returns
    -------
    None
        In eager mode, raises `ValueError` if any distribution along the given
        axis sums to zero or is not normalized. Under JAX tracing/JIT, invalid
        distributions are signaled via `eqx.error_if`.
    """

    sums = jnp.sum(tensor, axis=axis)

    # When traced under JIT we still need a JAX-compatible runtime assertion.
    # Equinox handles that case well for valid tensors, and the jittable Agent
    # constructor tests rely on this path. The eager path below avoids the
    # callback-based runtime error mechanism for non-jitted CUDA validation.
    if isinstance(sums, jax.core.Tracer):
        sums = eqx.error_if(
            sums,
            jnp.any(jnp.isclose(sums, 0.0)),
            f"Please ensure that none of the distributions along {tensor_name}'s "
            f"{axis}-th axis sum to zero...",
        )
        sums = eqx.error_if(
            sums,
            jnp.any(~jnp.isclose(sums, 1.0)),
            f"Please ensure that all distributions along {tensor_name}'s {axis}-th "
            "axis are properly normalised and sum to 1...",
        )
        return

    # This validation runs during eager model construction outside JIT-compiled
    # kernels. Using Python-side checks avoids Equinox's runtime callback path,
    # which requires a local CPU device and fails under `JAX_PLATFORMS=cuda`.
    has_zero = bool(jnp.any(jnp.isclose(sums, 0.0)))
    if has_zero:
        raise ValueError(
            f"Please ensure that none of the distributions along {tensor_name}'s "
            f"{axis}-th axis sum to zero..."
        )

    is_normalized = bool(jnp.all(jnp.isclose(sums, 1.0)))
    if not is_normalized:
        raise ValueError(
            f"Please ensure that all distributions along {tensor_name}'s {axis}-th "
            "axis are properly normalised and sum to 1..."
        )

def random_factorized_categorical(key: Array, dims_per_var: Sequence[int]) -> list[Array]:
    """
    Create random factorized Categorical distributions.

    Parameters
    ----------
    key: Array
        PRNG key for sampling.
    dims_per_var: Sequence[int]
        Number of levels per variable.

    Returns
    -------
    list[Array]
        A list of sampled categorical vectors.
    """

    num_vars = len(dims_per_var)
    keys = jr.split(key, num_vars)

    return jtu.tree_map(lambda dim, i: jr.dirichlet(keys[i], alpha=jnp.ones(dim)), dims_per_var, list(range(num_vars)))

def random_A_array(
    key: Array,
    num_obs: int | Sequence[int],
    num_states: int | Sequence[int],
    A_dependencies: list[list[int]] | None = None,
) -> list[Array]:
    """Create random observation likelihood tensors.

    Parameters
    ----------
    key: Array
        PRNG key for sampling.
    num_obs: int | Sequence[int]
        Number of discrete observations.
    num_states: int | Sequence[int]
        Number of hidden states per factor.
    A_dependencies: list[list[int]] | None
        Optional dependency structure per modality.

    Returns
    -------
    list[Array]
        Randomized A tensors.
    """
    num_obs    = [num_obs]    if isinstance(num_obs, int)    else num_obs
    num_states = [num_states] if isinstance(num_states, int) else num_states
    num_modalities = len(num_obs)

    A_dependencies = resolve_a_dependencies(len(num_states), num_modalities, A_dependencies)

    keys = jr.split(key, num_modalities)
    A = []
    for m, n_o in enumerate(num_obs):
        lagging_dimensions = tuple(num_states[idx] for idx in A_dependencies[m])  # trailing axes
        # Sample Dirichlet with batch_shape=lagging_dimensions; returns shape = lagging_dimensions + (n_o,)
        A_m = jr.dirichlet(keys[m], alpha=jnp.ones(n_o), shape=lagging_dimensions)
        # Move event/observation dim to the front -> (n_o, *lagging_dimensions)
        A.append(jnp.moveaxis(A_m, -1, 0))
    return A

def random_B_array(
    key: Array,
    num_states: int | Sequence[int],
    num_controls: int | Sequence[int],
    B_dependencies: list[list[int]] | None = None,
    B_action_dependencies: list[list[int]] | None = None,
) -> list[Array]:
    """Create random transition tensors.

    Parameters
    ----------
    key: Array
        PRNG key for sampling.
    num_states: int | Sequence[int]
        Number of states per hidden-state factor.
    num_controls: int | Sequence[int]
        Number of controls per factor.
    B_dependencies: list[list[int]] | None
        Optional state-factor dependency structure per factor.
    B_action_dependencies: list[list[int]] | None
        Optional action-factor dependency structure per hidden-state factor.

    Returns
    -------
    list[Array]
        Randomized B tensors.
    """

    num_states = [num_states] if isinstance(num_states, int) else num_states
    num_controls = [num_controls] if isinstance(num_controls, int) else num_controls
    num_factors = len(num_states)

    B_dependencies = resolve_b_dependencies(num_factors, B_dependencies)

    if B_action_dependencies is None:
        assert len(num_controls) == len(num_states)
    else:
        unique_controls = set(sum(B_action_dependencies, []))
        assert unique_controls == set(range(len(num_controls)))
    B_action_dependencies = resolve_b_action_dependencies(num_factors, B_action_dependencies)

    keys = jr.split(key, num_factors)
    B = []
    for f, ns in enumerate(num_states):
        lagging_shape = [num_states[d] for d in B_dependencies[f]]
        control_shape = [num_controls[d] for d in B_action_dependencies[f]]
        # Sample Dirichlet with batch_shape=lagging_shape+control_shape; returns shape = lagging_shape + control_shape + (ns,)
        B_f = jr.dirichlet(keys[f], alpha=jnp.ones(ns), shape=lagging_shape+control_shape)
        # Move event/state dim to the front -> (ns, *lagging_shape, *control_shape)
        B.append(jnp.moveaxis(B_f, -1, 0))
    return B


def create_controllable_B(
    num_states: int | Sequence[int], num_controls: int | Sequence[int]
) -> list[Array]:
    """Create deterministic fully-controllable transition matrices.

    Parameters
    ----------
    num_states: int | Sequence[int]
        Number of hidden states per factor.
    num_controls: int | Sequence[int]
        Number of controls per factor.

    Returns
    -------
    list[Array]
        A list of fully controllable transition tensors.
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

def list_array_uniform(shape_list: Sequence[Sequence[int]]) -> list[Array]:
    """
    Creates uniform Categorical arrays for each requested shape.

    Parameters
    ----------
    shape_list: Sequence[Sequence[int]]
        Target tensor shapes.

    Returns
    -------
    list[Array]
        Uniform distributions for each shape.
    """
    arr = []
    for shape in shape_list:
        arr.append(norm_dist(jnp.ones(shape)))
    return arr


def list_array_zeros(shape_list: Sequence[Sequence[int]]) -> list[Array]:
    """Create zero arrays for each requested shape.

    Parameters
    ----------
    shape_list: Sequence[Sequence[int]]
        Target tensor shapes.

    Returns
    -------
    list[Array]
        Zero-filled arrays for each shape.
    """
    arr = []
    for shape in shape_list:
        arr.append(jnp.zeros(shape))
    return arr


def list_array_scaled(
    shape_list: Sequence[Sequence[int]], scale: float = 1.0
) -> list[Array]:
    """Create arrays filled with a constant scale value.

    Parameters
    ----------
    shape_list: Sequence[Sequence[int]]
        Target tensor shapes.
    scale: float, default=1.0
        Fill value.

    Returns
    -------
    list[Array]
        Arrays filled with `scale`.
    """
    arr = []
    for shape in shape_list:
        arr.append(scale * jnp.ones(shape))

    return arr


def get_combination_index(x: jax.Array | np.ndarray, dims: Sequence[int]) -> jax.Array | np.ndarray:
    """
    Find the index of an array of categorical values in an array of categorical dimensions

    Parameters
    ----------
    x: jax.Array | np.ndarray of shape (batch_size, act_dims)
        Categorical values to be converted into combination index.
    dims: Sequence[int]
        Categorical dimensions used for conversion.

    Returns
    ----------
    index: jax.Array | np.ndarray of shape (batch_size)
        Index of the combination.
    """
    assert isinstance(x, jax.Array) or isinstance(x, np.ndarray)
    assert x.shape[-1] == len(dims)

    index = 0
    product = 1
    for i in reversed(range(len(dims))):
        index += x[..., i] * product
        product *= dims[i]
    return index


def index_to_combination(index: jax.Array | np.ndarray, dims: Sequence[int]) -> jax.Array:
    """
    Convert the combination index according to an array of categorical dimensions back to an array of categorical values

    Parameters
    ----------
    index: np.ndarray or jax.Array of shape (batch_size)
        Index of the combination.
    dims: Sequence[int]
        Categorical dimensions used for conversion.

    Returns
    ----------
    x: jax.Array | np.ndarray of shape (batch_size, act_dims)
        Categorical values corresponding to each factor.
    """
    x = []
    for base in reversed(dims):
        x.append(index % base)
        index = index // base

    x = jnp.flip(jnp.stack(x, axis=-1), axis=-1)
    return x

def make_A_full(
    A_reduced: list[Array],
    A_dependencies: list[list[int]],
    num_obs: list[int],
    num_states: list[int],
) -> list[Array]:
    """Lift reduced likelihood tensors into full modality tensors.

    Parameters
    ----------
    A_reduced: list[Array]
        Reduced likelihood tensors.
    A_dependencies: list[list[int]]
        Dependency structure between modalities and state factors.
    num_obs: list[int]
        Observation dimensions.
    num_states: list[int]
        State dimensions.

    Returns
    -------
    list[Array]
        Full likelihood tensors with redundant factor dimensions restored.
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


def fig2img(fig: Any) -> np.ndarray:
    """
    Utility conversion from Matplotlib figure to RGB image array.

    Parameters
    ----------
    fig: Any
        Matplotlib figure object.

    Returns
    -------
    np.ndarray
        RGB image array extracted from the figure.
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

def A_dep_factors_dist(num_states: Sequence[int], A_dep_len: int) -> Array:
    """Probability over hidden-state factors when building A dependencies.

    Parameters
    ----------
    num_states: Sequence[int]
        Hidden-state dimensionalities.
    A_dep_len: int
        Candidate dependency-list length.

    Returns
    -------
    Array
        Probability vector over factors, same shape as `num_states`.
    """
    num_states_arr = jnp.asarray(num_states, dtype=jnp.float32)
    if A_dep_len == 1:
        # in the shortest possible case, use a uniform distribution
        return jnp.ones_like(num_states_arr) / num_states_arr.shape[0]

    # otherwise, use something similar to a softmax distribution
    tmp = jnp.exp(-num_states_arr / (jnp.max(num_states_arr) / (A_dep_len - 1)))
    return tmp / jnp.sum(tmp)


def A_dep_len_dist(choices: Array, curr_sf_dim: int, max_sf_dim: int) -> Array:
    """Distribution over A-dependency list lengths.

    Parameters
    ----------
    choices: Array
        Candidate dependency-list lengths.
    curr_sf_dim: int
        Dimensionality of a currently assigned hidden state.
    max_sf_dim: int
        Maximum hidden-state dimensionality.

    Returns
    -------
    Array
        Normalized weights over `choices`.
    """
    choices = jnp.asarray(choices, dtype=jnp.float32)
    tmp = jnp.exp(-choices * curr_sf_dim / max_sf_dim)
    return tmp / jnp.sum(tmp)


def A_dep_len_dist_unconditional(choices: Array) -> Array:
    """Unconditional prior for A-dependency list lengths.

    Parameters
    ----------
    choices: Array
        Candidate dependency-list lengths.

    Returns
    -------
    Array
        Normalized weights over `choices`.
    """
    choices = jnp.asarray(choices, dtype=jnp.float32)
    tmp = jnp.exp(-choices / 3)
    return tmp / jnp.sum(tmp)


def generate_agent_spec(
        num_factors: int,
        num_modalities: int,
        state_dim_limits: tuple[int, int],
        obs_dim_limits: tuple[int, int],
        A_dep_len_limits: tuple[int, int],
        dim_sampling_type: str,
        A_dep_len_prior: str = 'uniform',
        key: Array | None = None,
) -> tuple[list[int], list[int], list[list[int]]]:
    """Generate a random agent specification from high-level constraints.

    Parameters
    ----------
    num_factors: int
        Total number of hidden state factors.
    num_modalities: int
        Total number of observation modalities.
    state_dim_limits: tuple[int, int]
        Inclusive lower/upper bounds for state dimensions.
    obs_dim_limits: tuple[int, int]
        Inclusive lower/upper bounds for observation dimensions.
    A_dep_len_limits: tuple[int, int]
        Lower/upper bounds for A-dependency list length.
    dim_sampling_type: str
        Sampling strategy used for dimensionalities.
    A_dep_len_prior: str, default='uniform'
        Prior for A-dependency length.
    key: Array | None
        Optional PRNG key.

    Returns
    -------
    tuple[list[int], list[int], list[list[int]]]
        `(num_states, num_obs, A_dependencies)`.
    """

    if key is None:
        key = jr.PRNGKey(0)
    
    if dim_sampling_type == 'uniform':

        # when using uniform dimensionality sampling, no constraints are enforced
        keys = jr.split(key, num_factors + num_modalities + 1)
        key = keys[0]
        num_states = [int(jr.randint(keys[i+1], (), *state_dim_limits)) for i in range(num_factors)]
        num_obs = [int(jr.randint(keys[num_factors+i+1], (), *obs_dim_limits)) for i in range(num_modalities)]

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

        # Split keys for sampling num_states and num_obs
        keys = jr.split(key, num_factors + num_modalities + 1)
        key = keys[0]

        num_states = [
            int(jr.randint(keys[i+1], (), s_lower, int(np.ceil(s_lower + m*s_range))))
            for i in range(num_factors - int(h * num_factors))
        ] + [
            int(jr.randint(keys[num_factors - int(h * num_factors) + i + 1], (),
                          int(np.floor(s_upper - m*s_range)), s_upper))
            for i in range(int(h * num_factors))
        ]

        # doing the same for observation modalities
        o_lower, o_upper = obs_dim_limits
        o_range = o_upper - o_lower
        num_obs = [
            int(jr.randint(keys[num_factors + i + 1], (), o_lower, int(np.ceil(o_lower + m*o_range))))
            for i in range(num_modalities - int(h * num_modalities))
        ] + [
            int(jr.randint(keys[num_factors + num_modalities - int(h * num_modalities) + i + 1], (),
                          int(np.floor(o_upper - m*o_range)), o_upper))
            for i in range(int(h * num_modalities))
        ]

    # ensure that each hidden state factor influences at least one observation
    # modality (i.e. is used in at least one A dependency list)
    key, subkey = jr.split(key)
    tmp = jr.choice(subkey, num_modalities, shape=(num_factors,), replace=False)
    A_dependencies = [[] for m in range(num_modalities)]
    for sf, om in enumerate(tmp):
        A_dependencies[om].append(int(sf))
        
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
            key, subkey = jr.split(key)
            probs = A_dep_len_dist_unconditional(A_dep_len_choices) if A_dep_len_prior == 'exponential' else None
            A_dep_len = int(jr.choice(subkey, A_dep_len_choices, p=probs))

            # after sampling the length of a list, randomly choose that many hidden states
            # as its elements (without repetition), and ensure a negative correlation
            # between the hidden state dimensions and the length of the list
            key, subkey = jr.split(key)
            A_dependencies[om] = jr.choice(
                subkey,
                num_factors,
                shape=(A_dep_len,),
                replace=False,
                p=A_dep_factors_dist(num_states, A_dep_len)
            ).tolist()

        else:

            # if a hidden state has already been assigned to an A dependency list, choose
            # its final length so that it is negatively correlated with the dimensionality
            # of the hidden state (additionally, scale the probabilities according to the
            # prior distribution of lengths to avoid awkward edge cases that would skew
            # the statistics)
            p = jnp.ones_like(A_dep_len_choices, dtype=jnp.float32)
            if A_dep_len_prior == 'exponential':
                p = A_dep_len_dist_unconditional(A_dep_len_choices)
            p *= A_dep_len_dist(A_dep_len_choices, A_dependencies[om][0], state_dim_limits[1])
            p /= jnp.sum(p)
            key, subkey = jr.split(key)
            A_dep_len = int(jr.choice(subkey, A_dep_len_choices, p=p))

            # fill out the rest of the A dependency lists with hidden states other than
            # the one that is already there, and ensure a negative correlation between
            # the hidden state dimensions and the length of the list
            remaining_factors = jnp.array([sf for sf in range(num_factors) if sf != A_dependencies[om][0]])
            key, subkey = jr.split(key)
            additional_deps = jr.choice(
                subkey,
                remaining_factors,
                shape=(A_dep_len - 1,),
                replace=False,
                p=A_dep_factors_dist(
                    [ns for sf, ns in enumerate(num_states) if sf != A_dependencies[om][0]],
                    A_dep_len
                )
            ).tolist()
            A_dependencies[om] += additional_deps
    
    A_dependencies = [sorted(A_dep) for A_dep in A_dependencies]

    return num_states, num_obs, A_dependencies


def generate_agent_specs_from_parameter_sets(
        parameter_sets: Sequence[tuple[int, int, int, int, str, str]],
        num_agents_per_set: int = 1,
        max_A_dependency_list_size: int = 10,
        output_file: str | None = 'agent_specs.json',
        seed: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Generate multiple agent specs from parameter grids.

    Parameters
    ----------
    parameter_sets: Sequence[tuple[int, int, int, int, str, str]]
        Tuples of
        `(num_factors, num_modalities, state_dim_upper_limit, obs_dim_upper_limit, dim_sampling_type, label)`.
    num_agents_per_set: int, default=1
        Number of samples to draw for each parameter set.
    max_A_dependency_list_size: int, default=10
        Maximum allowed A-dependency list size.
    output_file: str | None, default='agent_specs.json'
        Optional path to save generated specs.
    seed: int | None, default=None
        RNG seed.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Mapping to generated specification records.
    """

    # Create JAX PRNGKey with seed for reproducibility
    key = jr.PRNGKey(seed if seed is not None else 0)

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

            key, subkey = jr.split(key)
            num_states, num_obs, A_dependencies = generate_agent_spec(
                num_factors,
                num_modalities,
                (2, state_dim_upper_limit),
                (2, obs_dim_upper_limit),
                (1, max_A_dependency_list_size+1), # allow A dependency lists of lengths 1 to max_A_dependency_list_size
                dim_sampling_type,
                A_dep_len_prior='exponential', # favoring shorter dependency lists in general
                key=subkey,
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






def apply_padding_batched(xs: list[Array]) -> Array:
    """
    Pad and concatenate variable-size arrays along a new batch axis.

    Parameters
    ----------
    xs: list[Array]
        Arrays to concatenate.

    Returns
    -------
    Array
        Padded batch tensor.
    """
    
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

def get_sample_obs(num_obs: Sequence[int], batch_size: int = 1) -> list[Array]:
    """Generate random observations for each modality.

    Parameters
    ----------
    num_obs: Sequence[int]
        Outcome counts per modality.
    batch_size: int, default=1
        Number of samples per modality.

    Returns
    -------
    list[Array]
        Random observations of shape `(batch_size, 1)` per modality.
    """
    obs = [np.random.randint(0, obs_dim, (batch_size, 1)) for obs_dim in num_obs]
    return [jnp.array(o) for o in obs]

def init_A_and_D_from_spec(
    num_obs: Sequence[int],
    num_states: Sequence[int],
    A_dependencies: list[list[int]],
    A_sparsity_level: float | None = None,
    batch_size: int = 1,
) -> tuple[list[Array], list[Array]]:
    """Create initial A and D tensors from explicit model metadata.

    Parameters
    ----------
    num_obs: Sequence[int]
        Observation cardinalities.
    num_states: Sequence[int]
        Hidden-state cardinalities.
    A_dependencies: list[list[int]]
        Modality-to-state dependencies.
    A_sparsity_level: float | None, default=None
        Optional sparsity level when constructing A.
    batch_size: int, default=1
        Number of sampled model instances.

    Returns
    -------
    tuple[list[Array], list[Array]]
        Initialized A and D arrays.
    """

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
def build_block_diag_A(
    A_list: list[Array],
) -> tuple[Array, tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Build a block-diagonal representation from modality-wise likelihood tensors.

    Parameters
    ----------
    A_list: list[Array]
        List of likelihood tensors.

    Returns
    -------
    tuple[Array, tuple[tuple[int, ...], ...], tuple[int, ...]]
        `(A_big, state_shapes, cuts)`.
    """
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
def preprocess_A_for_block_diag(
    A: list[Array],
) -> tuple[Array, tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Preprocess A matrices for block-diagonal likelihood evaluation.

    Parameters
    ----------
    A: list[Array]
        Likelihood tensors.

    Returns
    -------
    tuple[Array, tuple[tuple[int, ...], ...], tuple[int, ...]]
        Block-diagonal representation and auxiliary metadata.
    """
    A_big, state_shapes, cuts = build_block_diag_A(A)
    return A_big, state_shapes, cuts

def prepare_obs_for_block_diag(obs: list[Array], num_obs: Sequence[int]) -> list[Array]:
    """Prepare observation vectors for block-diagonal calculations.

    Parameters
    ----------
    obs: list[Array]
        Raw observation tensors.
    num_obs: Sequence[int]
        Observation cardinalities.

    Returns
    -------
    list[Array]
        One-hot encoded observation arrays prepared for block-diagonal use.
    """
    # Convert to one-hot if needed
    o_vec = [nn.one_hot(o, num_obs[m]) for m, o in enumerate(obs)]
    obs_tmp = jtu.tree_map(lambda x: x[-1], o_vec)
    return obs_tmp

def concatenate_observations_block_diag(obs_list: list[Array]) -> Array:
    """Concatenate observation vectors for block-diagonal processing.

    Concatenation happens along the trailing (observation) axis, so this
    supports both single-timestep inputs of shape `(batch, obs_dim)` (e.g. FPI)
    and sequence inputs of shape `(batch, T, obs_dim)` (e.g. VMP/MMP).

    Parameters
    ----------
    obs_list: list[Array]
        One-hot encoded observations per modality, sharing all leading axes.

    Returns
    -------
    Array
        Concatenated observation tensor.
    """
    return jnp.concatenate(obs_list, axis=-1)

def apply_A_end2end_padding_batched(A: list[Array]) -> Array:
    """Pad A tensors for end-to-end batched processing.

    Parameters
    ----------
    A: list[Array]
        A tensors per modality.

    Returns
    -------
    Array
        Batched padded A tensor.
    """
    
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

def apply_obs_end2end_padding_batched(obs: list[Array], max_obs_dim: int) -> Array:
    """Pad observations for end-to-end batched processing.

    Only the trailing (observation) axis is padded to `max_obs_dim`; all leading
    axes are taken from `obs[0]`. This supports both single-timestep inputs of
    shape `(batch, obs_dim)` (e.g. FPI) and sequence inputs of shape
    `(batch, T, obs_dim)` (e.g. VMP/MMP).

    Parameters
    ----------
    obs: list[Array]
        Observation tensors, one per modality, sharing all leading axes.
    max_obs_dim: int
        Dimensionality to pad the trailing observation axis to.

    Returns
    -------
    Array
        Batched padded observation tensor of shape
        `(num_modalities,) + obs[0].shape[:-1] + (max_obs_dim,)`.
    """

    full_shape = (len(obs),) + tuple(obs[0].shape[:-1]) + (max_obs_dim,)

    obs_padded = jnp.zeros(full_shape, dtype=obs[0].dtype)

    for i, o in enumerate(obs):
        obs_padded = obs_padded.at[(i, ...) + (slice(0, o.shape[-1]),)].set(o)

    return obs_padded


# Sequence-based (VMP/MMP) optimized-inference helpers

def pad_individual_Bs(B: list[Array]) -> list[Array]:
    """Pad each transition tensor's state axes to the largest state dimension.

    Assumes trivial `B_dependencies` (each factor depends only on itself), so
    every `B[f]` has shape `(batch, num_states[f], num_states[f], num_controls[f])`.

    Parameters
    ----------
    B: list[Array]
        Transition tensors per factor.

    Returns
    -------
    list[Array]
        Per-factor tensors zero-padded to
        `(batch, max_state_dim, max_state_dim, num_controls[f])`.
    """

    maxdim = max(b.shape[1] for b in B)
    B_padded = [jnp.zeros((b.shape[0], maxdim, maxdim, b.shape[-1]), dtype=B[0].dtype) for b in B]

    for i in range(len(B)):
        nb, ns, ns, na = B[i].shape
        B_padded[i] = B_padded[i].at[:nb, :ns, :ns, :na].set(B[i])

    return B_padded


def preprocess_B(B: list[Array], past_actions: Array) -> list[Array]:
    """Condition transition tensors on a sequence of past actions.

    This is the core path of
    :func:`pymdp.inference._condition_transitions_on_actions` specialized to
    all-valid, non-negative action indices (no `invalid_action_mode` handling
    and no sparse-`B` densification). It is used by the optimized VMP routines,
    which assume exactly this setting.

    Parameters
    ----------
    B: list[Array]
        Transition tensors per factor, with the policy/action axis rightmost.
    past_actions: Array
        Action indices of shape `(T, num_factors)`.

    Returns
    -------
    list[Array]
        Per-factor action-conditioned transitions with time as the leading axis.
    """

    nf = len(B)
    actions_tree = [past_actions[:, i] for i in range(nf)]

    # move time steps to the leading axis (leftmost)
    # this assumes that a policy is always specified as the rightmost axis of Bs
    B = jtu.tree_map(
        lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0),
        B,
        actions_tree,
    )

    return B


def init_B_from_spec(
    num_states: Sequence[int],
    num_controls: Sequence[int],
    batch_size: int = 1,
) -> list[Array]:
    """Create random transition tensors with trivial `B_dependencies`.

    Companion to :func:`init_A_and_D_from_spec` for sequence-based inference.
    Each factor's transitions depend only on that factor's own state (i.e.
    `B_dependencies = [[0], [1], ...]`), matching the constraint of the
    optimized VMP routines.

    Parameters
    ----------
    num_states: Sequence[int]
        Hidden-state cardinalities.
    num_controls: Sequence[int]
        Control cardinalities per factor.
    batch_size: int, default=1
        Number of sampled model instances.

    Returns
    -------
    list[Array]
        Per-factor tensors of shape `(batch, ns, ns, nc)`, column-normalized.
    """

    B = []
    for ns, nc in zip(num_states, num_controls):
        B_m = [np.random.rand(ns, ns, nc) for batch in range(batch_size)]
        B_m = [b / b.sum(axis=0, keepdims=True) for b in B_m]
        B_m = np.concatenate([b[np.newaxis, ...] for b in B_m], axis=0)
        B.append(B_m)

    return [jnp.array(b) for b in B]


def get_sample_action_seq(num_controls: Sequence[int], T: int, batch_size: int = 1) -> Array:
    """Generate a random action sequence for each batch element.

    Parameters
    ----------
    num_controls: Sequence[int]
        Control cardinalities per factor.
    T: int
        Number of timesteps.
    batch_size: int, default=1
        Number of batch elements.

    Returns
    -------
    Array
        Integer actions of shape `(batch_size, T, num_factors)`.
    """
    action_seq = []
    for i in range(batch_size):
        batch = []
        for j in range(T):
            batch.append([np.random.randint(0, control_dim) for control_dim in num_controls])
        action_seq.append(batch)
    return jnp.array(action_seq)


def get_sample_state(num_states: Sequence[int], batch_size: int = 1) -> list[Array]:
    """Generate a random hidden state for each batch element.

    Parameters
    ----------
    num_states: Sequence[int]
        Hidden-state cardinalities.
    batch_size: int, default=1
        Number of batch elements.

    Returns
    -------
    list[Array]
        Per-factor state indices of shape `(batch_size,)`.
    """
    state = [[np.random.randint(0, state_dim) for _ in range(batch_size)] for state_dim in num_states]
    return [jnp.array(s) for s in state]


def get_obs_seq_from_actions(
    action_seq: Array,
    start_state: list[Array],
    A: list[Array],
    A_dependencies: list[list[int]],
    B: list[Array],
    B_dependencies: list[list[int]],
) -> list[Array]:
    """Roll out an observation sequence by simulating the generative model.

    Starting from `start_state`, states are propagated through the
    action-conditioned transitions `B` and observations are sampled from the
    likelihoods `A` at every step, yielding `T + 1` observations for `T`
    actions.

    Parameters
    ----------
    action_seq: Array
        Action indices of shape `(batch_size, T, num_factors)`.
    start_state: list[Array]
        Initial per-factor state indices of shape `(batch_size,)`.
    A: list[Array]
        Likelihood tensors per modality.
    A_dependencies: list[list[int]]
        Modality-to-state dependencies.
    B: list[Array]
        Transition tensors per factor.
    B_dependencies: list[list[int]]
        Factor-to-factor dependencies.

    Returns
    -------
    list[Array]
        Per-modality observation indices of shape `(batch_size, T + 1)`.
    """

    nm, nf = len(A), len(B)
    batch_size, T = action_seq.shape[:2]

    obs_seq = [[[] for i in range(batch_size)] for k in range(nm)]
    for k in range(nm):
        for i in range(batch_size):
            s_idx = tuple(start_state[f][i] for f in A_dependencies[k])
            A_tmp = A[k][(i, slice(None, None)) + s_idx]
            if np.sum(A_tmp) < 0.0001:
                A_tmp = np.ones_like(A_tmp) / A_tmp.shape[0]
            obs_seq[k][i].append(_rnd_choices(list(range(A_tmp.shape[0])), weights=A_tmp, k=1)[0])

    curr_state = start_state
    for j in range(T):

        next_state = [[None for i in range(batch_size)] for k in range(nf)]
        for k in range(nf):
            for i in range(batch_size):
                s_idx = tuple(curr_state[f][i] for f in B_dependencies[k])
                a_idx = (action_seq[i, j, k],)
                B_tmp = B[k][(i, slice(None, None)) + s_idx + a_idx]
                if np.sum(B_tmp) < 0.0001:
                    B_tmp = np.ones_like(B_tmp) / B_tmp.shape[0]
                next_state[k][i] = _rnd_choices(list(range(B_tmp.shape[0])), weights=B_tmp, k=1)[0]

        curr_state = next_state
        for k in range(nm):
            for i in range(batch_size):
                s_idx = tuple(curr_state[f][i] for f in A_dependencies[k])
                A_tmp = A[k][(i, slice(None, None)) + s_idx]
                if np.sum(A_tmp) < 0.0001:
                    A_tmp = np.ones_like(A_tmp) / A_tmp.shape[0]
                obs_seq[k][i].append(_rnd_choices(list(range(A_tmp.shape[0])), weights=A_tmp, k=1)[0])

    return [jnp.array(o) for o in obs_seq]


# Modality-clustering utilities for optimized inference

def get_A_dep_clusters(A_dependencies: list[list[int]]) -> list[list[int]]:
    """Group modalities into clusters by the number of factors they depend on.

    Modalities whose `A_dependencies` have equal length share padded array
    shapes and can therefore be batched together with less padding overhead
    than one global padded batch.

    Parameters
    ----------
    A_dependencies: list[list[int]]
        Modality-to-state dependencies.

    Returns
    -------
    list[list[int]]
        Cluster-to-modality mapping: one list of modality indices per cluster.
    """

    num_modalities = len(A_dependencies)
    lengths = sorted({len(A_dep) for A_dep in A_dependencies})
    num_lengths = len(lengths)

    o2c_mapping = [lengths.index(len(A_dep)) for A_dep in A_dependencies]

    c2o_mapping = [
        [i for i in range(num_modalities) if o2c_mapping[i] == c]
        for c in range(num_lengths)
    ]

    return c2o_mapping


def apply_padding_per_cluster(xs: list[Array], c2o_mapping: list[list[int]]) -> list[Array]:
    """Apply :func:`apply_padding_batched` separately within each cluster.

    Parameters
    ----------
    xs: list[Array]
        Arrays to pad and concatenate (e.g. `A` tensors or observations).
    c2o_mapping: list[list[int]]
        Cluster-to-modality mapping from :func:`get_A_dep_clusters`.

    Returns
    -------
    list[Array]
        One padded batch tensor per cluster.
    """
    return [
        apply_padding_batched([xs[i] for i in c2o_mapping[c]])
        for c in range(len(c2o_mapping))
    ]


def apply_A_end2end_padding_per_cluster(A: list[Array], c2o_mapping: list[list[int]]) -> list[Array]:
    """Apply :func:`apply_A_end2end_padding_batched` separately within each cluster.

    Parameters
    ----------
    A: list[Array]
        Likelihood tensors per modality.
    c2o_mapping: list[list[int]]
        Cluster-to-modality mapping from :func:`get_A_dep_clusters`.

    Returns
    -------
    list[Array]
        One batched padded `A` tensor per cluster.
    """
    return [
        apply_A_end2end_padding_batched([A[i] for i in c2o_mapping[c]])
        for c in range(len(c2o_mapping))
    ]


def apply_obs_end2end_padding_per_cluster(
    obs: list[Array], c2o_mapping: list[list[int]], max_obs_dims: Sequence[int]
) -> list[Array]:
    """Apply :func:`apply_obs_end2end_padding_batched` separately within each cluster.

    Parameters
    ----------
    obs: list[Array]
        Observation tensors per modality.
    c2o_mapping: list[list[int]]
        Cluster-to-modality mapping from :func:`get_A_dep_clusters`.
    max_obs_dims: Sequence[int]
        Observation-axis padding target per cluster.

    Returns
    -------
    list[Array]
        One batched padded observation tensor per cluster.
    """
    return [
        apply_obs_end2end_padding_batched([obs[i] for i in c2o_mapping[c]], max_obs_dims[c])
        for c in range(len(c2o_mapping))
    ]


@dataclass
class _BlockGroup:
    R: int
    C: int
    members: set
    alive: bool = True
    ver: int = 0  # version number, increments when the group changes (for stale-heap detection)


def _block_group_objective(groups: list[_BlockGroup]) -> int:
    return sum(g.R * g.C for g in groups if g.alive)


def _block_group_merge_delta(a: _BlockGroup, b: _BlockGroup) -> int:
    # Increase in sum_g R_g*C_g if we merge a and b
    return a.R * b.C + b.R * a.C


def agglomerative_min_structural_zeros(
    rc_list: list[tuple[int, int]],
    target_K: int,
    *,
    R_max: int | None = None,  # forbid merges that would exceed this R
    C_max: int | None = None,  # forbid merges that would exceed this C
    area_max: int | None = None,  # forbid merges that would exceed this R*C
    record_history: bool = True,  # keep (K, objective, snapshot) after each merge
    keep_snapshots: bool = False,  # store group membership per step (can be big)
) -> dict[str, Any]:
    """
    Bottom-up agglomerative grouping to minimize structural zeros with a fixed #groups.

    This is a host-side (pure Python, non-jittable) preprocessing utility used
    to decide how modalities are grouped for clustered block-diagonal
    likelihood evaluation.

    Args:
        rc_list: iterable of (r_i, c_i) per block.
        target_K: stop when this many groups remain (1..M).
        R_max, C_max, area_max: optional caps for merged groups.
        record_history: whether to return a 'history' list of (num_groups, objective, snapshot?).
        keep_snapshots: if True, store a snapshot of group memberships at each step.

    Returns:
        dict with keys:
          - 'groups': list[list[int]] final groups (indices into the original rc_list)
          - 'objective': int, objective value sum_g R_g*C_g at the end
          - 'zeros': int, structural zeros = objective - sum_i r_i*c_i
          - 'history': optional list of dicts with per-step info for elbow analysis
    """
    rc = list(rc_list)
    M = len(rc)
    if not (1 <= target_K <= M):
        raise ValueError("target_K must be between 1 and M.")

    # Create singleton groups
    groups: list[_BlockGroup] = [
        _BlockGroup(R=r, C=c, members={i}) for i, (r, c) in enumerate(rc)
    ]

    # Precompute total useful area once (constant term)
    total_area = sum(r * c for r, c in rc)

    # Build initial heap of all candidate merges
    # We store (delta, a_idx, a_ver, b_idx, b_ver)
    heap: list[tuple[int, int, int, int, int]] = []
    for a in range(M):
        for b in range(a + 1, M):
            d = _block_group_merge_delta(groups[a], groups[b])
            heappush(heap, (d, a, groups[a].ver, b, groups[b].ver))

    # Helper: check caps
    def allowed_merge(a: _BlockGroup, b: _BlockGroup) -> bool:
        R = a.R + b.R
        C = a.C + b.C
        if (R_max is not None) and (R > R_max):
            return False
        if (C_max is not None) and (C > C_max):
            return False
        if (area_max is not None) and (R * C > area_max):
            return False
        return True

    # History (optional)
    history: list[dict[str, Any]] = []

    def snapshot() -> list[list[int]]:
        return [sorted(g.members) for g in groups if g.alive]

    if record_history:
        history.append(
            {
                "num_groups": M,
                "objective": _block_group_objective(groups),
                **({"groups": snapshot()} if keep_snapshots else {}),
            }
        )

    alive_count = M

    # Merge loop until we have target_K groups
    while alive_count > target_K:
        # Get best feasible, fresh pair
        while heap:
            d, a_idx, a_ver, b_idx, b_ver = heappop(heap)
            if a_idx >= len(groups) or b_idx >= len(groups):
                continue
            ga, gb = groups[a_idx], groups[b_idx]
            # Skip if either dead or version changed (stale)
            if (
                (not ga.alive)
                or (not gb.alive)
                or (ga.ver != a_ver)
                or (gb.ver != b_ver)
            ):
                continue
            # Respect caps
            if not allowed_merge(ga, gb):
                # If the best pair is forbidden, we still need to consider other pairs;
                # just continue popping.
                continue
            # Fresh and feasible
            break
        else:
            # No feasible merge found (caps too strict)
            break

        # Merge b into a (keep index a)
        ga.R += gb.R
        ga.C += gb.C
        ga.members |= gb.members
        ga.ver += 1

        gb.alive = False
        alive_count -= 1

        # Push new deltas involving 'a'
        for k, gk in enumerate(groups):
            if k == a_idx or (not gk.alive):
                continue
            heappush(heap, (_block_group_merge_delta(ga, gk), a_idx, ga.ver, k, gk.ver))

        # Record history
        if record_history:
            history.append(
                {
                    "num_groups": alive_count,
                    "objective": _block_group_objective(groups),
                    **({"groups": snapshot()} if keep_snapshots else {}),
                }
            )

    # Collect final groups
    final_groups = [sorted(g.members) for g in groups if g.alive]
    final_objective = _block_group_objective(groups)
    structural_zeros = final_objective - total_area

    return {
        "groups": final_groups,
        "objective": final_objective,
        "zeros": structural_zeros,
        "history": history if record_history else None,
    }


def _extract_FK(history: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    """Return Ks (descending) and F(K) from the agglomerative history."""
    pairs = [(h["num_groups"], h["objective"]) for h in history]
    pairs.sort(reverse=True)  # Ks: M, M-1, ..., 1
    Ks = [int(n) for n, _ in pairs]
    Fs = [float(f) for _, f in pairs]
    return Ks, Fs


def kneedle_max_distance_to_chord(
    history: list[dict[str, Any]],
    plot: bool = False,
    chords_sample: int = 20,  # draw ~this many chords on the normalized plot
) -> dict[str, Any]:
    """
    Kneedle-style knee using max distance to the chord between endpoints, after
    normalizing both axes to [0,1].

    Args:
        history: result["history"] from agglomerative_min_structural_zeros(...).
                 Must contain dicts with keys: "num_groups", "objective".
        plot:    if True, show two figures:
                 (1) normalized curve with endpoint chord and perpendiculars
                 (2) original F(K) vs K with the knee highlighted
        chords_sample: approx number of perpendicular chords to draw (to avoid clutter).

    Returns:
        dict with:
          - K_knee: selected number of groups (int)
          - knee_index: index in the (sorted-desc) Ks/Fs arrays
          - Ks, Fs: the arrays used
          - distances: per-point distances to the chord on the normalized curve
    """
    Ks, Fs = _extract_FK(history)

    # Normalize Ks and Fs to [0,1]
    kmin, kmax = float(min(Ks)), float(max(Ks))
    fmin, fmax = float(min(Fs)), float(max(Fs))
    x = [(k - kmin) / (kmax - kmin + 1e-12) for k in Ks]
    y = [(f - fmin) / (fmax - fmin + 1e-12) for f in Fs]

    # Endpoints of the chord (first and last points in this order)
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    denom = math.hypot(x1 - x0, y1 - y0) + 1e-12  # euclidean distance between endpoints

    # Perpendicular distance of each point to the endpoint chord
    distances = [
        abs((y1 - y0) * xi - (x1 - x0) * yi + x1 * y0 - y1 * x0) / denom
        for xi, yi in zip(x, y)
    ]
    knee_index = max(range(len(distances)), key=lambda i: distances[i])
    K_knee = Ks[knee_index]

    if plot:
        # Helper: project a point onto the chord
        vx, vy = (x1 - x0), (y1 - y0)
        v2 = vx * vx + vy * vy + 1e-12

        def project_to_chord(xi, yi):
            t = ((xi - x0) * vx + (yi - y0) * vy) / v2
            return (x0 + t * vx, y0 + t * vy)

        # (1) Normalized plot with endpoint chord + perpendicular chords
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, marker="o", label="Normalized F(K)")
        plt.plot([x0, x1], [y0, y1], linestyle="--", label="Chord (endpoints)")

        # draw ~chords_sample chords to avoid clutter
        step = max(1, len(x) // max(1, chords_sample))
        for i in range(0, len(x), step):
            xp, yp = project_to_chord(x[i], y[i])
            plt.plot([x[i], xp], [y[i], yp], color="0.8", linewidth=1)

        # knee chord in bold
        xk, yk = x[knee_index], y[knee_index]
        xkp, ykp = project_to_chord(xk, yk)
        plt.plot([xk, xkp], [yk, ykp], linewidth=3, label=f"Knee chord (K={K_knee})")

        plt.title("Kneedle (max distance) with chords to the endpoint line")
        plt.xlabel("Normalized K")
        plt.ylabel("Normalized F(K)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # (2) Original-scale elbow curve with knee highlighted
        plt.figure(figsize=(7, 5))
        plt.plot(Ks, Fs, marker="o")
        plt.plot(Ks[knee_index], Fs[knee_index], marker="o", color="red", markersize=10)
        plt.title(f"Elbow curve with Kneedle knee highlighted (K={K_knee})")
        plt.xlabel("K (number of groups)")
        plt.ylabel("F(K) = sum of group rectangles")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "K_knee": K_knee,
        "knee_index": knee_index,
        "Ks": Ks,
        "Fs": Fs,
        "distances": distances,
    }


def prep_clustered_block_data(
    A: list[Array], o_vec: list[Array]
) -> tuple[list[Array], list[Array], list[tuple], list[tuple], list[list[int]]]:
    """Cluster modalities and build per-group block-diagonal data.

    Runs the (host-side, non-jittable) agglomerative clustering over the
    flattened `A` shapes, selects the number of groups with a Kneedle knee
    criterion, and constructs one block-diagonal `A` and one concatenated
    observation tensor per group via :func:`build_block_diag_A`.

    Parameters
    ----------
    A: list[Array]
        Likelihood tensors per modality.
    o_vec: list[Array]
        One-hot observation tensors per modality.

    Returns
    -------
    tuple[list[Array], list[Array], list[tuple], list[tuple], list[list[int]]]
        `(A_block, obs_block, state_shapes_block, cuts_block, group_mapping)`,
        each indexed by group.
    """

    A_rc = [(a.shape[0], math.prod(a.shape[1:])) for a in A]
    clustering_result = agglomerative_min_structural_zeros(A_rc, target_K=1, record_history=True, keep_snapshots=True)
    knee = kneedle_max_distance_to_chord(clustering_result['history'])
    group_mapping = list(filter(lambda x: x['num_groups'] == knee["K_knee"], clustering_result["history"]))[0]['groups']

    A_block = []
    state_shapes_block = []
    cuts_block = []
    for group in group_mapping:
        A_block_group, state_shapes_group, cuts_group = build_block_diag_A([jnp.moveaxis(A[i], 1, -1) for i in group])
        A_block.append(A_block_group)
        state_shapes_block.append(state_shapes_group)
        cuts_block.append(cuts_group)

    obs_block = []
    for group in group_mapping:
        obs_block_group = jnp.concatenate([o_vec[i] for i in group], axis=-1)
        obs_block.append(obs_block_group)

    return A_block, obs_block, state_shapes_block, cuts_block, group_mapping