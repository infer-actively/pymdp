#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Utility functions

__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import numpy as np

from pymdp.distributions import Categorical, Dirichlet


def obj_array(shape):
    return np.empty(shape, dtype=object)


def onehot(value, num_values):
    arr = np.zeros(num_values)
    arr[value] = 1.0
    return arr


def random_A_matrix(num_obs, num_states):
    if type(num_obs) is int:
        num_obs = [num_obs]
    if type(num_states) is int:
        num_states = [num_states]
    num_modalities = len(num_obs)

    A = obj_array(num_modalities)
    for modality, modality_obs in enumerate(num_obs):
        modality_shape = [modality_obs] + num_states
        modality_dist = np.random.rand(*modality_shape)
        A[modality] = norm_dist(modality_dist)
    return A


def random_B_matrix(num_states, num_controls):
    if type(num_states) is int:
        num_states = [num_states]
    if type(num_controls) is int:
        num_controls = [num_controls]
    num_factors = len(num_states)
    assert len(num_controls) == len(num_states)

    B = obj_array(num_factors)
    for factor in range(num_factors):
        factor_shape = (num_states[factor], num_states[factor], num_controls[factor])
        factor_dist = np.random.rand(*factor_shape)
        B[factor] = norm_dist(factor_dist)
    return B


def get_model_dimensions(A, B):
    num_obs = [a.shape[0] for a in A] if is_arr_of_arr(A) else [A.shape[0]]
    num_states = [b.shape[0] for b in B] if is_arr_of_arr(B) else [B.shape[0]]
    num_modalities = len(num_obs)
    num_factors = len(num_states)
    return num_obs, num_states, num_modalities, num_factors


def norm_dist(dist):
    if len(dist.shape) == 3:
        new_dist = np.zeros_like(dist)
        for c in range(dist.shape[2]):
            new_dist[:, :, c] = np.divide(dist[:, :, c], dist[:, :, c].sum(axis=0))
        return new_dist
    else:
        return np.divide(dist, dist.sum(axis=0))


def to_numpy(dist, flatten=False):
    """
    If flatten is True, then the individual entries of the object array will be 
    flattened into row vectors(common operation when dealing with array of arrays 
    with 1D numpy array entries)
    """
    if isinstance(dist, (Categorical, Dirichlet)):
        values = np.copy(dist.values)
        if flatten:
            if dist.IS_AOA:
                for idx, arr in enumerate(values):
                    values[idx] = arr.flatten()
            else:
                values = values.flatten()
    else:
        values = dist
        if flatten:
            if is_arr_of_arr(values):
                for idx, arr in enumerate(values):
                    values[idx] = arr.flatten()
            else:
                values = values.flatten()
    return values


def is_distribution(obj):
    return isinstance(obj, (Categorical, Dirichlet))


def is_arr_of_arr(arr):
    return arr.dtype == "object"


def to_arr_of_arr(arr):
    if is_arr_of_arr(arr):
        return arr
    arr_of_arr = np.empty(1, dtype=object)
    arr_of_arr[0] = arr.squeeze()
    return arr_of_arr


def to_categorical(values):
    return Categorical(values=values)


def to_dirichlet(values):
    return Dirichlet(values=values)

