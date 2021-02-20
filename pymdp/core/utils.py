#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Utility functions

__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import numpy as np

from pymdp.distributions import Categorical, Dirichlet

def sample(probabilities):
    # TODO dont assume dist class
    # if probabilities.shape[1] > 1:
    #     raise ValueError("Can only currently sample from [n x 1] distribution")
    sample_onehot = np.random.multinomial(1, probabilities.squeeze())
    return np.where(sample_onehot == 1)[0][0]


def obj_array(num_arr):
    """
    Creates a generic object array with the desired number of sub-arrays, given by `num_arr`
    """
    return np.empty(num_arr, dtype=object)

def obj_array_zeros(shape_list):
    """ 
    Creates a numpy object array whose sub-arrays are 1-D vectors
    filled with zeros, with shapes given by shape_list[i]
    """
    arr = np.empty(len(shape_list), dtype=object)
    for i, shape in enumerate(shape_list):
        arr[i] = np.zeros(shape)
    return arr

def obj_array_uniform(shape_list):
    """ 
    Creates a numpy object array whose sub-arrays are uniform Categorical
    distributions with shapes given by shape_list[i]
    """
    arr = np.empty(len(shape_list), dtype=object)
    for i, shape in enumerate(shape_list):
        arr[i] = np.ones(shape)/shape
    return arr

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


def get_model_dimensions(A=None, B=None):

    if A is None and B is None:
        raise ValueError(
                    "Must provide either `A` or `B`"
                )

    if A is not None:
        num_obs = [a.shape[0] for a in A] if is_arr_of_arr(A) else [A.shape[0]]
        num_modalities = len(num_obs)
    else:
        num_obs, num_modalities = None, None
    
    if B is not None:
        num_states = [b.shape[0] for b in B] if is_arr_of_arr(B) else [B.shape[0]]
        num_factors = len(num_states)
    else:
        num_states, num_factors = None, None
    
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

def process_observation_seq(obs_seq, n_modalities, n_observations):
    """
    Helper function for formatting observations    

        Observations can either be `Categorical`, `int` (converted to one-hot)
        or `tuple` (obs for each modality)
    
    @TODO maybe provide error messaging about observation format
    """
    proc_obs_seq = np.empty(len(obs_seq), dtype=object)
    for t in range(len(obs_seq)):
        proc_obs_seq[t] = process_observation(obs_seq[t], n_modalities, n_observations)
    return proc_obs_seq

def process_observation(obs, n_modalities, n_observations):
    """
    Helper function for formatting observations    

        Observations can either be `Categorical`, `int` (converted to one-hot)
        or `tuple` (obs for each modality)
    
    @TODO maybe provide error messaging about observation format
    """
    if is_distribution(obs):
        obs = to_numpy(obs)
        if n_modalities == 1:
            obs = obs.squeeze()
        else:
            for m in range(n_modalities):
                obs[m] = obs[m].squeeze()

    if isinstance(obs, (int, np.integer)):
        obs = np.eye(n_observations[0])[obs]

    if isinstance(obs, tuple):
        obs_arr_arr = np.empty(n_modalities, dtype=object)
        for m in range(n_modalities):
            obs_arr_arr[m] = np.eye(n_observations[m])[obs[m]]
        obs = obs_arr_arr

    return obs

def process_prior(prior, n_factors):
    """
    Helper function for formatting prior beliefs  
        """
    if is_distribution(prior):
        prior_arr = obj_array(n_factors)
        if n_factors == 1:
            prior_arr[0] = prior.values.squeeze()
        else:
            for factor in range(n_factors):
                prior_arr[factor] = prior[factor].values.squeeze()
        prior = prior_arr

    elif not is_arr_of_arr(prior):
        prior = to_arr_of_arr(prior)

    return prior

