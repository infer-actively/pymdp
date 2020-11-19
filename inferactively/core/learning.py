#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

""" Functions
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import itertools
import numpy as np
import torch
from scipy import special
from inferactively.core import utils
import copy


def update_likelihood_dirichlet(pA, A, obs, qs, lr=1.0, modalities="all", return_numpy=True):
    """ Update Dirichlet parameters of the likelihood distribution 

    Parameters
    -----------
    - pA [numpy nd.array, array-of-arrays (with np.ndarray entries), or Dirichlet (either single-modality or AoA)]:
        The prior Dirichlet parameters of the generative model, parameterizing the agent's beliefs about the observation likelihood. 
    - A [numpy nd.array, object-like array of arrays, or Categorical (either single-modality or AoA)]:
        The observation likelihood of the generative model. 
    - obs [numpy 1D array, array-of-arrays (with 1D numpy array entries), int or tuple]:
        A discrete observation (possible multi-modality) used in the update equation
    - qs [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Current marginal posterior beliefs about hidden state factors
    - lr [float, optional]:
        Learning rate.
    - return_numpy [bool, optional]:
        Logical flag to determine whether output is a numpy array or a Dirichlet
    - modalities [list, optional]:
        Indices (in terms of range(n_modalities)) of the observation modalities to include in learning.
        Defaults to 'all', meaning that observation likelihood matrices for all modalities
        are updated using their respective observations.
    """

    pA = utils.to_numpy(pA)
    A = utils.to_numpy(A)

    if utils.is_arr_of_arr(pA):
        n_modalities = len(pA)
        n_observations = [pA[modality].shape[0] for modality in range(n_modalities)]
    else:
        n_modalities = 1
        n_observations = [pA.shape[0]]

    if return_numpy:
        pA_updated = copy.deepcopy(pA)
    else:
        pA_updated = utils.to_dirichlet(copy.deepcopy(pA))

    # observation index
    if isinstance(obs, (int, np.integer)):
        obs = np.eye(A.shape[0])[obs]

    # observation indices
    elif isinstance(obs, tuple):
        obs = np.array(
            [np.eye(n_observations[modality])[obs[modality]] for modality in range(n_modalities)],
            dtype=object,
        )

    # convert to Categorical to make the cross product easier
    obs = utils.to_categorical(obs)

    if modalities == "all":
        if n_modalities == 1:
            dfda = obs.cross(qs, return_numpy=True)
            dfda = dfda * (A > 0).astype("float")
            pA_updated = pA_updated + (lr * dfda)

        elif n_modalities > 1:
            for modality in range(n_modalities):
                dfda = obs[modality].cross(qs, return_numpy=True)
                dfda = dfda * (A[modality] > 0).astype("float")
                pA_updated[modality] = pA_updated[modality] + (lr * dfda)
    else:
        for modality in modalities:
            dfda = obs[modality].cross(qs, return_numpy=True)
            dfda = dfda * (A[modality] > 0).astype("float")
            pA_updated[modality] = pA_updated[modality] + (lr * dfda)

    return pA_updated


def update_transition_dirichlet(
    pB, B, actions, qs, qs_prev, lr=1.0, factors="all", return_numpy=True
):
    """
    Update Dirichlet parameters that parameterize the transition model of the generative model 
    (describing the probabilistic mapping between hidden states over time).

    Parameters
    -----------
   -  pB [numpy nd.array, array-of-arrays (with np.ndarray entries), or Dirichlet (either single-modality or AoA)]:
        The prior Dirichlet parameters of the generative model, parameterizing the agent's beliefs about the transition likelihood. 
    - B [numpy nd.array, object-like array of arrays, or Categorical (either single-modality or AoA)]:
        The transition likelihood of the generative model. 
    - actions [numpy 1D array]:
        A 1D numpy array of shape (num_control_factors,) containing the action(s) performed at a given timestep.
    - qs [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Current marginal posterior beliefs about hidden state factors
    - qs_prev [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Past marginal posterior beliefs about hidden state factors
    - lr [float, optional]:
        Learning rate.
    - return_numpy [bool, optional]:
        Logical flag to determine whether output is a numpy array or a Dirichlet
    - factors [list, optional]:
        Indices (in terms of range(Nf)) of the hidden state factors to include in learning.
        Defaults to 'all', meaning that transition likelihood matrices for all hidden state factors
        are updated as a function of transitions in the different control factors (i.e. actions)
    """

    pB = utils.to_numpy(pB)
    B = utils.to_numpy(B)

    if utils.is_arr_of_arr(pB):
        n_factors = len(pB)
    else:
        n_factors = 1

    if return_numpy:
        pB_updated = copy.deepcopy(pB)
    else:
        pB_updated = utils.to_dirichlet(copy.deepcopy(pB))

    if not utils.is_distribution(qs):
        qs = utils.to_categorical(qs)

    if factors == "all":
        if n_factors == 1:
            dfdb = qs.cross(qs_prev, return_numpy=True)
            dfdb = dfdb * (B[:, :, actions[0]] > 0).astype("float")
            pB_updated[:, :, actions[0]] = pB_updated[:, :, actions[0]] + (lr * dfdb)

        elif n_factors > 1:
            for factor in range(n_factors):
                dfdb = qs[factor].cross(qs_prev[factor], return_numpy=True)
                dfdb = dfdb * (B[factor][:, :, actions[factor]] > 0).astype("float")
                pB_updated[factor][:, :, actions[factor]] = pB_updated[factor][
                    :, :, actions[factor]
                ] + (lr * dfdb)
    else:
        for factor in factors:
            dfdb = qs[factor].cross(qs_prev[factor], return_numpy=True)
            dfdb = dfdb * (B[factor][:, :, actions[factor]] > 0).astype("float")
            pB_updated[factor][:, :, actions[factor]] = pB_updated[factor][
                :, :, actions[factor]
            ] + (lr * dfdb)

    return pB_updated
