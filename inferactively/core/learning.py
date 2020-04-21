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


def update_likelihood_dirichlet(pA, A, obs, qs, lr=1.0, return_numpy=True, modalities="all"):
    """ Update Dirichlet parameters of the likelihood distribution 

    Parameters
    -----------
    - pA [numpy nd.array, array-of-arrays (with np.ndarray entries), or Dirichlet (either single-modality or AoA)]:
        The prior Dirichlet parameters of the generative model, parameterizing the agent's beliefs about the observation likelihood. 
    - A [numpy nd.array, object-like array of arrays, or Categorical (either single-modality or AoA)]:
        The observation likelihood of the generative model. 
    - obs [numpy 1D array, array-of-arrays (with 1D numpy array entries), int or tuple]:
        A discrete observation used in the update equation
    - Qx [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
            Current marginal posterior beliefs about hidden state factors
    - lr [float, optional]:
            Learning rate.
    - return_numpy [bool, optional]:
        Logical flag to determine whether output is a numpy array or a Dirichlet
    - modalities [list, optional]:
        Indices (in terms of range(n_modalities)) of the observation modalities to include in learning.
        Defaults to 'all, meaning that observation likelihood matrices for all modalities
        are updated as a function of observations in the different modalities.
    """

    pA = utils.to_numpy(pA)

    if utils.is_arr_of_arr(pA):
        n_modalities = len(pA)
        n_observations = [pA[m].shape[0] for m in range(n_modalities)]
    else:
        n_modalities = 1
        n_observations = [pA.shape[0]]

    if return_numpy:
        pA_updated = pA.copy()
    else:
        pA_updated = utils.to_dirichlet(pA.copy())

    # observation index
    if isinstance(obs, (int, np.integer)):
        obs = np.eye(A.shape[0])[obs]

    # observation indices
    elif isinstance(obs, tuple):
        obs = np.array(
            [np.eye(n_observations[g])[obs[g]] for g in range(n_modalities)], dtype=object
        )

    # convert to Categorical to make the cross product easier
    obs = utils.to_categorical(obs)

    if modalities == "all":
        if n_modalities == 1:
            da = obs.cross(qs, return_numpy=True)
            da = da * (A > 0).astype("float")
            pA_updated = pA_updated + (lr * da)

        elif n_modalities > 1:
            for g in range(n_modalities):
                da = obs[g].cross(qs, return_numpy=True)
                da = da * (A[g] > 0).astype("float")
                pA_updated[g] = pA_updated[g] + (lr * da)
    else:
        for g_idx in modalities:
            da = obs[g_idx].cross(qs, return_numpy=True)
            da = da * (A[g_idx] > 0).astype("float")
            pA_updated[g_idx] = pA_updated[g_idx] + (lr * da)

    return pA_updated


def update_transition_dirichlet(
    pB, B, actions, qs, qs_prev, lr=1.0, return_numpy=True, factors="all"
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
    - actions [tuple]:
        A tuple containing the action(s) performed at a given timestep.
    - Qs_curr [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Current marginal posterior beliefs about hidden state factors
    - Qs_prev [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Past marginal posterior beliefs about hidden state factors
    - eta [float, optional]:
        Learning rate.
    - return_numpy [bool, optional]:
        Logical flag to determine whether output is a numpy array or a Dirichlet
    - which_factors [list, optional]:
        Indices (in terms of range(Nf)) of the hidden state factors to include in learning.
        Defaults to 'all', meaning that transition likelihood matrices for all hidden state factors
        are updated as a function of transitions in the different control factors (i.e. actions)
    """

    pB = utils.to_numpy(pB)

    if utils.is_arr_of_arr(pB):
        n_factors = len(pB)
    else:
        n_factors = 1

    if return_numpy:
        pB_updated = pB.copy()
    else:
        pB_updated = utils.to_dirichlet(pB.copy())

    if not utils.is_distribution(qs):
        qs = utils.to_categorical(qs)

    if factors == "all":
        if n_factors == 1:
            db = qs.cross(qs_prev, return_numpy=True)
            db = db * (B[:, :, actions[0]] > 0).astype("float")
            pB_updated = pB_updated + (lr * db)

        elif n_factors > 1:
            for f in range(n_factors):
                db = qs[f].cross(qs_prev[f], return_numpy=True)
                db = db * (B[f][:, :, actions[f]] > 0).astype("float")
                pB_updated[f] = pB_updated[f] + (lr * db)
    else:
        for f_idx in factors:
            db = qs[f_idx].cross(qs_prev[f_idx], return_numpy=True)
            db = db * (B[f_idx][:, :, actions[f_idx]] > 0).astype("float")
            pB_updated[f_idx] = pB_updated[f_idx] + (lr * db)

    return pB_updated
