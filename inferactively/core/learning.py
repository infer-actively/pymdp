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
from inferactively.distributions import Categorical, Dirichlet


def update_dirichlet_likelihood(pA, A, obs, Qs, eta=1.0, return_numpy=True, which_modalities="all"):
    """
    Update Dirichlet parameters that parameterize the observation model of the generative model (describing the probabilistic mapping from hidden states to observations).
    Parameters
    -----------
    pA [numpy nd.array, array-of-arrays (with np.ndarray entries), or Dirichlet (either single-modality or AoA)]:
        The prior Dirichlet parameters of the generative model, parameterizing the agent's beliefs about the observation likelihood. 
    A [numpy nd.array, object-like array of arrays, or Categorical (either single-modality or AoA)]:
        The observation likelihood of the generative model. 
    obs [numpy 1D array, array-of-arrays (with 1D numpy array entries), int or tuple]:
        A discrete observation used in the update equation
    Qs [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Current marginal posterior beliefs about hidden state factors
    eta [float, optional]:
        Learning rate.
    return_numpy [bool, optional]:
        Logical flag to determine whether output is a numpy array or a Dirichlet
    which_modalities [list, optional]:
        Indices (in terms of range(Ng)) of the observation modalities to include in learning.
        Defaults to 'all, meaning that observation likelihood matrices for all modalities
        are updated as a function of observations in the different modalities.
    """

    if isinstance(pA, Dirichlet):
        if pA.IS_AOA:
            Ng = len(pA)
            No = [pA[g].shape[0] for g in range(Ng)]
        else:
            Ng = 1
            No = [pA.shape[0]]
        if return_numpy:
            pA_new = pA.values.copy()
        else:
            pA_new = Dirichlet(values=pA.values.copy())

    else:
        if pA.dtype == object:
            Ng = len(pA)
            No = [pA[g].shape[0] for g in range(Ng)]
        else:
            Ng = 1
            No = [pA.shape[0]]
        if return_numpy:
            pA_new = pA.copy()
        else:
            pA_new = Dirichlet(values=pA.copy())

    if isinstance(A, Categorical):
        A = A.values

    if isinstance(obs, (int, np.integer)):
        obs = np.eye(A.shape[0])[obs]

    elif isinstance(obs, tuple):
        obs = np.array([np.eye(No[g])[obs[g]] for g in range(Ng)], dtype=object)

    obs = Categorical(values=obs)  # convert to Categorical to make the cross product easier

    if which_modalities == "all":
        if Ng == 1:
            da = obs.cross(Qs, return_numpy=True)
            da = da * (A > 0).astype("float")
            pA_new = pA_new + (eta * da)
        elif Ng > 1:
            for g in range(Ng):
                da = obs[g].cross(Qs, return_numpy=True)
                da = da * (A[g] > 0).astype("float")
                pA_new[g] = pA_new[g] + (eta * da)
    else:
        for g_idx in which_modalities:
            da = obs[g_idx].cross(Qs, return_numpy=True)
            da = da * (A[g_idx] > 0).astype("float")
            pA_new[g_idx] = pA_new[g_idx] + (eta * da)

    return pA_new


def update_dirichlet_transition(
    pB, B, action, Qs_curr, Qs_prev, eta=1.0, return_numpy=True, which_factors="all"
):
    """
    Update Dirichlet parameters that parameterize the transition model of the generative model 
    (describing the probabilistic mapping between hidden states over time).

    Parameters
    -----------
    pB [numpy nd.array, array-of-arrays (with np.ndarray entries), or Dirichlet (either single-modality or AoA)]:
        The prior Dirichlet parameters of the generative model, parameterizing the agent's beliefs about the transition likelihood. 
    B [numpy nd.array, object-like array of arrays, or Categorical (either single-modality or AoA)]:
        The transition likelihood of the generative model. 
    action [tuple]:
        A tuple containing the action(s) performed at a given timestep.
    Qs_curr [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Current marginal posterior beliefs about hidden state factors
    Qs_prev [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Past marginal posterior beliefs about hidden state factors
    eta [float, optional]:
        Learning rate.
    return_numpy [bool, optional]:
        Logical flag to determine whether output is a numpy array or a Dirichlet
    which_factors [list, optional]:
        Indices (in terms of range(Nf)) of the hidden state factors to include in learning.
        Defaults to 'all', meaning that transition likelihood matrices for all hidden state factors
        are updated as a function of transitions in the different control factors (i.e. actions)
    """

    if isinstance(pB, Dirichlet):
        if pB.IS_AOA:
            Nf = len(pB)
        else:
            Nf = 1
        if return_numpy:
            pB_new = pB.values.copy()
        else:
            pB_new = Dirichlet(values=pB.values.copy())

    else:
        if pB.dtype == object:
            Nf = len(pB)
        else:
            Nf = 1
        if return_numpy:
            pB_new = pB.copy()
        else:
            pB_new = Dirichlet(values=pB.copy())

    if isinstance(B, Categorical):
        B = B.values

    if not isinstance(Qs_curr, Categorical):
        Qs_curr = Categorical(values=Qs_curr)

    if which_factors == "all":
        if Nf == 1:
            db = Qs_curr.cross(Qs_prev, return_numpy=True)
            db = db * (B[:, :, action[0]] > 0).astype("float")
            pB_new = pB_new + (eta * db)
        elif Nf > 1:
            for f in range(Nf):
                db = Qs_curr[f].cross(Qs_prev[f], return_numpy=True)
                db = db * (B[f][:, :, action[f]] > 0).astype("float")
                pB_new[f] = pB_new[f] + (eta * db)
    else:
        for f_idx in which_factors:
            db = Qs_curr[f_idx].cross(Qs_prev[f_idx], return_numpy=True)
            db = db * (B[f_idx][:, :, action[f_idx]] > 0).astype("float")
            pB_new[f_idx] = pB_new[f_idx] + (eta * db)

    return pB_new
