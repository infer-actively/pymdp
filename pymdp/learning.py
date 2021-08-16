#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

""" Functions
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import numpy as np
from pymdp import utils, maths
import copy

def update_likelihood_dirichlet(pA, A, obs, qs, lr=1.0, modalities="all"):
    """ Update Dirichlet parameters of the likelihood distribution 

    Parameters
    -----------
    - pA [numpy object array]:
        The prior Dirichlet parameters of the generative model, parameterizing the 
        agent's beliefs about the observation likelihood. 
    - A [numpy object array]:
        The observation likelihood of the generative model. 
    - obs [numpy 1D array, array-of-arrays (with 1D numpy array entries), int, list, or tuple]:
        A discrete observation (possible multi-modality) used in the update equation
    - qs [numpy object array (where each entry is a numpy 1D array)]:
        Current marginal posterior beliefs about hidden state factors
    - lr [float, optional]:
        Learning rate.
    - modalities [list, optional]:
        Indices (in terms of range(n_modalities)) of the observation modalities to include 
        in learning.Defaults to 'all', meaning that observation likelihood matrices 
        for all modalities are updated using their respective observations.
    """


    num_modalities = len(pA)
    num_observations = [pA[modality].shape[0] for modality in range(num_modalities)]

    obs_processed = utils.process_observation(obs, num_modalities, num_observations)
    obs = utils.to_arr_of_arr(obs_processed)

    if modalities == "all":
        modalities = list(range(num_modalities))

    pA_updated = copy.deepcopy(pA)
        
    for modality in modalities:
        dfda = maths.spm_cross(obs[modality], qs)
        dfda = dfda * (A[modality] > 0).astype("float")
        pA_updated[modality] = pA_updated[modality] + (lr * dfda)

    return pA_updated

def update_transition_dirichlet(
    pB, B, actions, qs, qs_prev, lr=1.0, factors="all"
):
    """
    Update Dirichlet parameters that parameterize the transition model of the generative model 
    (describing the probabilistic mapping between hidden states over time).

    Parameters
    -----------
   -  pB [numpy object array]:
        The prior Dirichlet parameters of the generative model, parameterizing the agent's 
        beliefs about the transition likelihood. 
    - B [numpy object array]:
        The transition likelihood of the generative model. 
    - actions [numpy 1D array]:
        A 1D numpy array of shape (num_control_factors,) containing the action(s) performed at 
        a given timestep.
    - qs [numpy object array (where each entry is a numpy 1D array)]:
        Current marginal posterior beliefs about hidden state factors
    - qs_prev [numpy object array (where each entry is a numpy 1D array)]:
        Past marginal posterior beliefs about hidden state factors
    - lr [float, optional]:
        Learning rate.
    - factors [list, optional]:
        Indices (in terms of range(num_factors)) of the hidden state factors to include in learning.
        Defaults to 'all', meaning that transition likelihood matrices for all hidden state factors
        are updated as a function of transitions in the different control factors (i.e. actions)
    """

    num_factors = len(pB)

    pB_updated = copy.deepcopy(pB)
   
    if factors == "all":
        factors = list(range(num_factors))

    for factor in factors:
        dfdb = maths.spm_cross(qs[factor], qs_prev[factor])
        dfdb *= (B[factor][:, :, actions[factor]] > 0).astype("float")
        pB_updated[factor][:,:,int(actions[factor])] += (lr*dfdb)

    return pB_updated
