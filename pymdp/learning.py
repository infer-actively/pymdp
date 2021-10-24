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
        A discrete observation (possible multi-modality) used in the update equation.
        @NOTE on input formats:
        - If `obs` is a 1D numpy array, it must be a one-hot vector, where one entry (the entry of the observation) is 1.0 
        and all other entries are 0. This therefore assumes it's a single modality observation. 
        - If `obs` is an int, it assumes this is a single modality observation, whose observation index is given by the value of `obs`
        - If `obs` is a list, it assumes this is a multiple modality observation, whose len is equal to the number of observation modalities,
        and where each entry `obs[m]` is the index of the observation, for that modality.
        - If `obs` is a tuple, same logic as applies for list (see above).
        - if `obs` is a numpy object array (array of arrays), then this assumes the observation is a multiple modality observation, where each
        sub-array of `obs` is a 1D numpy array with a 1.0 in one entry and 0 everywhere else -- i.e. an object array of onehots.
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

def update_state_prior_dirichlet(
    pD, qs, lr=1.0, factors="all"
):
    """
    Update Dirichlet parameters that parameterize the hidden state prior of the generative model 
    (prior beliefs about hidden states at the beginning of the inference window).

    Parameters
    -----------
   -  pD [numpy object array]:
        The prior Dirichlet parameters of the generative model, parameterizing the agent's 
        beliefs about initial hidden states
    - qs [numpy object array (where each entry is a numpy 1D array)]:
        Current marginal posterior beliefs about hidden state factors
    - lr [float, optional]:
        Learning rate.
    - factors [list, optional]:
        Indices (in terms of range(num_factors)) of the hidden state factors to include in learning.
        Defaults to 'all', meaning that the priors over initial hidden states for all hidden state factors
        are updated.
    """

    num_factors = len(pD)

    pD_updated = copy.deepcopy(pD)
   
    if factors == "all":
        factors = list(range(num_factors))

    for factor in factors:
        idx = pD[factor] > 0 # only update those state level indices that have some prior probability
        pD_updated[factor][idx] += (lr * qs[factor][idx])
       
    return pD_updated

def prune_prior(prior, levels_to_remove):
    """
    Function for pruning a prior (with potentially multiple hidden state factors)
    Arguments:
    =========
    `prior` [1D np.ndarray or numpy object array with 1D entries]: The prior vector(s) containing the priors of a generative model, e.g. the prior over hidden states (D vector). 
    `levels_to_remove` [list]: a list of the hidden state or observation levels to remove. If the prior in question has multiple hidden state factors / multiple observation modalities, 
        then this will be a list of lists, where each sub-list within `levels_to_remove` will contain the levels to prune for a particular hidden state factor or modality 
    `dirichlet` [bool]: a flag telling whether the input array is a Dirichlet and therefore should not be normalized at the end. @TODO: Instead, the dirichlet parameters from the
                        pruned columns should somehow be re-distributed among the remaining columns
    Returns:
    =========
    `reduced_prior` [1D np.ndarray or numpy object array with 1D entries]: The prior vector(s), after pruning, lacks the hidden state or modality levels indexed by `levels_to_remove`
    """

    if utils.is_arr_of_arr(prior): # in case of multiple hidden state factors

        assert all([type(levels) == list for levels in levels_to_remove])

        num_factors = len(prior)

        reduced_prior = utils.obj_array(num_factors)

        factors_to_remove = []
        for f, s_i in enumerate(prior): # loop over factors (or modalities)
            
            ns = len(s_i)
            levels_to_keep = list(set(range(ns)) - set(levels_to_remove[f]))
            if len(levels_to_keep) == 0:
                print(f'Warning... removing ALL levels of factor {f} - i.e. the whole hidden state factor is being removed\n')
                factors_to_remove.append(f)
            else:
                reduced_prior[f] = utils.norm_dist(s_i[levels_to_keep])

        if len(factors_to_remove) > 0:
            factors_to_keep = list(set(range(num_factors)) - set(factors_to_remove))
            reduced_prior = reduced_prior[factors_to_keep]

    else: # in case of one hidden state factor

        assert all([type(level_i) == int for level_i in levels_to_remove])

        ns = len(prior)
        levels_to_keep = list(set(range(ns)) - set(levels_to_remove))

        reduced_prior = utils.norm_dist(prior[levels_to_keep])

    return reduced_prior

def prune_A(A, obs_levels_to_prune, state_levels_to_prune, dirichlet = False):
    """
    Function for pruning a observation likelihood (with potentially multiple hidden state factors)
    Arguments:
    =========
    `A` [np.ndarray or numpy object array]: The observation model or mapping from hidden states to observations (A matrix) of the generative model.
    `obs_levels_to_prune` [list]: a list of the observation levels to remove. If the likelihood in question has multiple observation modalities, 
        then this will be a list of lists, where each sub-list within `obs_levels_to_prune` will contain the observation levels to prune for a particular observation modality 
    `state_levels_to_prune` [list]: a list of the hidden state levels to remove (this will be the same across modalities, so it won't matter whether the `likelihood` array that's
    passed in is an object array or not)
    `dirichlet` [bool]: a flag telling whether the input array is a Dirichlet and therefore should not be normalized at the end. @TODO: Instead, the dirichlet parameters from the
                        pruned columns should somehow be re-distributed among the remaining columns
    Returns:
    =========
    `reduced_A` [np.ndarray or numpy object array]: The observation model, after pruning, which lacks the observation or hidden state levels given by the arguments `obs_levels_to_prune` and `state_levels_to_prune`
    """

    columns_to_keep_list = []
    if utils.is_arr_of_arr(A):
        num_states = A[0].shape[1:]
        for f, ns in enumerate(num_states):
            indices_f = np.array( list(set(range(ns)) - set(state_levels_to_prune[f])), dtype = np.intp)
            columns_to_keep_list.append(indices_f)
    else:
        num_states = A.shape[1]
        indices = np.array( list(set(range(num_states)) - set(state_levels_to_prune)), dtype = np.intp )
        columns_to_keep_list.append(indices)

    if utils.is_arr_of_arr(A): # in case of multiple observation modality

        assert all([type(o_m_levels) == list for o_m_levels in obs_levels_to_prune])

        num_modalities = len(A)

        reduced_A = utils.obj_array(num_modalities)
        
        for m, A_i in enumerate(A): # loop over modalities
            
            no = A_i.shape[0]
            rows_to_keep = np.array(list(set(range(no)) - set(obs_levels_to_prune[m])), dtype = np.intp)
            
            reduced_A[m] = A_i[np.ix_(rows_to_keep, *columns_to_keep_list)]
        if not dirichlet:    
            reduced_A = utils.norm_dist_obj_arr(reduced_A)
        else:
            raise(NotImplementedError("Need to figure out how to re-distribute concentration parameters from pruned rows/columns, across remaining rows/columns"))
    else: # in case of one observation modality

        assert all([type(o_levels_i) == int for o_levels_i in obs_levels_to_prune])

        no = A.shape[0]
        rows_to_keep = np.array(list(set(range(no)) - set(obs_levels_to_prune)), dtype = np.intp)
            
        reduced_A = A[np.ix_(rows_to_keep, *columns_to_keep_list)]

        if not dirichlet:
            reduced_A = utils.norm_dist(reduced_A)
        else:
            raise(NotImplementedError("Need to figure out how to re-distribute concentration parameters from pruned rows/columns, across remaining rows/columns"))

    return reduced_A

def prune_B(B, state_levels_to_prune, action_levels_to_prune, dirichlet = False):
    """
    Function for pruning a transition likelihood (with potentially multiple hidden state factors)
    Arguments:
    =========
    `B` [np.ndarray or numpy object array]: The transition model or mapping from hidden states at time t to hidden states at time t+1 (B matrix) of the generative model.
    `state_levels_to_prune` [list]: a list of the hidden state levels to remove, one per hidden state factor. Or just a list of integers if there's one hidden state factor
    `action_levels_to_prune` [list]: a list of the action levels to remove, one per control state factor. Or just a list of integers if there's one control state factor
    `dirichlet` [bool]: a flag telling whether the input array is a Dirichlet and therefore should not be normalized at the end. @TODO: Instead, the dirichlet parameters from the
                        pruned columns should somehow be re-distributed among the remaining columns
    Returns:
    =========
    `reduced_B` [np.ndarray or numpy object array]: The transition model, after pruning, which lacks the hidden state levels /action levels given by the arguments `state_levels_to_prune` and `action_levels_to_prune`
    """

    slices_to_keep_list = []

    if utils.is_arr_of_arr(B):

        num_controls = [B_arr.shape[2] for _, B_arr in enumerate(B)]

        for c, nc in enumerate(num_controls):
            indices_c = np.array( list(set(range(nc)) - set(action_levels_to_prune[c])), dtype = np.intp)
            slices_to_keep_list.append(indices_c)
    else:
        num_controls = B.shape[2]
        slices_to_keep = np.array( list(set(range(num_controls)) - set(action_levels_to_prune)), dtype = np.intp )

    if utils.is_arr_of_arr(B): # in case of multiple hidden state factors

        assert all([type(ns_f_levels) == list for ns_f_levels in state_levels_to_prune])

        num_factors = len(B)

        reduced_B = utils.obj_array(num_factors)
        
        for f, B_f in enumerate(B): # loop over modalities
            
            ns = B_f.shape[0]
            states_to_keep = np.array(list(set(range(ns)) - set(state_levels_to_prune[f])), dtype = np.intp)
            
            reduced_B[f] = B_f[np.ix_(states_to_keep, states_to_keep, slices_to_keep_list[f])]

        if not dirichlet:    
            reduced_B = utils.norm_dist_obj_arr(reduced_B)
        else:
            raise(NotImplementedError("Need to figure out how to re-distribute concentration parameters from pruned rows/columns, across remaining rows/columns"))

    else: # in case of one hidden state factor

        assert all([type(state_level_i) == int for state_level_i in state_levels_to_prune])

        ns = B.shape[0]
        states_to_keep = np.array(list(set(range(ns)) - set(state_levels_to_prune)), dtype = np.intp)
            
        reduced_B = B[np.ix_(states_to_keep, states_to_keep, slices_to_keep)]

        if not dirichlet:
            reduced_B = utils.norm_dist(reduced_B)
        else:
            raise(NotImplementedError("Need to figure out how to re-distribute concentration parameters from pruned rows/columns, across remaining rows/columns"))

    return reduced_B
