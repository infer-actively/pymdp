#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

import numpy as np
from .maths import multidimensional_outer
from jax.tree_util import tree_map
from jax import vmap
import jax.numpy as jnp

def update_obs_likelihood_dirichlet_m(pA_m, obs_m, qs, dependencies_m, lr=1.0):
    """ JAX version of ``pymdp.learning.update_obs_likelihood_dirichlet_m`` """
    # pA_m - parameters of the dirichlet from the prior
    # pA_m.shape = (no_m x num_states[k] x num_states[j] x ... x num_states[n]) where (k, j, n) are indices of the hidden state factors that are parents of modality m

    # \alpha^{*} = \alpha_{0} + \kappa * \sum_{t=t_begin}^{t=T} o_{m,t} \otimes \mathbf{s}_{f \in parents(m), t}

    # \alpha^{*} is the VFE-minimizing solution for the parameters of q(A)
    # \alpha_{0} are the Dirichlet parameters of p(A)
    # o_{m,t} = observation (one-hot vector) of modality m at time t
    # \mathbf{s}_{f \in parents(m), t} = categorical parameters of marginal posteriors over hidden state factors that are parents of modality m, at time t
    # \otimes is a multidimensional outer product, not just a outer product of two vectors
    # \kappa is an optional learning rate

    relevant_factors = tree_map(lambda f_idx: qs[f_idx], dependencies_m)

    dfda = vmap(multidimensional_outer)([obs_m] + relevant_factors).sum(axis=0)

    return pA_m + lr * dfda
    
def update_obs_likelihood_dirichlet(pA, obs, qs, A_dependencies, lr=1.0):
    """ JAX version of ``pymdp.learning.update_obs_likelihood_dirichlet`` """

    update_A_fn = lambda pA_m, obs_m, dependencies_m: update_obs_likelihood_dirichlet_m(pA_m, obs_m, qs, dependencies_m, lr=lr)
    qA = tree_map(update_A_fn, pA, obs, A_dependencies)

    return qA

def update_state_likelihood_dirichlet_f(pB_f, actions_f, current_qs, qs_seq, dependencies_f, lr=1.0):
    """ JAX version of ``pymdp.learning.update_state_likelihood_dirichlet_f`` """
    # pB_f - parameters of the dirichlet from the prior
    # pB_f.shape = (num_states[f] x num_states[f] x num_actions[f]) where f is the index of the hidden state factor

    # \alpha^{*} = \alpha_{0} + \kappa * \sum_{t=t_begin}^{t=T} \mathbf{s}_{f, t} \otimes \mathbf{s}_{f, t-1} \otimes \mathbf{a}_{f, t-1}

    # \alpha^{*} is the VFE-minimizing solution for the parameters of q(B)
    # \alpha_{0} are the Dirichlet parameters of p(B)
    # \mathbf{s}_{f, t} = categorical parameters of marginal posteriors over hidden state factor f, at time t
    # \mathbf{a}_{f, t-1} = categorical parameters of marginal posteriors over control factor f, at time t-1
    # \otimes is a multidimensional outer product, not just a outer product of two vectors
    # \kappa is an optional learning rate

    past_qs = tree_map(lambda f_idx: qs_seq[f_idx][:-1], dependencies_f)
    dfdb = vmap(multidimensional_outer)([current_qs[1:]] + past_qs + [actions_f]).sum(axis=0)
    qB_f = pB_f + lr * dfdb

    return qB_f

def update_state_likelihood_dirichlet(pB, beliefs, actions_onehot, B_dependencies, lr=1.0):

    update_B_f_fn = lambda pB_f, action_f, qs_f, dependencies_f: update_state_likelihood_dirichlet_f(pB_f, action_f, qs_f, beliefs, dependencies_f, lr=lr)    
    qB = tree_map(update_B_f_fn, pB, actions_onehot, beliefs, B_dependencies)

    return qB
    

def update_state_prior_dirichlet(
    pD, qs, lr=1.0, factors="all"
):
    """
    Update Dirichlet parameters of the initial hidden state distribution 
    (prior beliefs about hidden states at the beginning of the inference window).

    Parameters
    -----------
    pD: ``numpy.ndarray`` of dtype object
        Prior Dirichlet parameters over initial hidden state prior (same shape as ``qs``)
    qs: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
        Marginal posterior beliefs over hidden states at current timepoint
    lr: float, default ``1.0``
        Learning rate, scale of the Dirichlet pseudo-count update.
    factors: ``list``, default "all"
        Indices (ranging from 0 to ``n_factors - 1``) of the hidden state factors to include 
        in learning. Defaults to "all", meaning that factor-specific sub-vectors of ``pD``
        are all updated using the corresponding hidden state distributions.
    
    Returns
    -----------
    qD: ``numpy.ndarray`` of dtype object
        Posterior Dirichlet parameters over initial hidden state prior (same shape as ``qs``), after having updated it with state beliefs.
    """

    num_factors = len(pD)

    qD = copy.deepcopy(pD)
   
    if factors == "all":
        factors = list(range(num_factors))

    for factor in factors:
        idx = pD[factor] > 0 # only update those state level indices that have some prior probability
        qD[factor][idx] += (lr * qs[factor][idx])
       
    return qD

def _prune_prior(prior, levels_to_remove, dirichlet = False):
    """
    Function for pruning a prior Categorical distribution (e.g. C, D)

    Parameters
    -----------
    prior: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
        The vector(s) containing the priors over hidden states of a generative model, e.g. the prior over hidden states (``D`` vector). 
    levels_to_remove: ``list`` of ``int``, ``list`` of ``list``
        A ``list`` of the levels (indices of the support) to remove. If the prior in question has multiple hidden state factors / multiple observation modalities, 
        then this will be a ``list`` of ``list``, where each sub-list within ``levels_to_remove`` will contain the levels to prune for a particular hidden state factor or modality 
    dirichlet: ``Bool``, default ``False``
        A Boolean flag indicating whether the input vector(s) is/are a Dirichlet distribution, and therefore should not be normalized at the end. 
        @TODO: Instead, the dirichlet parameters from the pruned levels should somehow be re-distributed among the remaining levels

    Returns
    -----------
    reduced_prior: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
        The prior vector(s), after pruning, that lacks the hidden state or modality levels indexed by ``levels_to_remove``
    """

    if utils.is_obj_array(prior): # in case of multiple hidden state factors

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
                if not dirichlet:
                    reduced_prior[f] = utils.norm_dist(s_i[levels_to_keep])
                else:
                    raise(NotImplementedError("Need to figure out how to re-distribute concentration parameters from pruned levels, across remaining levels"))


        if len(factors_to_remove) > 0:
            factors_to_keep = list(set(range(num_factors)) - set(factors_to_remove))
            reduced_prior = reduced_prior[factors_to_keep]

    else: # in case of one hidden state factor

        assert all([type(level_i) == int for level_i in levels_to_remove])

        ns = len(prior)
        levels_to_keep = list(set(range(ns)) - set(levels_to_remove))

        if not dirichlet:
            reduced_prior = utils.norm_dist(prior[levels_to_keep])
        else:
            raise(NotImplementedError("Need to figure out how to re-distribute concentration parameters from pruned levels, across remaining levels"))

    return reduced_prior

def _prune_A(A, obs_levels_to_prune, state_levels_to_prune, dirichlet = False):
    """
    Function for pruning a observation likelihood model (with potentially multiple hidden state factors)
    :meta private:
    Parameters
    -----------
    A: ``numpy.ndarray`` with ``ndim >= 2``, or ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    obs_levels_to_prune: ``list`` of int or ``list`` of ``list``: 
        A ``list`` of the observation levels to remove. If the likelihood in question has multiple observation modalities, 
        then this will be a ``list`` of ``list``, where each sub-list within ``obs_levels_to_prune`` will contain the observation levels 
        to remove for a particular observation modality 
    state_levels_to_prune: ``list`` of ``int``
        A ``list`` of the hidden state levels to remove (this will be the same across modalities)
    dirichlet: ``Bool``, default ``False``
        A Boolean flag indicating whether the input array(s) is/are a Dirichlet distribution, and therefore should not be normalized at the end. 
        @TODO: Instead, the dirichlet parameters from the pruned columns should somehow be re-distributed among the remaining columns

    Returns
    -----------
    reduced_A: ``numpy.ndarray`` with ndim >= 2, or ``numpy.ndarray ``of dtype object
        The observation model, after pruning, which lacks the observation or hidden state levels given by the arguments ``obs_levels_to_prune`` and ``state_levels_to_prune``
    """

    columns_to_keep_list = []
    if utils.is_obj_array(A):
        num_states = A[0].shape[1:]
        for f, ns in enumerate(num_states):
            indices_f = np.array( list(set(range(ns)) - set(state_levels_to_prune[f])), dtype = np.intp)
            columns_to_keep_list.append(indices_f)
    else:
        num_states = A.shape[1]
        indices = np.array( list(set(range(num_states)) - set(state_levels_to_prune)), dtype = np.intp )
        columns_to_keep_list.append(indices)

    if utils.is_obj_array(A): # in case of multiple observation modality

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

def _prune_B(B, state_levels_to_prune, action_levels_to_prune, dirichlet = False):
    """
    Function for pruning a transition likelihood model (with potentially multiple hidden state factors)

    Parameters
    -----------
    B: ``numpy.ndarray`` of ``ndim == 3`` or ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at `t` to hidden states at `t+1`, given some control state `u`.
        Each element B[f] of this object array stores a 3-D tensor for hidden state factor `f`, whose entries `B[f][s, v, u] store the probability
        of hidden state level `s` at the current time, given hidden state level `v` and action `u` at the previous time.
    state_levels_to_prune: ``list`` of ``int`` or ``list`` of ``list`` 
        A ``list`` of the state levels to remove. If the likelihood in question has multiple hidden state factors, 
        then this will be a ``list`` of ``list``, where each sub-list within ``state_levels_to_prune`` will contain the state levels 
        to remove for a particular hidden state factor 
    action_levels_to_prune: ``list`` of ``int`` or ``list`` of ``list`` 
        A ``list`` of the control state or action levels to remove. If the likelihood in question has multiple control state factors, 
        then this will be a ``list`` of ``list``, where each sub-list within ``action_levels_to_prune`` will contain the control state levels 
        to remove for a particular control state factor 
    dirichlet: ``Bool``, default ``False``
        A Boolean flag indicating whether the input array(s) is/are a Dirichlet distribution, and therefore should not be normalized at the end. 
        @TODO: Instead, the dirichlet parameters from the pruned rows/columns should somehow be re-distributed among the remaining rows/columns

    Returns
    -----------
    reduced_B: ``numpy.ndarray`` of `ndim == 3` or ``numpy.ndarray`` of dtype object
        The transition model, after pruning, which lacks the hidden state levels/action levels given by the arguments ``state_levels_to_prune`` and ``action_levels_to_prune``
    """

    slices_to_keep_list = []

    if utils.is_obj_array(B):

        num_controls = [B_arr.shape[2] for _, B_arr in enumerate(B)]

        for c, nc in enumerate(num_controls):
            indices_c = np.array( list(set(range(nc)) - set(action_levels_to_prune[c])), dtype = np.intp)
            slices_to_keep_list.append(indices_c)
    else:
        num_controls = B.shape[2]
        slices_to_keep = np.array( list(set(range(num_controls)) - set(action_levels_to_prune)), dtype = np.intp )

    if utils.is_obj_array(B): # in case of multiple hidden state factors

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
