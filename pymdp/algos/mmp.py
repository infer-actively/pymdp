#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Marginal message passing
__author__: Conor Heins, Beren Millidge, Alexander Tschantz, Brennan Klein
"""

import numpy as np

from pymdp.utils import to_arr_of_arr, get_model_dimensions, obj_array, obj_array_zeros, obj_array_uniform
from pymdp.maths import spm_dot, spm_norm, softmax, calc_free_energy, spm_log_single
import copy


def run_mmp(
    lh_seq, B, policy, prev_actions=None, prior=None, num_iter=10, grad_descent=True, tau=0.25, last_timestep = False):
    """
    Marginal message passing scheme for updating posterior beliefs about multi-factor hidden states over time, 
    conditioned on a particular policy.
    Parameters:
    --------------
    `lh_seq`[numpy object array]:
        (log) Likelihoods of hidden state factors given a sequence of observations over time. This is assumed to already be log-transformed.
    `B`[numpy object array]:
        Transition likelihood of the generative model, mapping from hidden states at T to hidden states at T+1. One B matrix per modality (e.g. `B[f]` corresponds to f-th factor's B matrix)
        This is used in inference to compute the 'forward' and 'backward' messages conveyed between beliefs about temporally-adjacent timepoints.
    `policy` [2-D numpy.ndarray]:
        Matrix of shape (policy_len, num_control_factors) that indicates the indices of each action (control state index) upon timestep t and control_factor f in the element `policy[t,f]` for a given policy.
    `prev_actions` [None or 2-D numpy.ndarray]:
        If provided, should be a matrix of previous actions of shape (infer_len, num_control_factors) that indicates the indices of each action (control state index) taken in the past (up until the current timestep).
    `prior`[None or numpy object array]:
        If provided, this a numpy object array with one sub-array per hidden state factor, that stores the prior beliefs about initial states (at t = 0, relative to `infer_len`). IF None, this defaults
        to a flat (uninformative) prior over hidden states.
    `num_iter`[Int]:
        Number of variational iterations.
    `grad_descent` [Bool]:
        Flag for whether to use gradient descent (predictive coding style)
    `tau` [Float]:
        Decay constant for use in `grad_descent` version
    `last_timestep` [Bool]:
        Flag for whether we are at the last timestep of belief updating
    Returns:
    --------------
    `qs_seq`[list]: the sequence of beliefs about the different hidden state factors over time, one multi-factor posterior belief per timestep in `infer_len`
    `F`[Float]: variational free energy for the given policy 
    """

    # window
    past_len = len(lh_seq)
    future_len = policy.shape[0]

    if last_timestep:
        infer_len = past_len + future_len - 1
    else:
        infer_len = past_len + future_len
    
    future_cutoff = past_len + future_len - 2

    # dimensions
    _, num_states, _, num_factors = get_model_dimensions(A=None, B=B)

    # beliefs
    qs_seq = obj_array(infer_len)
    for t in range(infer_len):
        qs_seq[t] = obj_array_uniform(num_states)

    # last message
    qs_T = obj_array_zeros(num_states)

    # prior
    if prior is None:
        prior = obj_array_uniform(num_states)

    # transposed transition
    trans_B = obj_array(num_factors)
        
    for f in range(num_factors):
        trans_B[f] = spm_norm(np.swapaxes(B[f],0,1))

    if prev_actions is not None:
        policy = np.vstack((prev_actions, policy))

    for itr in range(num_iter):
        F = 0.0 # reset variational free energy (accumulated over time and factors, but reset per iteration)
        for t in range(infer_len):
            for f in range(num_factors):
                # likelihood
                if t < past_len:
                    lnA = spm_log_single(spm_dot(lh_seq[t], qs_seq[t], [f]))
                else:
                    lnA = np.zeros(num_states[f])
                
                # past message
                if t == 0:
                    lnB_past = spm_log_single(prior[f])
                else:
                    past_msg = B[f][:, :, int(policy[t - 1, f])].dot(qs_seq[t - 1][f])
                    lnB_past = spm_log_single(past_msg)

                # future message
                if t >= future_cutoff:
                    lnB_future = qs_T[f]
                else:
                    future_msg = trans_B[f][:, :, int(policy[t, f])].dot(qs_seq[t + 1][f])
                    lnB_future = spm_log_single(future_msg)
                
                # inference
                if grad_descent:
                    sx = qs_seq[t][f] # save this as a separate variable so that it can be used in VFE computation
                    lnqs = spm_log_single(sx)
                    coeff = 1 if (t >= future_cutoff) else 2
                    err = (coeff * lnA + lnB_past + lnB_future) - coeff * lnqs
                    lnqs = lnqs + tau * (err - err.mean())
                    qs_seq[t][f] = softmax(lnqs)
                    if (t == 0) or (t == (infer_len-1)):
                        F += sx.dot(0.5*err)
                    else:
                        F += sx.dot(0.5*(err - (num_factors - 1)*lnA/num_factors)) # @NOTE: not sure why Karl does this in SPM_MDP_VB_X, we should look into this
                else:
                    qs_seq[t][f] = softmax(lnA + lnB_past + lnB_future)
            
            if not grad_descent:

                if t < past_len:
                    F += calc_free_energy(qs_seq[t], prior, num_factors, likelihood = spm_log_single(lh_seq[t]) )
                else:
                    F += calc_free_energy(qs_seq[t], prior, num_factors)

    return qs_seq, F

def run_mmp_testing(
    lh_seq, B, policy, prev_actions=None, prior=None, num_iter=10, grad_descent=True, tau=0.25, last_timestep = False):
    """
    Marginal message passing scheme for updating posterior beliefs about multi-factor hidden states over time, 
    conditioned on a particular policy.
    Parameters:
    --------------
    `lh_seq`[numpy object array]:
        (log) Likelihoods of hidden state factors given a sequence of observations over time. This is assumed to already be log-transformed.
    `B`[numpy object array]:
        Transition likelihood of the generative model, mapping from hidden states at T to hidden states at T+1. One B matrix per modality (e.g. `B[f]` corresponds to f-th factor's B matrix)
        This is used in inference to compute the 'forward' and 'backward' messages conveyed between beliefs about temporally-adjacent timepoints.
    `policy` [2-D numpy.ndarray]:
        Matrix of shape (policy_len, num_control_factors) that indicates the indices of each action (control state index) upon timestep t and control_factor f in the element `policy[t,f]` for a given policy.
    `prev_actions` [None or 2-D numpy.ndarray]:
        If provided, should be a matrix of previous actions of shape (infer_len, num_control_factors) taht indicates the indices of each action (control state index) taken in the past (up until the current timestep).
    `prior`[None or numpy object array]:
        If provided, this a numpy object array with one sub-array per hidden state factor, that stores the prior beliefs about initial states (at t = 0, relative to `infer_len`).
    `num_iter`[Int]:
        Number of variational iterations.
    `grad_descent` [Bool]:
        Flag for whether to use gradient descent (predictive coding style)
    `tau` [Float]:
        Decay constant for use in `grad_descent` version
    `last_timestep` [Bool]:
        Flag for whether we are at the last timestep of belief updating
    Returns:
    --------------
    `qs_seq`[list]: the sequence of beliefs about the different hidden state factors over time, one multi-factor posterior belief per timestep in `infer_len`
    `F`[Float]: variational free energy for the given policy 
    `xn` [list]: the sequence of beliefs as they're computed across iterations of marginal message passing
    `vn` [list]: the sequence of prediction errors as they're computed across iterations of marginal message passing
    """

    # window
    past_len = len(lh_seq)
    future_len = policy.shape[0]

    if last_timestep:
        infer_len = past_len + future_len - 1
    else:
        infer_len = past_len + future_len
    
    future_cutoff = past_len + future_len - 2

    # dimensions
    _, num_states, _, num_factors = get_model_dimensions(A=None, B=B)

    # beliefs
    qs_seq = obj_array(infer_len)
    for t in range(infer_len):
        qs_seq[t] = obj_array_uniform(num_states)

    # last message
    qs_T = obj_array_zeros(num_states)

    # prior
    if prior is None:
        prior = obj_array_uniform(num_states)

    # transposed transition
    trans_B = obj_array(num_factors)
        
    for f in range(num_factors):
        trans_B[f] = spm_norm(np.swapaxes(B[f],0,1))

    if prev_actions is not None:
        policy = np.vstack((prev_actions, policy))

    xn = [] # list for storing beliefs across iterations
    vn = [] # list for storing prediction errors across iterations

    shape_list = [ [num_states[f], infer_len] for f in range(num_factors) ]
    
    for itr in range(num_iter):

        xn_itr_all_factors = obj_array_zeros(shape_list) # temporary cache for storing beliefs across different hidden state factors, for a fixed iteration of the belief updating scheme
        vn_itr_all_factors = obj_array_zeros(shape_list) # temporary cache for storing prediction errors across different hidden state factors, for a fixed iteration of the belief updating scheme

        F = 0.0 # reset variational free energy (accumulated over time and factors, but reset per iteration)
        for t in range(infer_len):

            if t == (infer_len - 1):
                debug_flag = True

            for f in range(num_factors):
                # likelihood
                if t < past_len:
                    # if itr == 0:
                    #     print(f'obs from timestep {t}\n')
                    lnA = spm_log_single(spm_dot(lh_seq[t], qs_seq[t], [f]))
                else:
                    lnA = np.zeros(num_states[f])
                
                # past message
                if t == 0:
                    lnB_past = spm_log_single(prior[f])
                else:
                    past_msg = B[f][:, :, int(policy[t - 1, f])].dot(qs_seq[t - 1][f])
                    lnB_past = spm_log_single(past_msg)

                # future message
                if t >= future_cutoff:
                    lnB_future = qs_T[f]
                else:
                    future_msg = trans_B[f][:, :, int(policy[t, f])].dot(qs_seq[t + 1][f])
                    lnB_future = spm_log_single(future_msg)

                # inference
                if grad_descent:
                    sx = qs_seq[t][f] # save this as a separate variable so that it can be used in VFE computation
                    lnqs = spm_log_single(sx)
                    coeff = 1 if (t >= future_cutoff) else 2
                    err = (coeff * lnA + lnB_past + lnB_future) - coeff * lnqs
                    vn_tmp = err - err.mean()
                    lnqs = lnqs + tau * vn_tmp
                    qs_seq[t][f] = softmax(lnqs)
                    if (t == 0) or (t == (infer_len-1)):
                        F += sx.dot(0.5*err)
                    else:
                        F += sx.dot(0.5*(err - (num_factors - 1)*lnA/num_factors)) # @NOTE: not sure why Karl does this in SPM_MDP_VB_X, we should look into this
                    
                    xn_itr_all_factors[f][:,t] = np.copy(qs_seq[t][f])
                    vn_itr_all_factors[f][:,t] = np.copy(vn_tmp)

                else:
                    qs_seq[t][f] = softmax(lnA + lnB_past + lnB_future)
            
            if not grad_descent:

                if t < past_len:
                    F += calc_free_energy(qs_seq[t], prior, num_factors, likelihood = spm_log_single(lh_seq[t]) )
                else:
                    F += calc_free_energy(qs_seq[t], prior, num_factors)
        xn.append(xn_itr_all_factors)
        vn.append(vn_itr_all_factors)

    return qs_seq, F, xn, vn
