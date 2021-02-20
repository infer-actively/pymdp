#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Marginal message passing
__author__: Conor Heins, Beren Millidge, Alexander Tschantz, Brennan Klein
"""

import numpy as np

from pymdp.core.utils import to_arr_of_arr, get_model_dimensions, obj_array, obj_array_zeros, obj_array_uniform
from pymdp.core.maths import spm_dot, spm_norm, softmax, calc_free_energy, spm_log
import copy


def run_mmp(
    lh_seq, B, policy, prev_actions=None, prior=None, num_iter=10, grad_descent=False, tau=0.25, last_timestep = False, save_vfe_seq=False):
    """
    Marginal message passing scheme for updating posterior beliefs about multi-factor hidden states over time, 
    conditioned on a particular policy.
    Parameters:
    --------------
    `lh_seq`[numpy object array]:
        Likelihoods of hidden state factors given a sequence of observations over time. This is logged beforehand
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
        Number of variational iterations
    `grad_descent` [Bool]:
        Flag for whether to use gradient descent (predictive coding style)
    `tau` [Float]:
        Decay constant for use in `grad_descent` version
    `last_timestep` [Bool]:
        Flag for whether we are at the last timestep of belief updating
    `save_vfe_seq` [Bool]:
        Flag for whether to save the sequence of variational free energies over time (for this policy). If `False`, then VFE is integrated across time/iterations.
    Returns:
    --------------
    `qs_seq`[list]: the sequence of beliefs about the different hidden state factors over time, one multi-factor posterior belief per timestep in `infer_len`
    `F`[Float or list, depending on setting of save_vfe_seq]
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
    B = to_arr_of_arr(B)

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

    # full policy
    if prev_actions is None:
        prev_actions = np.zeros((past_len, policy.shape[1]))
    policy = np.vstack((prev_actions, policy))

    # initialise variational free energy of policy (accumulated over time)

    if save_vfe_seq:
        F = []
        F.append(0.0)
    else:
        F = 0.0

    for itr in range(num_iter):
        for t in range(infer_len):
            for f in range(num_factors):
                # likelihood
                if t < past_len:
                    lnA = spm_log(spm_dot(lh_seq[t], qs_seq[t], [f]))
                else:
                    lnA = np.zeros(num_states[f])
                
                # past message
                if t == 0:
                    lnB_past = spm_log(prior[f])
                else:
                    past_msg = B[f][:, :, int(policy[t - 1, f])].dot(qs_seq[t - 1][f])
                    lnB_past = spm_log(past_msg)

                # future message
                if t >= future_cutoff:
                    lnB_future = qs_T[f]
                else:
                    future_msg = trans_B[f][:, :, int(policy[t, f])].dot(qs_seq[t + 1][f])
                    lnB_future = spm_log(future_msg)
                
                # inference
                if grad_descent:
                    lnqs = spm_log(qs_seq[t][f])
                    coeff = 1 if (t >= future_cutoff) else 2
                    err = (coeff * lnA + lnB_past + lnB_future) - coeff * lnqs
                    err -= err.mean()
                    lnqs = lnqs + tau * err
                    qs_seq[t][f] = softmax(lnqs)
                    if (t == 0) or (t == (infer_len-1)):
                        F += + 0.5*lnqs.dot(0.5*err)
                    else:
                        F += lnqs.dot(0.5*(err - (num_factors - 1)*lnA/num_factors)) # @NOTE: not sure why Karl does this in SPM_MDP_VB_X, we should look into this
                else:
                    qs_seq[t][f] = softmax(lnA + lnB_past + lnB_future)
            
            if not grad_descent:

                if save_vfe_seq:
                    if t < past_len:
                        F.append(F[-1] + calc_free_energy(qs_seq[t], prior, num_factors, likelihood = spm_log(lh_seq[t]) )[0] )
                    else:
                        F.append(F[-1] + calc_free_energy(qs_seq[t], prior, num_factors)[0] )
                else:
                    if t < past_len:
                        F += calc_free_energy(qs_seq[t], prior, num_factors, likelihood = spm_log(lh_seq[t]) )
                    else:
                        F += calc_free_energy(qs_seq[t], prior, num_factors)

    return qs_seq, F
