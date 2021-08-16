#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

""" Functions for performing variational inference on hidden states 

__author__: Conor Heins, Beren Millidge, Alexander Tschantz, Brennan Klein
"""

import numpy as np
import sys
import pathlib

from pymdp.maths import spm_dot, get_joint_likelihood, spm_norm, softmax, calc_free_energy
from pymdp import utils


def run_mmp_old(
    A,
    B,
    obs_t,
    policy,
    curr_t,
    t_horizon,
    T,
    qs_bma=None,
    prior=None,
    num_iter=10,
    dF=1.0,
    dF_tol=0.001,
    previous_actions=None,
    use_gradient_descent=False,
    tau=0.25,
):
    """
    Optimise marginal posterior beliefs about hidden states using marginal message-passing scheme (MMP) developed
    by Thomas Parr and colleagues, see https://github.com/tejparr/nmpassing
   
    Parameters
    ----------
    - 'A' [numpy nd.array (matrix or tensor or array-of-arrays)]:
        Observation likelihood of the generative model, mapping from hidden states to observations. 
        Used in inference to get the likelihood of an observation, under different hidden state configurations.
    - 'B' [numpy.ndarray (tensor or array-of-arrays)]:
        Transition likelihood of the generative model, mapping from hidden states at t to hidden states at t+1.
        Used in inference to get expected future (or past) hidden states, given past (or future) hidden 
        states (or expectations thereof).
    - 'obs_t' [list of length t_horizon of numpy 1D array or array of arrays (with 1D numpy array entries)]:
        Sequence of observations sampled from beginning of time horizon the current timestep t. 
        The first observation (the start of the time horizon) is either the first timestep of the generative 
        process or the first timestep of the policy horizon (whichever is closer to 'curr_t' in time).
        The observations over time are stored as a list of numpy arrays, where in case of multi-modalities 
        each numpy array is an array-of-arrays, with one 1D numpy.ndarray for each modality. 
        In the case of a single modality, each observation is a single 1D numpy.ndarray.
    - 'policy' [2D np.ndarray]:
        Array of actions constituting a single policy. Policy is a shape 
        (n_steps, n_control_factors) numpy.ndarray, the values of which indicate actions along a given control 
        factor (column index) at a given timestep (row index).
    - 'curr_t' [int]:
        Current timestep (relative to the 'absolute' time of the generative process).
    - 't_horizon'[int]:
        Temporal horizon of inference for states and policies.
    - 'T' [int]:
        Temporal horizon of the generative process (absolute time)
    - `qs_bma` [numpy 1D array, array of arrays (with 1D numpy array entries) or None]:
    - 'prior' [numpy 1D array, array of arrays (with 1D numpy array entries) or None]:
        Prior beliefs of the agent at the beginning of the time horizon, to be integrated 
        with the marginal likelihood to obtain posterior at the first timestep.
        If absent, prior is set to be a uniform distribution over hidden states (identical to the 
        initialisation of the posterior.
    -'num_iter' [int]:
        Number of variational iterations to run. (optional)
    -'dF' [float]:
        Starting free energy gradient (dF/dt) before updating in the course of gradient descent. (optional)
    -'dF_tol' [float]:
        Threshold value of the gradient of the variational free energy (dF/dt), to be checked 
        at each iteration. If dF <= dF_tol, the iterations are halted pre-emptively and the final 
        marginal posterior belief(s) is(are) returned.  (optional)
    -'previous_actions' [numpy.ndarray with shape (num_steps, n_control_factors) or None]:
        Array of previous actions, which can be used to constrain the 'past' messages in inference 
        to only consider states of affairs that were possible under actions that are known to have been taken. 
        The first dimension of previous-arrays (previous_actions.shape[0]) encodes how far back in time the agent is 
        considering. The first timestep of this either corresponds to either the first timestep of the generative 
        process or the first timestep of the policy horizon (whichever is sooner in time).  (optional)
    -'use_gradient_descent' [bool]:
        Flag to indicate whether to use gradient descent to optimise posterior beliefs.
    -'tau' [float]:
        Learning rate for gradient descent (only used if use_gradient_descent is True)
 
  
    Returns
    ----------
    -'qs' [list of length T of numpy 1D arrays or array of arrays (with 1D numpy array entries):
        Marginal posterior beliefs over hidden states (single- or multi-factor) achieved 
        via marginal message pasing
    -'qss' [list of lists of length T of numpy 1D arrays or array of arrays (with 1D numpy array entries):
        Marginal posterior beliefs about hidden states (single- or multi-factor) held at 
        each timepoint, *about* each timepoint of the observation
        sequence
    -'F' [2D np.ndarray]:
        Variational free energy of beliefs about hidden states, indexed by time point and variational iteration
    -'F_pol' [float]:
        Total free energy of the policy under consideration.
    """

    # get temporal window for inference
    min_time = max(0, curr_t - t_horizon)
    max_time = min(T, curr_t + t_horizon)
    window_idxs = np.array([t for t in range(min_time, max_time)])
    window_len = len(window_idxs)
    # TODO: needs a better name - the point at which we ignore future messages
    future_cutoff = window_len - 1
    inference_len = window_len + 1
    obs_seq_len = len(obs_t)

    # get relevant observations, given our current time point
    if curr_t == 0:
        obs_range = [0]
    else:
        min_obs_idx = max(0, curr_t - t_horizon)
        max_obs_idx = curr_t + 1
        obs_range = range(min_obs_idx, max_obs_idx)

    # get model dimensions
    # TODO: make a general function in `utils` for extracting model dimensions
    if utils.is_arr_of_arr(obs_t[0]):
        num_obs = [obs.shape[0] for obs in obs_t[0]]
    else:
        num_obs = [obs_t[0].shape[0]]

    if utils.is_arr_of_arr(B):
        num_states = [b.shape[0] for b in B]
    else:
        num_states = [B[0].shape[0]]
        B = utils.to_arr_of_arr(B)

    num_modalities = len(num_obs)
    num_factors = len(num_states)

    """
    =========== Step 1 ===========
        Calculate likelihood
        Loop over modalities and use assumption of independence among observation modalities
        to combine each modality-specific likelihood into a single joint likelihood over hidden states 
    """

    # likelihood of observations under configurations of hidden states (over time)
    likelihood = np.empty(len(obs_range), dtype=object)
    for t, obs in enumerate(obs_range):
        # likelihood_t = np.ones(tuple(num_states))

        # if num_modalities == 1:
        #     likelihood_t *= spm_dot(A[0], obs_t[obs], obs_mode=True)
        # else:
        #     for modality in range(num_modalities):
        #         likelihood_t *= spm_dot(A[modality], obs_t[obs][modality], obs_mode=True)
        
        likelihood_t = get_joint_likelihood(A, obs_t, num_states)

        # The Thomas Parr MMP version, you log the likelihood first
        # likelihood[t] = np.log(likelihood_t + 1e-16)

        # Karl SPM version, logging doesn't happen until *after* the dotting with the posterior
        likelihood[t] = likelihood_t

    """
    =========== Step 2 ===========
        Initialise a flat posterior (and prior if necessary)
        If a prior is not provided, initialise a uniform prior
    """

    qs = [np.empty(num_factors, dtype=object) for i in range(inference_len)]

    for t in range(inference_len):
        # if t == window_len:
        #     # final message is zeros - has no effect on inference
        #     # TODO: this may be redundant now that we skip last step
        #     for f in range(num_factors):
        #         qs[t][f] = np.zeros(num_states[f])
        # else:
            # for f in range(num_factors):
            #     qs[t][f] = np.ones(num_states[f]) / num_states[f]
        for f in range(num_factors):
                qs[t][f] = np.ones(num_states[f]) / num_states[f]

    if prior is None:
        prior = np.empty(num_factors, dtype=object)
        for f in range(num_factors):
            prior[f] = np.ones(num_states[f]) / num_states[f]

    """ 
    =========== Step 3 ===========
        Create a normalized transpose of the transition distribution `B_transposed`
        Used for computing backwards messages 'from the future'
    """

    B_transposed = np.empty(num_factors, dtype=object)
    for f in range(num_factors):
        B_transposed[f] = np.zeros_like(B[f])
        for u in range(B[f].shape[2]):
            B_transposed[f][:, :, u] = spm_norm(B[f][:, :, u].T)

    # zero out final message
    # TODO: may be redundant now we skip final step
    last_message = np.empty(num_factors, dtype=object)
    for f in range(num_factors):
        last_message[f] = np.zeros(num_states[f])

    # if previous actions not given, zero out to stop any influence on inference
    if previous_actions is None:
        previous_actions = np.zeros((1, policy.shape[1]))

    full_policy = np.vstack((previous_actions, policy))

    # print(full_policy.shape)

    """
    =========== Step 3 ===========
        Loop over time indices of time window, updating posterior as we go
        This includes past time steps and future time steps
    """

    qss = [[] for i in range(num_iter)]
    free_energy = np.zeros((len(qs), num_iter))
    free_energy_pol = 0.0

    # print(obs_seq_len)

    print('Full policy history')
    print('------------------')
    print(full_policy)

    for n in range(num_iter):
        for t in range(inference_len):

            lnB_past_tensor = np.empty(num_factors, dtype=object)
            for f in range(num_factors):

                # if t == 0 and n == 0:
                #     print(f"qs at time t = {t}, factor f = {f}, iteration i = {n}: {qs[t][f]}")

                """
                =========== Step 3.a ===========
                    Calculate likelihood
                """
                if t < len(obs_range):
                    # if t < len(obs_seq_len):
                    # Thomas Parr MMP version
                    # lnA = spm_dot(likelihood[t], qs[t], [f])

                    # Karl SPM version
                    lnA = np.log(spm_dot(likelihood[t], qs[t], [f]) + 1e-16)
                else:
                    lnA = np.zeros(num_states[f])

                if t == 1 and n == 0:
                    # pass
                    print(f"lnA at time t = {t}, factor f = {f}, iteration i = {n}: {lnA}")

                # print(f"lnA at time t = {t}, factor f = {f}, iteration i = {n}: {lnA}")

                """
                =========== Step 3.b ===========
                    Calculate past message
                """
                if t == 0 and window_idxs[0] == 0:
                    lnB_past = np.log(prior[f] + 1e-16)
                else:
                    # Thomas Parr MMP version
                    # lnB_past = 0.5 * np.log(B[f][:, :, full_policy[t - 1, f]].dot(qs[t - 1][f]) + 1e-16)

                    # Karl SPM version
                    if t == 1 and n == 0 and f == 1:
                        print('past action:')
                        print('-------------')
                        print(full_policy[t - 1, :])
                        print(B[f][:,:,0])
                        print(B[f][:,:,1])
                        print(qs[t - 1][f])
                    lnB_past = np.log(B[f][:, :, full_policy[t - 1, f]].dot(qs[t - 1][f]) + 1e-16)
                    # if t == 0:
                    # print(
                    # f"qs_t_1 at time t = {t}, factor f = {f}, iteration i = {n}: {qs[t - 1][f]}"
                    # )

                if t == 1 and n == 0:
                    print(
                        f"lnB_past at time t = {t}, factor f = {f}, iteration i = {n}: {lnB_past}"
                    )

                """
                =========== Step 3.c ===========
                    Calculate future message
                """
                if t >= future_cutoff:
                    # TODO: this is redundant - not used in code
                    lnB_future = last_message[f]
                else:
                    # Thomas Parr MMP version
                    # B_future = B_transposed[f][:, :, int(full_policy[t, f])].dot(qs[t + 1][f])
                    # lnB_future = 0.5 * np.log(B_future + 1e-16)

                    # Karl Friston SPM version
                    B_future = B_transposed[f][:, :, int(full_policy[t, f])].dot(qs[t + 1][f])
                    lnB_future = np.log(B_future + 1e-16)
                
                # Thomas Parr MMP version
                # lnB_past_tensor[f] = 2 * lnBpast
                # Karl SPM version
                lnB_past_tensor[f] = lnB_past

                """
                =========== Step 3.d ===========
                    Update posterior
                """
                if use_gradient_descent:
                    lns = np.log(qs[t][f] + 1e-16)

                    # Thomas Parr MMP version
                    # error = (lnA + lnBpast + lnBfuture) - lns

                    # Karl SPM version
                    if t >= future_cutoff:
                        error = lnA + lnB_past - lns

                    else:
                        error = (2 * lnA + lnB_past + lnB_future) - 2 * lns
                        

                    # print(f"prediction error at time t = {t}, factor f = {f}, iteration i = {n}: {error}")
                    # print(f"OG {t} {f} {error}")
                    error -= error.mean()
                    lns = lns + tau * error
                    qs_t_f = softmax(lns)
                    free_energy_pol += 0.5 * qs[t][f].dot(error)
                    qs[t][f] = qs_t_f
                else:
                    qs[t][f] = softmax(lnA + lnB_past + lnB_future)

            # TODO: probably works anyways
            # free_energy[t, n] = calc_free_energy(qs[t], lnB_past_tensor, num_factors, likelihood[t])
            # free_energy_pol += F[t, n]
        qss[n].append(qs)

    return qs, qss, free_energy, free_energy_pol
