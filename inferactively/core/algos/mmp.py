#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

""" Functions for performing variational inference on hidden states 

__author__: Conor Heins, Beren Millidge, Alexander Tschantz, Brennan Klein
"""

import numpy as np
from inferactively.core.maths import spm_dot, spm_norm, softmax, calc_free_energy
        
def run_mmp(A, B, obs_t, policy, curr_t, t_horizon, T, prior=None, num_iter=10, dF=1.0, dF_tol=0.001, previous_actions=None, use_gradient_descent=False, tau = 0.25):
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
        Used in inference to get expected future (or past) hidden states, given past (or future) hidden states (or expectations thereof).
    - 'obs_t' [list of length t_horizon of numpy 1D array or array of arrays (with 1D numpy array entries)]:
        Sequence of observations sampled from beginning of time horizon the current timestep t. The first observation (the start of the time horizon) 
        is either the first timestep of the generative process or the first timestep of the policy horizon (whichever is sooner in time).
        The observations over time are stored as a list of numpy arrays, where in case of multi-modalities each numpy array is an array-of-arrays, with
        one 1D numpy.ndarray for each modality. In the case of a single modality, each observation is a single 1D numpy.ndarray.
    - 'policy' [2D np.ndarray]:
        Array of actions constituting a single policy. Policy is a shape (n_steps, n_control_factors) numpy.ndarray, the values of which
        indicate actions along a given control factor (column index) at a given timestep (row index).
    - 'curr_t' [int]:
        Current timestep (relative to the 'absolute' time of the generative process).
    - 't_horizon'[int]:
        Temporal horizon of inference for states and policies.
    - 'T' [int]:
        Temporal horizon of the generative process (absolute time)
    - 'prior' [numpy 1D array, array of arrays (with 1D numpy array entries) or None]:
        Prior beliefs of the agent at the beginning of the time horizon, to be integrated with the marginal likelihood to obtain posterior at the first timestep.
        If absent, prior is set to be a uniform distribution over hidden states (identical to the initialisation of the posterior.
    -'num_iter' [int]:
        Number of variational iterations to run. (optional)
    -'dF' [float]:
        Starting free energy gradient (dF/dt) before updating in the course of gradient descent.  (optional)
    -'dF_tol' [float]:
        Threshold value of the gradient of the variational free energy (dF/dt), to be checked at each iteration. If 
        dF <= dF_tol, the iterations are halted pre-emptively and the final marginal posterior belief(s) is(are) returned.  (optional)
    -'previous_actions' [numpy.ndarray with shape (num_steps, n_control_factors) or None]:
        Array of previous actions, which can be used to constrain the 'past' messages in inference to only consider states of affairs that were possible
        under actions that are known to have been taken. The first dimension of previous-arrays (previous_actions.shape[0]) encodes how far back in time
        the agent is considering. The first timestep of this either corresponds to either the first timestep of the generative process or the f
        first timestep of the policy horizon (whichever is sooner in time).  (optional)
    -'use_gradient_descent' [bool]:
        Flag to indicate whether to use gradient descent to optimise posterior beliefs.
    -'tau' [float]:
        Learning rate for gradient descent (only used if use_gradient_descent is True)
 
  
    Returns
    ----------
    -'qs' [list of length T of numpy 1D arrays or array of arrays (with 1D numpy array entries):
        Marginal posterior beliefs over hidden states (single- or multi-factor) achieved via marginal message pasing
    -'qss' [list of lists of length T of numpy 1D arrays or array of arrays (with 1D numpy array entries):
        Marginal posterior beliefs about hidden states (single- or multi-factor) held at each timepoint, *about* each timepoint of the observation
        sequence
    -'F' [2D np.ndarray]:
        Variational free energy of beliefs about hidden states, indexed by time point and variational iteration
    -'F_pol' [float]:
        Total free energy of the policy under consideration.
    """

    # get model dimensions
    time_window_idxs = np.array([i for i in range(max(0,curr_t-t_horizon),min(T,curr_t+t_horizon))])
    window_len = len(time_window_idxs)
    if utils.is_arr_of_arr(obs_t[0]):
        n_observations = [ obs_array_i.shape[0] for obs_array_i in obs_t[0] ]
    else:
        n_observations = [obs_t[0].shape[0]]
    
    if utils.is_arr_of_arr(qs_t[0][0]):
        n_states = [ qs_array_i.shape[0] for obs_array_i in qs_t[0][0] ]
    else:
        n_states = [qs_t[0][0].shape[0]]
    
    n_modalities = len(n_observations)
    n_factors = len(n_states)

    """
    =========== Step 1 ===========
        Loop over the observation modalities and use assumption of independence among observation modalities
        to multiply each modality-specific likelihood onto a single joint likelihood over hidden states [shape = n_states]
    """

    # compute time-window, taking into account boundary conditions
    obs_range = range(max(0,curr_t-t_horizon),curr_t)

    # likelihood of observations under configurations of hidden causes (over time)
    likelihood = np.empty(len(obs_range), dtype = object)
    for t in range(len(obs_range)):
        likelihood_t = np.ones(tuple(n_states))

        if n_modalities is 1:
            likelihood_t *= spm_dot(A, obs_t[obs_range[t]], obs_mode=True)
        else:
            for modality in range(n_modalities):
                likelihood_t *= spm_dot(A[modality], obs[obs_range[t]][modality], obs_mode=True)
        likelihood[t] = np.log(likelihood_t + 1e-16)

    """
    =========== Step 2 ===========
        Create a flat posterior (and prior if necessary)
        If prior is not provided, initialise prior to be identical to posterior
        (namely, a flat categorical distribution).
    """

    qs = [np.empty(n_factors,dtype=object) for i in range(window_len)+2]
    if prior is None:
        prior = np.array([np.ones(n_states[factor]) / n_states[factor] for f in range(n_factors)],dtype=object)
    # setup prior as first backwards message
    qs[0] = prior
    #set final future message as all ones at the time horizon (no information from beyond the horizon)
    qs[-1] = np.array([np.ones(n_states[f]) for f in n_factors],dtype=object)
    

    """
    =========== Step 3 ===========
        Loop over time indices of time window, which includes time before the policy horizon 
        as well as including the policy horizon
        n_steps, n_factors [0 1 2 0;
                            1 2 0 1]
    """

    if previous_actions is None:
        full_policy = policies
    else:
        full_policy = np.vstack( (previous_actions, policies))

    qss = [[] for i in range(1, len(qs)-1)]
    F = np.zeros((len(qs), num_iter))
    F_pol = 0.0
    for t in range(1, len(qs)-1):
        for n in range(num_iter):
            lnBpast_tensor = np.empty(n_factors,dtype=object)         
            for f in range(n_factors):
                if t <=len(obs):
                    lnA = spm_dot(likelihood[t],qs[t],f)
                else:
                    lnA = np.zeros(num_states[f])
                lnBpast = 0.5 * B[f][:,:,full_policy[t-1,f].dot(qs[t-1][f])
                lnBfuture = 0.5 * spm_norm(B[f][:,:,full_policy[t+1,f]].T).dot(qs[t+1][f])
                lnBpast_tensor[f] = 2 * lnBpast         
                if use_gradient_descent:
                    # gradients
                    lns = np.log(qs[t][f] + 1e-16) # current estimate
                    e = (lnA + lnBpast + lnBfuture) - lns # prediction error
                    lns += tau * e # increment the current (log) belief with the prediction error
                    qs = softmax(lns)
                    qs[t][f] = qs
                    qss[t].append(qs)
                else:
                    # free energy minimum for the factor in question
                    qs[t][f] = softmax(lnA + lnBpast + lnBfuture)
                    qss[t].append(qs[t][f])
            
            F[t,n] = calc_free_energy(qs[t], lnBpast_tensor, n_factors, likelihood[t])
            F_pol += F[t,n]
    
    return qs, qss, F, F_pol




