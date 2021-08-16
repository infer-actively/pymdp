#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

""" Functions for performing variational inference on hidden states 

__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import numpy as np

from pymdp import utils
from pymdp.maths import get_joint_likelihood_seq
from pymdp.algos import run_fpi, run_mmp, run_mmp_testing

VANILLA = "VANILLA"
VMP = "VMP"
MMP = "MMP"
BP = "BP"
EP = "EP"
CV = "CV"

def update_posterior_states_v2(
    A,
    B,
    prev_obs,
    policies,
    prev_actions=None,
    prior=None,
    policy_sep_prior = True,
    **kwargs,
):
    """
    Update posterior over hidden states using marginal message passing
    Parameters
    ----------
    `A` [numpy object array]:
        - Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element A[m] of
        this object array stores an np.ndarray multidimensional array that stores the mapping from hidden states to observations. 
    `B` [numpy object array]:
        - Dynamics likelihood mapping or 'transition model', mapping from hidden states at `t` to hidden states at `t+1`, given some control state `u`.
        Each element B[f] of this object array stores a 3-D tensor that stores the mapping between hidden states and actions at `t` to hidden states at `t+1`.
    `prev_obs` [list]:
        - List of observations over time. Each observation in the list can be an int, a list of ints, a tuple of ints, a one-hot vector or an object array of one-hot vectors.
    `prior` [numpy object array or None]:
        - If provided, this a numpy object array with one sub-array per hidden state factor, that stores the prior beliefs about initial states (at t = 0, relative to `infer_len`). If `None`, this defaults
        to a flat (uninformative) prior over hidden states.
    `policy_sep_prior` [Bool, default True]:s
        - Flag determining whether the prior beliefs from the past are unconditioned on policy, or separated by /conditioned on the policy variable.
    **kwargs [optional keyword arguments to `run_mmp`]:
        - Optional keyword arguments for the function `run_mmp`

    Returns:
    ---------
    `qs_seq_pi` [numpy object array]:
        - posterior beliefs over hidden states for each policy. Structure is policies --> timepoints --> factors/marginals,
        e.g. `qs_seq_pi[p][t][f]` gets the marginal belief about factor `f` at timepoint `t` under policy `p`
    """

    num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A, B)
    
    prev_obs = utils.process_observation_seq(prev_obs, num_modalities, num_obs)
   
    lh_seq = get_joint_likelihood_seq(A, prev_obs, num_states)

    if prev_actions is not None:
        prev_actions = np.stack(prev_actions,0)

    qs_seq_pi = utils.obj_array(len(policies))
    F = np.zeros(len(policies)) # variational free energy of policies

    if policy_sep_prior:
        for p_idx, policy in enumerate(policies):
            # get sequence and the free energy for policy
            qs_seq_pi[p_idx], F[p_idx] = run_mmp(
                lh_seq,
                B,
                policy,
                prev_actions=prev_actions,
                prior=prior[p_idx], 
                **kwargs
            )
    else:
        for p_idx, policy in enumerate(policies):
            # get sequence and the free energy for policy
            qs_seq_pi[p_idx], F[p_idx] = run_mmp(
                lh_seq,
                B,
                policy,
                prev_actions=prev_actions,
                prior=prior, 
                **kwargs
            )

    return qs_seq_pi, F

def update_posterior_states_v2_test(
    A,
    B,
    prev_obs,
    policies,
    prev_actions=None,
    prior=None,
    policy_sep_prior = True,
    **kwargs,
):
    """
    Update posterior over hidden states using marginal message passing
    Parameters
    ----------
    `A` [numpy object array]:
        - Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element A[m] of
        this object array stores an np.ndarray multidimensional array that stores the mapping from hidden states to observations. 
    `B` [numpy object array]:
        - Dynamics likelihood mapping or 'transition model', mapping from hidden states at `t` to hidden states at `t+1`, given some control state `u`.
        Each element B[f] of this object array stores a 3-D tensor that stores the mapping between hidden states and actions at `t` to hidden states at `t+1`.
    `prev_obs` [list]:
        - List of observations over time. Each observation in the list can be an int, a list of ints, a tuple of ints, a one-hot vector or an object array of one-hot vectors.
    `prior` [numpy object array or None]:
        - If provided, this a numpy object array with one sub-array per hidden state factor, that stores the prior beliefs about initial states (at t = 0, relative to `infer_len`). If `None`, this defaults
        to a flat (uninformative) prior over hidden states.
    `policy_sep_prior` [Bool, default True]:s
        - Flag determining whether the prior beliefs from the past are unconditioned on policy, or separated by /conditioned on the policy variable.
    **kwargs [optional keyword arguments to `run_mmp`]:
        - Optional keyword arguments for the function `run_mmp`

    Returns:
    ---------
    `qs_seq_pi` [numpy object array]:
        - posterior beliefs over hidden states for each policy. Structure is policies --> timepoints --> factors/marginals,
        e.g. `qs_seq_pi[p][t][f]` gets the marginal belief about factor `f` at timepoint `t` under policy `p`
    """
    # safe convert to numpy

    num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A, B)

    prev_obs = utils.process_observation_seq(prev_obs, num_modalities, num_obs)
    
    lh_seq = get_joint_likelihood_seq(A, prev_obs, num_states)

    if prev_actions is not None:
        prev_actions = np.stack(prev_actions,0)

    qs_seq_pi = utils.obj_array(len(policies))
    xn_seq_pi = utils.obj_array(len(policies))
    vn_seq_pi = utils.obj_array(len(policies))
    F = np.zeros(len(policies)) # variational free energy of policies

    if policy_sep_prior:
        for p_idx, policy in enumerate(policies):
            # get sequence and the free energy for policy
            qs_seq_pi[p_idx], F[p_idx], xn_seq_pi[p_idx], vn_seq_pi[p_idx] = run_mmp_testing(
                lh_seq,
                B,
                policy,
                prev_actions=prev_actions,
                prior=prior[p_idx], 
                **kwargs
            )
    else:
        for p_idx, policy in enumerate(policies):
            # get sequence and the free energy for policy
            qs_seq_pi[p_idx], F[p_idx], xn_seq_pi[p_idx], vn_seq_pi[p_idx] = run_mmp_testing(
                lh_seq,
                B,
                policy,
                prev_actions=prev_actions,
                prior=prior, 
                **kwargs
            )

    return qs_seq_pi, F, xn_seq_pi, vn_seq_pi

def average_states_over_policies(qs_pi, q_pi):
    """
    Parameters
    ----------
    `qs_seq_pi` - marginal posteriors over hidden states, per policy, at the current time point
    `q_pi` - posterior beliefs about policies  - (num_policies x 1) numpy 1D array

    Returns:
    ---------
    `qs_bma` - marginal posterior over hidden states for the current timepoint, averaged across policies according to their posterior probability given by `q_pi`
    """

    num_factors = len(qs_pi[0]) # get the number of hidden state factors using the shape of the first-policy-conditioned posterior
    num_states = [qs_f.shape[0] for qs_f in qs_pi[0]] # get the dimensionalities of each hidden state factor 

    qs_bma = utils.obj_array(num_factors)
    for f in range(num_factors):
        qs_bma[f] = np.zeros(num_states[f])

    for p_idx, policy_weight in enumerate(q_pi):

        for f in range(num_factors):

            qs_bma[f] += qs_pi[p_idx][f] * policy_weight

    return qs_bma

def update_posterior_states(A, obs, prior=None, **kwargs):
    """
    Update marginal posterior over hidden states using mean-field fixed point iteration 
    FPI or Fixed point iteration
            - http://www.cs.cmu.edu/~guestrin/Class/10708/recitations/r9/VI-view.pdf,
            slides 13- 18
            - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.137.221&rep=rep1&type=pdf,
            slides 24 - 38
    Parameters
    ----------
    - 'A' [numpy nd.array (matrix or tensor or array-of-arrays)]:
        Observation likelihood of the generative model, mapping from hidden states to observations
        Used to invert generative model to obtain marginal likelihood over hidden states,
        given the observation
    - 'obs' [numpy 1D array, array of arrays (with 1D numpy array entries), int or tuple]:
        The observation (generated by the environment). If single modality, this can be a 1D array
        (one-hot vector representation) or an int (observation index)
        If multi-modality, this can be an array of arrays (whose entries are 1D one-hot vectors)
        or a tuple (of observation indices)
    - 'prior' [numpy 1D array, array of arrays (with 1D numpy array entries) or None]:
        Prior beliefs about hidden states, to be integrated with the marginal likelihood to obtain
         a posterior distribution.
        If None, prior is set to be equal to a flat categorical distribution (at the level of
        the individual inference functions).
        (optional)
    - **kwargs:
        List of keyword/parameter arguments corresponding to parameter values for the fixed-point iteration
        algorithm.

    Returns
    ----------
    - 'qs' [numpy 1D array, array of arrays (with 1D numpy array entries), or Categorical]:
        Marginal posterior beliefs over hidden states
    """

    num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A = A)
    
    obs = utils.process_observation(obs, num_modalities, num_obs)

    if prior is not None:
        prior = utils.to_arr_of_arr(prior)

    return run_fpi(A, obs, num_obs, num_states, prior, **kwargs)
   

def print_inference_methods():
    print(f"Avaliable Inference methods: {FPI}, {MMP}")
