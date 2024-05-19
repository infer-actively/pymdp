#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-an-iterable

import itertools
import numpy as np
from pymdp.maths import softmax, softmax_obj_arr, spm_dot, spm_wnorm, spm_MDP_G, spm_log_single, spm_log_obj_array
from pymdp import utils
import copy

def update_posterior_policies_full(
    qs_seq_pi,
    A,
    B,
    C,
    policies,
    use_utility=True,
    use_states_info_gain=True,
    use_param_info_gain=False,
    prior=None,
    pA=None,
    pB=None,
    F=None,
    E=None,
    I=None,
    gamma=16.0
):  
    """
    Update posterior beliefs about policies by computing expected free energy of each policy and integrating that
    with the variational free energy of policies ``F`` and prior over policies ``E``. This is intended to be used in conjunction
    with the ``update_posterior_states_full`` method of ``inference.py``, since the full posterior over future timesteps, under all policies, is
    assumed to be provided in the input array ``qs_seq_pi``.

    Parameters
    ----------
    qs_seq_pi: ``numpy.ndarray`` of dtype object
        Posterior beliefs over hidden states for each policy. Nesting structure is policies, timepoints, factors,
        where e.g. ``qs_seq_pi[p][t][f]`` stores the marginal belief about factor ``f`` at timepoint ``t`` under policy ``p``.
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    C: ``numpy.ndarray`` of dtype object
       Prior over observations or 'prior preferences', storing the "value" of each outcome in terms of relative log probabilities. 
       This is softmaxed to form a proper probability distribution before being used to compute the expected utility term of the expected free energy.
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy in ``policies[p_idx]``. Shape of ``policies[p_idx]`` is ``(num_timesteps, num_factors)`` where `num_timesteps` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    use_utility: ``Bool``, default ``True``
        Boolean flag that determines whether expected utility should be incorporated into computation of EFE.
    use_states_info_gain: ``Bool``, default ``True``
        Boolean flag that determines whether state epistemic value (info gain about hidden states) should be incorporated into computation of EFE.
    use_param_info_gain: ``Bool``, default ``False`` 
        Boolean flag that determines whether parameter epistemic value (info gain about generative model parameters) should be incorporated into computation of EFE. 
    prior: ``numpy.ndarray`` of dtype object, default ``None``
        If provided, this is a ``numpy`` object array with one sub-array per hidden state factor, that stores the prior beliefs about initial states. 
        If ``None``, this defaults to a flat (uninformative) prior over hidden states.
    pA: ``numpy.ndarray`` of dtype object, default ``None``
        Dirichlet parameters over observation model (same shape as ``A``)
    pB: ``numpy.ndarray`` of dtype object, default ``None``
        Dirichlet parameters over transition model (same shape as ``B``)
    F: 1D ``numpy.ndarray``, default ``None``
        Vector of variational free energies for each policy
    E: 1D ``numpy.ndarray``, default ``None``
        Vector of prior probabilities of each policy (what's referred to in the active inference literature as "habits"). If ``None``, this defaults to a flat (uninformative) prior over policies.
    I: ``numpy.ndarray`` of dtype object
        For each state factor, contains a 2D ``numpy.ndarray`` whose element i,j yields the probability 
        of reaching the goal state backwards from state j after i steps.
    gamma: ``float``, default 16.0
        Prior precision over policies, scales the contribution of the expected free energy to the posterior over policies

    Returns
    ----------
    q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
    G: 1D ``numpy.ndarray``
        Negative expected free energies of each policy, i.e. a vector containing one negative expected free energy per policy.
    """

    num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A, B)
    horizon = len(qs_seq_pi[0])
    num_policies = len(qs_seq_pi)

    qo_seq = utils.obj_array(horizon)
    for t in range(horizon):
        qo_seq[t] = utils.obj_array_zeros(num_obs)

    # initialise expected observations
    qo_seq_pi = utils.obj_array(num_policies)

    # initialize (negative) expected free energies for all policies
    G = np.zeros(num_policies)

    if F is None:
        F = spm_log_single(np.ones(num_policies) / num_policies)

    if E is None:
        lnE = spm_log_single(np.ones(num_policies) / num_policies)
    else:
        lnE = spm_log_single(E) 

    if I is not None:
        init_qs_all_pi = [qs_seq_pi[p][0] for p in range(num_policies)]
        qs_bma = inference.average_states_over_policies(init_qs_all_pi, softmax(E))

    for p_idx, policy in enumerate(policies):

        qo_seq_pi[p_idx] = get_expected_obs(qs_seq_pi[p_idx], A)

        if use_utility:
            G[p_idx] += calc_expected_utility(qo_seq_pi[p_idx], C)
        
        if use_states_info_gain:
            G[p_idx] += calc_states_info_gain(A, qs_seq_pi[p_idx])
        
        if use_param_info_gain:
            if pA is not None:
                G[p_idx] += calc_pA_info_gain(pA, qo_seq_pi[p_idx], qs_seq_pi[p_idx])
            if pB is not None:
                G[p_idx] += calc_pB_info_gain(pB, qs_seq_pi[p_idx], prior, policy)
        
        if I is not None:
            G[p_idx] += calc_inductive_cost(qs_bma, qs_seq_pi[p_idx], I)

    q_pi = softmax(G * gamma - F + lnE)
    
    return q_pi, G

def update_posterior_policies_full_factorized(
    qs_seq_pi,
    A,
    B,
    C,
    A_factor_list,
    B_factor_list,
    policies,
    use_utility=True,
    use_states_info_gain=True,
    use_param_info_gain=False,
    prior=None,
    pA=None,
    pB=None,
    F=None,
    E=None,
    I=None,
    gamma=16.0
):  
    """
    Update posterior beliefs about policies by computing expected free energy of each policy and integrating that
    with the variational free energy of policies ``F`` and prior over policies ``E``. This is intended to be used in conjunction
    with the ``update_posterior_states_full`` method of ``inference.py``, since the full posterior over future timesteps, under all policies, is
    assumed to be provided in the input array ``qs_seq_pi``.

    Parameters
    ----------
    qs_seq_pi: ``numpy.ndarray`` of dtype object
        Posterior beliefs over hidden states for each policy. Nesting structure is policies, timepoints, factors,
        where e.g. ``qs_seq_pi[p][t][f]`` stores the marginal belief about factor ``f`` at timepoint ``t`` under policy ``p``.
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    C: ``numpy.ndarray`` of dtype object
       Prior over observations or 'prior preferences', storing the "value" of each outcome in terms of relative log probabilities. 
       This is softmaxed to form a proper probability distribution before being used to compute the expected utility term of the expected free energy.
    A_factor_list: ``list`` of ``list``s of ``int``
        ``list`` that stores the indices of the hidden state factor indices that each observation modality depends on. For example, if ``A_factor_list[m] = [0, 1]``, then
        observation modality ``m`` depends on hidden state factors 0 and 1.
    B_factor_list: ``list`` of ``list``s of ``int``
        ``list`` that stores the indices of the hidden state factor indices that each hidden state factor depends on. For example, if ``B_factor_list[f] = [0, 1]``, then
        the transitions in hidden state factor ``f`` depend on hidden state factors 0 and 1.
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy in ``policies[p_idx]``. Shape of ``policies[p_idx]`` is ``(num_timesteps, num_factors)`` where `num_timesteps` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    use_utility: ``Bool``, default ``True``
        Boolean flag that determines whether expected utility should be incorporated into computation of EFE.
    use_states_info_gain: ``Bool``, default ``True``
        Boolean flag that determines whether state epistemic value (info gain about hidden states) should be incorporated into computation of EFE.
    use_param_info_gain: ``Bool``, default ``False`` 
        Boolean flag that determines whether parameter epistemic value (info gain about generative model parameters) should be incorporated into computation of EFE. 
    prior: ``numpy.ndarray`` of dtype object, default ``None``
        If provided, this is a ``numpy`` object array with one sub-array per hidden state factor, that stores the prior beliefs about initial states. 
        If ``None``, this defaults to a flat (uninformative) prior over hidden states.
    pA: ``numpy.ndarray`` of dtype object, default ``None``
        Dirichlet parameters over observation model (same shape as ``A``)
    pB: ``numpy.ndarray`` of dtype object, default ``None``
        Dirichlet parameters over transition model (same shape as ``B``)
    F: 1D ``numpy.ndarray``, default ``None``
        Vector of variational free energies for each policy
    E: 1D ``numpy.ndarray``, default ``None``
        Vector of prior probabilities of each policy (what's referred to in the active inference literature as "habits"). If ``None``, this defaults to a flat (uninformative) prior over policies.
    I: ``numpy.ndarray`` of dtype object
        For each state factor, contains a 2D ``numpy.ndarray`` whose element i,j yields the probability 
        of reaching the goal state backwards from state j after i steps.
    gamma: ``float``, default 16.0
        Prior precision over policies, scales the contribution of the expected free energy to the posterior over policies

    Returns
    ----------
    q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
    G: 1D ``numpy.ndarray``
        Negative expected free energies of each policy, i.e. a vector containing one negative expected free energy per policy.
    """

    num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A, B)
    horizon = len(qs_seq_pi[0])
    num_policies = len(qs_seq_pi)

    qo_seq = utils.obj_array(horizon)
    for t in range(horizon):
        qo_seq[t] = utils.obj_array_zeros(num_obs)

    # initialise expected observations
    qo_seq_pi = utils.obj_array(num_policies)

    # initialize (negative) expected free energies for all policies
    G = np.zeros(num_policies)

    if F is None:
        F = spm_log_single(np.ones(num_policies) / num_policies)

    if E is None:
        lnE = spm_log_single(np.ones(num_policies) / num_policies)
    else:
        lnE = spm_log_single(E) 

    if I is not None:
        init_qs_all_pi = [qs_seq_pi[p][0] for p in range(num_policies)]
        qs_bma = inference.average_states_over_policies(init_qs_all_pi, softmax(E))

    for p_idx, policy in enumerate(policies):

        qo_seq_pi[p_idx] = get_expected_obs_factorized(qs_seq_pi[p_idx], A, A_factor_list)

        if use_utility:
            G[p_idx] += calc_expected_utility(qo_seq_pi[p_idx], C)
        
        if use_states_info_gain:
            G[p_idx] += calc_states_info_gain_factorized(A, qs_seq_pi[p_idx], A_factor_list)
        
        if use_param_info_gain:
            if pA is not None:
                G[idx] += calc_pA_info_gain_factorized(pA, qo_seq_pi[p_idx], qs_seq_pi[p_idx], A_factor_list)
            if pB is not None:
                G[idx] += calc_pB_info_gain_interactions(pB, qs_seq_pi[p_idx], qs, B_factor_list, policy)
        
        if I is not None:
            G[p_idx] += calc_inductive_cost(qs_bma, qs_seq_pi[p_idx], I)
            
    q_pi = softmax(G * gamma - F + lnE)
    
    return q_pi, G


def update_posterior_policies(
    qs,
    A,
    B,
    C,
    policies,
    use_utility=True,
    use_states_info_gain=True,
    use_param_info_gain=False,
    pA=None,
    pB=None,
    E=None,
    I=None,
    gamma=16.0
):
    """
    Update posterior beliefs about policies by computing expected free energy of each policy and integrating that
    with the prior over policies ``E``. This is intended to be used in conjunction
    with the ``update_posterior_states`` method of the ``inference`` module, since only the posterior about the hidden states at the current timestep
    ``qs`` is assumed to be provided, unconditional on policies. The predictive posterior over hidden states under all policies Q(s, pi) is computed 
    using the starting posterior about states at the current timestep ``qs`` and the generative model (e.g. ``A``, ``B``, ``C``)

    Parameters
    ----------
    qs: ``numpy.ndarray`` of dtype object
        Marginal posterior beliefs over hidden states at current timepoint (unconditioned on policies)
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    C: ``numpy.ndarray`` of dtype object
       Prior over observations or 'prior preferences', storing the "value" of each outcome in terms of relative log probabilities. 
       This is softmaxed to form a proper probability distribution before being used to compute the expected utility term of the expected free energy.
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy in ``policies[p_idx]``. Shape of ``policies[p_idx]`` is ``(num_timesteps, num_factors)`` where `num_timesteps` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    use_utility: ``Bool``, default ``True``
        Boolean flag that determines whether expected utility should be incorporated into computation of EFE.
    use_states_info_gain: ``Bool``, default ``True``
        Boolean flag that determines whether state epistemic value (info gain about hidden states) should be incorporated into computation of EFE.
    use_param_info_gain: ``Bool``, default ``False`` 
        Boolean flag that determines whether parameter epistemic value (info gain about generative model parameters) should be incorporated into computation of EFE.
    pA: ``numpy.ndarray`` of dtype object, optional
        Dirichlet parameters over observation model (same shape as ``A``)
    pB: ``numpy.ndarray`` of dtype object, optional
        Dirichlet parameters over transition model (same shape as ``B``)
    E: 1D ``numpy.ndarray``, optional
        Vector of prior probabilities of each policy (what's referred to in the active inference literature as "habits")
    I: ``numpy.ndarray`` of dtype object
        For each state factor, contains a 2D ``numpy.ndarray`` whose element i,j yields the probability 
        of reaching the goal state backwards from state j after i steps.
    gamma: float, default 16.0
        Prior precision over policies, scales the contribution of the expected free energy to the posterior over policies

    Returns
    ----------
    q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
    G: 1D ``numpy.ndarray``
        Negative expected free energies of each policy, i.e. a vector containing one negative expected free energy per policy.
    """

    n_policies = len(policies)
    G = np.zeros(n_policies)
    q_pi = np.zeros((n_policies, 1))

    if E is None:
        lnE = spm_log_single(np.ones(n_policies) / n_policies)
    else:
        lnE = spm_log_single(E) 

    for idx, policy in enumerate(policies):
        qs_pi = get_expected_states(qs, B, policy)
        qo_pi = get_expected_obs(qs_pi, A)

        if use_utility:
            G[idx] += calc_expected_utility(qo_pi, C)

        if use_states_info_gain:
            G[idx] += calc_states_info_gain(A, qs_pi)

        if use_param_info_gain:
            if pA is not None:
                G[idx] += calc_pA_info_gain(pA, qo_pi, qs_pi)
            if pB is not None:
                G[idx] += calc_pB_info_gain(pB, qs_pi, qs, policy)
        
        if I is not None:
            G[idx] += calc_inductive_cost(qs, qs_pi, I)

    q_pi = softmax(G * gamma + lnE)    

    return q_pi, G

def update_posterior_policies_factorized(
    qs,
    A,
    B,
    C,
    A_factor_list,
    B_factor_list,
    policies,
    use_utility=True,
    use_states_info_gain=True,
    use_param_info_gain=False,
    pA=None,
    pB=None,
    E=None,
    I=None,
    gamma=16.0
):
    """
    Update posterior beliefs about policies by computing expected free energy of each policy and integrating that
    with the prior over policies ``E``. This is intended to be used in conjunction
    with the ``update_posterior_states`` method of the ``inference`` module, since only the posterior about the hidden states at the current timestep
    ``qs`` is assumed to be provided, unconditional on policies. The predictive posterior over hidden states under all policies Q(s, pi) is computed 
    using the starting posterior about states at the current timestep ``qs`` and the generative model (e.g. ``A``, ``B``, ``C``)

    Parameters
    ----------
    qs: ``numpy.ndarray`` of dtype object
        Marginal posterior beliefs over hidden states at current timepoint (unconditioned on policies)
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    C: ``numpy.ndarray`` of dtype object
       Prior over observations or 'prior preferences', storing the "value" of each outcome in terms of relative log probabilities. 
       This is softmaxed to form a proper probability distribution before being used to compute the expected utility term of the expected free energy.
    A_factor_list: ``list`` of ``list``s of ``int``
        ``list`` that stores the indices of the hidden state factor indices that each observation modality depends on. For example, if ``A_factor_list[m] = [0, 1]``, then
        observation modality ``m`` depends on hidden state factors 0 and 1.
    B_factor_list: ``list`` of ``list``s of ``int``
        ``list`` that stores the indices of the hidden state factor indices that each hidden state factor depends on. For example, if ``B_factor_list[f] = [0, 1]``, then
        the transitions in hidden state factor ``f`` depend on hidden state factors 0 and 1.
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy in ``policies[p_idx]``. Shape of ``policies[p_idx]`` is ``(num_timesteps, num_factors)`` where `num_timesteps` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    use_utility: ``Bool``, default ``True``
        Boolean flag that determines whether expected utility should be incorporated into computation of EFE.
    use_states_info_gain: ``Bool``, default ``True``
        Boolean flag that determines whether state epistemic value (info gain about hidden states) should be incorporated into computation of EFE.
    use_param_info_gain: ``Bool``, default ``False`` 
        Boolean flag that determines whether parameter epistemic value (info gain about generative model parameters) should be incorporated into computation of EFE.
    pA: ``numpy.ndarray`` of dtype object, optional
        Dirichlet parameters over observation model (same shape as ``A``)
    pB: ``numpy.ndarray`` of dtype object, optional
        Dirichlet parameters over transition model (same shape as ``B``)
    E: 1D ``numpy.ndarray``, optional
        Vector of prior probabilities of each policy (what's referred to in the active inference literature as "habits")
    I: ``numpy.ndarray`` of dtype object
        For each state factor, contains a 2D ``numpy.ndarray`` whose element i,j yields the probability 
        of reaching the goal state backwards from state j after i steps.
    gamma: float, default 16.0
        Prior precision over policies, scales the contribution of the expected free energy to the posterior over policies

    Returns
    ----------
    q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
    G: 1D ``numpy.ndarray``
        Negative expected free energies of each policy, i.e. a vector containing one negative expected free energy per policy.
    """

    n_policies = len(policies)
    G = np.zeros(n_policies)
    q_pi = np.zeros((n_policies, 1))

    if E is None:
        lnE = spm_log_single(np.ones(n_policies) / n_policies)
    else:
        lnE = spm_log_single(E) 

    for idx, policy in enumerate(policies):
        qs_pi = get_expected_states_interactions(qs, B, B_factor_list, policy)
        qo_pi = get_expected_obs_factorized(qs_pi, A, A_factor_list)

        if use_utility:
            G[idx] += calc_expected_utility(qo_pi, C)

        if use_states_info_gain:
            G[idx] += calc_states_info_gain_factorized(A, qs_pi, A_factor_list)

        if use_param_info_gain:
            if pA is not None:
                G[idx] += calc_pA_info_gain_factorized(pA, qo_pi, qs_pi, A_factor_list)
            if pB is not None:
                G[idx] += calc_pB_info_gain_interactions(pB, qs_pi, qs, B_factor_list, policy)
        
        if I is not None:
            G[idx] += calc_inductive_cost(qs, qs_pi, I)

    q_pi = softmax(G * gamma + lnE)    

    return q_pi, G

def get_expected_states(qs, B, policy):
    """
    Compute the expected states under a policy, also known as the posterior predictive density over states

    Parameters
    ----------
    qs: ``numpy.ndarray`` of dtype object
        Marginal posterior beliefs over hidden states at a given timepoint.
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    policy: 2D ``numpy.ndarray``
        Array that stores actions entailed by a policy over time. Shape is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.

    Returns
    -------
    qs_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over hidden states expected under the policy, where ``qs_pi[t]`` stores the beliefs about
        hidden states expected under the policy at time ``t``
    """
    n_steps = policy.shape[0]
    n_factors = policy.shape[1]

    # initialise posterior predictive density as a list of beliefs over time, including current posterior beliefs about hidden states as the first element
    qs_pi = [qs] + [utils.obj_array(n_factors) for t in range(n_steps)]
    
    # get expected states over time
    for t in range(n_steps):
        for control_factor, action in enumerate(policy[t,:]):
            qs_pi[t+1][control_factor] = B[control_factor][:,:,int(action)].dot(qs_pi[t][control_factor])

    return qs_pi[1:]
    
def get_expected_states_interactions(qs, B, B_factor_list, policy):
    """
    Compute the expected states under a policy, also known as the posterior predictive density over states

    Parameters
    ----------
    qs: ``numpy.ndarray`` of dtype object
        Marginal posterior beliefs over hidden states at a given timepoint.
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    B_factor_list: ``list`` of ``list`` of ``int``
        List of lists of hidden state factors each hidden state factor depends on. Each element ``B_factor_list[i]`` is a list of the factor indices that factor i's dynamics depend on.
    policy: 2D ``numpy.ndarray``
        Array that stores actions entailed by a policy over time. Shape is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.

    Returns
    -------
    qs_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over hidden states expected under the policy, where ``qs_pi[t]`` stores the beliefs about
        hidden states expected under the policy at time ``t``
    """
    n_steps = policy.shape[0]
    n_factors = policy.shape[1]

    # initialise posterior predictive density as a list of beliefs over time, including current posterior beliefs about hidden states as the first element
    qs_pi = [qs] + [utils.obj_array(n_factors) for t in range(n_steps)]
    
    # get expected states over time
    for t in range(n_steps):
        for control_factor, action in enumerate(policy[t,:]):
            factor_idx = B_factor_list[control_factor] # list of the hidden state factor indices that the dynamics of `qs[control_factor]` depend on
            qs_pi[t+1][control_factor] = spm_dot(B[control_factor][...,int(action)], qs_pi[t][factor_idx])

    return qs_pi[1:]
 
def get_expected_obs(qs_pi, A):
    """
    Compute the expected observations under a policy, also known as the posterior predictive density over observations

    Parameters
    ----------
    qs_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over hidden states expected under the policy, where ``qs_pi[t]`` stores the beliefs about
        hidden states expected under the policy at time ``t``
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``

    Returns
    -------
    qo_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over observations expected under the policy, where ``qo_pi[t]`` stores the beliefs about
        observations expected under the policy at time ``t``
    """

    n_steps = len(qs_pi) # each element of the list is the PPD at a different timestep

    # initialise expected observations
    qo_pi = []

    for t in range(n_steps):
        qo_pi_t = utils.obj_array(len(A))
        qo_pi.append(qo_pi_t)

    # compute expected observations over time
    for t in range(n_steps):
        for modality, A_m in enumerate(A):
            qo_pi[t][modality] = spm_dot(A_m, qs_pi[t])

    return qo_pi

def get_expected_obs_factorized(qs_pi, A, A_factor_list):
    """
    Compute the expected observations under a policy, also known as the posterior predictive density over observations

    Parameters
    ----------
    qs_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over hidden states expected under the policy, where ``qs_pi[t]`` stores the beliefs about
        hidden states expected under the policy at time ``t``
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    A_factor_list: ``list`` of ``list`` of ``int``
        List of lists of hidden state factor indices that each observation modality depends on. Each element ``A_factor_list[i]`` is a list of the factor indices that modality i's observation model depends on.
    Returns
    -------
    qo_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over observations expected under the policy, where ``qo_pi[t]`` stores the beliefs about
        observations expected under the policy at time ``t``
    """

    n_steps = len(qs_pi) # each element of the list is the PPD at a different timestep

    # initialise expected observations
    qo_pi = []

    for t in range(n_steps):
        qo_pi_t = utils.obj_array(len(A))
        qo_pi.append(qo_pi_t)

    # compute expected observations over time
    for t in range(n_steps):
        for modality, A_m in enumerate(A):
            factor_idx = A_factor_list[modality] # list of the hidden state factor indices that observation modality with the index `modality` depends on
            qo_pi[t][modality] = spm_dot(A_m, qs_pi[t][factor_idx])

    return qo_pi

def calc_expected_utility(qo_pi, C):
    """
    Computes the expected utility of a policy, using the observation distribution expected under that policy and a prior preference vector.

    Parameters
    ----------
    qo_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over observations expected under the policy, where ``qo_pi[t]`` stores the beliefs about
        observations expected under the policy at time ``t``
    C: ``numpy.ndarray`` of dtype object
       Prior over observations or 'prior preferences', storing the "value" of each outcome in terms of relative log probabilities. 
       This is softmaxed to form a proper probability distribution before being used to compute the expected utility.

    Returns
    -------
    expected_util: float
        Utility (reward) expected under the policy in question
    """
    n_steps = len(qo_pi)
    
    # initialise expected utility
    expected_util = 0

    # loop over time points and modalities
    num_modalities = len(C)

    # reformat C to be tiled across timesteps, if it's not already
    modalities_to_tile = [modality_i for modality_i in range(num_modalities) if C[modality_i].ndim == 1]

    # make a deepcopy of C where it has been tiled across timesteps
    C_tiled = copy.deepcopy(C)
    for modality in modalities_to_tile:
        C_tiled[modality] = np.tile(C[modality][:,None], (1, n_steps) )
    
    C_prob = softmax_obj_arr(C_tiled) # convert relative log probabilities into proper probability distribution

    for t in range(n_steps):
        for modality in range(num_modalities):

            lnC = spm_log_single(C_prob[modality][:, t])
            expected_util += qo_pi[t][modality].dot(lnC)

    return expected_util


def calc_states_info_gain(A, qs_pi):
    """
    Computes the Bayesian surprise or information gain about states of a policy, 
    using the observation model and the hidden state distribution expected under that policy.

    Parameters
    ----------
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    qs_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over hidden states expected under the policy, where ``qs_pi[t]`` stores the beliefs about
        hidden states expected under the policy at time ``t``

    Returns
    -------
    states_surprise: float
        Bayesian surprise (about states) or salience expected under the policy in question
    """

    n_steps = len(qs_pi)

    states_surprise = 0
    for t in range(n_steps):
        states_surprise += spm_MDP_G(A, qs_pi[t])

    return states_surprise

def calc_states_info_gain_factorized(A, qs_pi, A_factor_list):
    """
    Computes the Bayesian surprise or information gain about states of a policy, 
    using the observation model and the hidden state distribution expected under that policy.

    Parameters
    ----------
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    qs_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over hidden states expected under the policy, where ``qs_pi[t]`` stores the beliefs about
        hidden states expected under the policy at time ``t``
    A_factor_list: ``list`` of ``list`` of ``int``
        List of lists, where ``A_factor_list[m]`` is a list of the hidden state factor indices that observation modality with the index ``m`` depends on

    Returns
    -------
    states_surprise: float
        Bayesian surprise (about states) or salience expected under the policy in question
    """

    n_steps = len(qs_pi)

    states_surprise = 0
    for t in range(n_steps):
        for m, A_m in enumerate(A):
            factor_idx = A_factor_list[m] # list of the hidden state factor indices that observation modality with the index `m` depends on
            states_surprise += spm_MDP_G(A_m, qs_pi[t][factor_idx])

    return states_surprise


def calc_pA_info_gain(pA, qo_pi, qs_pi):
    """
    Compute expected Dirichlet information gain about parameters ``pA`` under a policy

    Parameters
    ----------
    pA: ``numpy.ndarray`` of dtype object
        Dirichlet parameters over observation model (same shape as ``A``)
    qo_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over observations expected under the policy, where ``qo_pi[t]`` stores the beliefs about
        observations expected under the policy at time ``t``
    qs_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over hidden states expected under the policy, where ``qs_pi[t]`` stores the beliefs about
        hidden states expected under the policy at time ``t``

    Returns
    -------
    infogain_pA: float
        Surprise (about Dirichlet parameters) expected under the policy in question
    """

    n_steps = len(qo_pi)
    
    num_modalities = len(pA)
    wA = utils.obj_array(num_modalities)
    for modality, pA_m in enumerate(pA):
        wA[modality] = spm_wnorm(pA[modality])

    pA_infogain = 0
    
    for modality in range(num_modalities):
        wA_modality = wA[modality] * (pA[modality] > 0).astype("float")
        for t in range(n_steps):
            pA_infogain -= qo_pi[t][modality].dot(spm_dot(wA_modality, qs_pi[t])[:, np.newaxis])

    return pA_infogain

def calc_pA_info_gain_factorized(pA, qo_pi, qs_pi, A_factor_list):
    """
    Compute expected Dirichlet information gain about parameters ``pA`` under a policy.
    In this version of the function, we assume that the observation model is factorized, i.e. that each observation modality depends on a subset of the hidden state factors.

    Parameters
    ----------
    pA: ``numpy.ndarray`` of dtype object
        Dirichlet parameters over observation model (same shape as ``A``)
    qo_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over observations expected under the policy, where ``qo_pi[t]`` stores the beliefs about
        observations expected under the policy at time ``t``
    qs_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over hidden states expected under the policy, where ``qs_pi[t]`` stores the beliefs about
        hidden states expected under the policy at time ``t``
    A_factor_list: ``list`` of ``list`` of ``int``
        List of lists, where ``A_factor_list[m]`` is a list of the hidden state factor indices that observation modality with the index ``m`` depends on

    Returns
    -------
    infogain_pA: float
        Surprise (about Dirichlet parameters) expected under the policy in question
    """

    n_steps = len(qo_pi)
    
    num_modalities = len(pA)
    wA = utils.obj_array(num_modalities)
    for modality, pA_m in enumerate(pA):
        wA[modality] = spm_wnorm(pA[modality])

    pA_infogain = 0
    
    for modality in range(num_modalities):
        wA_modality = wA[modality] * (pA[modality] > 0).astype("float")
        factor_idx = A_factor_list[modality]
        for t in range(n_steps):
            pA_infogain -= qo_pi[t][modality].dot(spm_dot(wA_modality, qs_pi[t][factor_idx])[:, np.newaxis])

    return pA_infogain

def calc_pB_info_gain(pB, qs_pi, qs_prev, policy):
    """
    Compute expected Dirichlet information gain about parameters ``pB`` under a given policy

    Parameters
    ----------
    pB: ``numpy.ndarray`` of dtype object
        Dirichlet parameters over transition model (same shape as ``B``)
    qs_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over hidden states expected under the policy, where ``qs_pi[t]`` stores the beliefs about
        hidden states expected under the policy at time ``t``
    qs_prev: ``numpy.ndarray`` of dtype object
        Posterior over hidden states at beginning of trajectory (before receiving observations)
    policy: 2D ``numpy.ndarray``
        Array that stores actions entailed by a policy over time. Shape is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    
    Returns
    -------
    infogain_pB: float
        Surprise (about dirichlet parameters) expected under the policy in question
    """

    n_steps = len(qs_pi)

    num_factors = len(pB)
    wB = utils.obj_array(num_factors)
    for factor, pB_f in enumerate(pB):
        wB[factor] = spm_wnorm(pB_f)

    pB_infogain = 0

    for t in range(n_steps):
        # the 'past posterior' used for the information gain about pB here is the posterior
        # over expected states at the timestep previous to the one under consideration
        # if we're on the first timestep, we just use the latest posterior in the
        # entire action-perception cycle as the previous posterior
        if t == 0:
            previous_qs = qs_prev
        # otherwise, we use the expected states for the timestep previous to the timestep under consideration
        else:
            previous_qs = qs_pi[t - 1]

        # get the list of action-indices for the current timestep
        policy_t = policy[t, :]
        for factor, a_i in enumerate(policy_t):
            wB_factor_t = wB[factor][:, :, int(a_i)] * (pB[factor][:, :, int(a_i)] > 0).astype("float")
            pB_infogain -= qs_pi[t][factor].dot(wB_factor_t.dot(previous_qs[factor]))

    return pB_infogain

def calc_pB_info_gain_interactions(pB, qs_pi, qs_prev, B_factor_list, policy):
    """
    Compute expected Dirichlet information gain about parameters ``pB`` under a given policy

    Parameters
    ----------
    pB: ``numpy.ndarray`` of dtype object
        Dirichlet parameters over transition model (same shape as ``B``)
    qs_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over hidden states expected under the policy, where ``qs_pi[t]`` stores the beliefs about
        hidden states expected under the policy at time ``t``
    qs_prev: ``numpy.ndarray`` of dtype object
        Posterior over hidden states at beginning of trajectory (before receiving observations)
    B_factor_list: ``list`` of ``list`` of ``int``
        List of lists, where ``B_factor_list[f]`` is a list of the hidden state factor indices that hidden state factor with the index ``f`` depends on
    policy: 2D ``numpy.ndarray``
        Array that stores actions entailed by a policy over time. Shape is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    
    Returns
    -------
    infogain_pB: float
        Surprise (about dirichlet parameters) expected under the policy in question
    """

    n_steps = len(qs_pi)

    num_factors = len(pB)
    wB = utils.obj_array(num_factors)
    for factor, pB_f in enumerate(pB):
        wB[factor] = spm_wnorm(pB_f)

    pB_infogain = 0

    for t in range(n_steps):
        # the 'past posterior' used for the information gain about pB here is the posterior
        # over expected states at the timestep previous to the one under consideration
        # if we're on the first timestep, we just use the latest posterior in the
        # entire action-perception cycle as the previous posterior
        if t == 0:
            previous_qs = qs_prev
        # otherwise, we use the expected states for the timestep previous to the timestep under consideration
        else:
            previous_qs = qs_pi[t - 1]

        # get the list of action-indices for the current timestep
        policy_t = policy[t, :]
        for factor, a_i in enumerate(policy_t):
            wB_factor_t = wB[factor][...,int(a_i)] * (pB[factor][...,int(a_i)] > 0).astype("float")
            f_idx = B_factor_list[factor]
            pB_infogain -= qs_pi[t][factor].dot(spm_dot(wB_factor_t, previous_qs[f_idx]))

    return pB_infogain

def calc_inductive_cost(qs, qs_pi, I, epsilon=1e-3):
    """
    Computes the inductive cost of a state.

    Parameters
    ----------
    qs: ``numpy.ndarray`` of dtype object
        Marginal posterior beliefs over hidden states at a given timepoint.
    qs_pi: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over hidden states expected under the policy, where ``qs_pi[t]`` stores the beliefs about
        states expected under the policy at time ``t``
    I: ``numpy.ndarray`` of dtype object
        For each state factor, contains a 2D ``numpy.ndarray`` whose element i,j yields the probability 
        of reaching the goal state backwards from state j after i steps.

    Returns
    -------
    inductive_cost: float
        Cost of visited this state using backwards induction under the policy in question
    """
    n_steps = len(qs_pi)
    
    # initialise inductive cost
    inductive_cost = 0

    # loop over time points and modalities
    num_factors = len(I)

    for t in range(n_steps):
        for factor in range(num_factors):
            # we also assume precise beliefs here?!
            idx = np.argmax(qs[factor])
            # m = arg max_n p_n < sup p
            # i.e. find first I idx equals 1 and m is the index before
            m = np.where(I[factor][:, idx] == 1)[0]
            # we might find no path to goal (i.e. when no goal specified)
            if len(m) > 0:
                m = np.max(m[0]-1, 0)
                I_m = (1-I[factor][m, :]) * np.log(epsilon)
                inductive_cost += I_m.dot(qs_pi[t][factor])
                
    return inductive_cost

def construct_policies(num_states, num_controls = None, policy_len=1, control_fac_idx=None):
    """
    Generate a ``list`` of policies. The returned array ``policies`` is a ``list`` that stores one policy per entry.
    A particular policy (``policies[i]``) has shape ``(num_timesteps, num_factors)`` 
    where ``num_timesteps`` is the temporal depth of the policy and ``num_factors`` is the number of control factors.

    Parameters
    ----------
    num_states: ``list`` of ``int``
        ``list`` of the dimensionalities of each hidden state factor
    num_controls: ``list`` of ``int``, default ``None``
        ``list`` of the dimensionalities of each control state factor. If ``None``, then is automatically computed as the dimensionality of each hidden state factor that is controllable
    policy_len: ``int``, default 1
        temporal depth ("planning horizon") of policies
    control_fac_idx: ``list`` of ``int``
        ``list`` of indices of the hidden state factors that are controllable (i.e. those state factors ``i`` where ``num_controls[i] > 1``)

    Returns
    ----------
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy as a 2D array in ``policies[p_idx]``. Shape of ``policies[p_idx]`` 
        is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    """

    num_factors = len(num_states)
    if control_fac_idx is None:
        if num_controls is not None:
            control_fac_idx = [f for f, n_c in enumerate(num_controls) if n_c > 1]
        else:
            control_fac_idx = list(range(num_factors))

    if num_controls is None:
        num_controls = [num_states[c_idx] if c_idx in control_fac_idx else 1 for c_idx in range(num_factors)]
        
    x = num_controls * policy_len
    policies = list(itertools.product(*[list(range(i)) for i in x]))
    for pol_i in range(len(policies)):
        policies[pol_i] = np.array(policies[pol_i]).reshape(policy_len, num_factors)

    return policies
    
def get_num_controls_from_policies(policies):
    """
    Calculates the ``list`` of dimensionalities of control factors (``num_controls``)
    from the ``list`` or array of policies. This assumes a policy space such that for each control factor, there is at least
    one policy that entails taking the action with the maximum index along that control factor.

    Parameters
    ----------
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy as a 2D array in ``policies[p_idx]``. Shape of ``policies[p_idx]`` 
        is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    
    Returns
    ----------
    num_controls: ``list`` of ``int``
        ``list`` of the dimensionalities of each control state factor, computed here automatically from a ``list`` of policies.
    """

    return list(np.max(np.vstack(policies), axis = 0) + 1)
    

def sample_action(q_pi, policies, num_controls, action_selection="deterministic", alpha = 16.0):
    """
    Computes the marginal posterior over actions and then samples an action from it, one action per control factor.

    Parameters
    ----------
    q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy as a 2D array in ``policies[p_idx]``. Shape of ``policies[p_idx]`` 
        is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    num_controls: ``list`` of ``int``
        ``list`` of the dimensionalities of each control state factor.
    action_selection: ``str``, default "deterministic"
        String indicating whether whether the selected action is chosen as the maximum of the posterior over actions,
        or whether it's sampled from the posterior marginal over actions
    alpha: ``float``, default 16.0
        Action selection precision -- the inverse temperature of the softmax that is used to scale the 
        action marginals before sampling. This is only used if ``action_selection`` argument is "stochastic"
   
    Returns
    ----------
    selected_policy: 1D ``numpy.ndarray``
        Vector containing the indices of the actions for each control factor
    """

    num_factors = len(num_controls)

    action_marginals = utils.obj_array_zeros(num_controls)

    # weight each action according to its integrated posterior probability under all policies at the current timestep
    for pol_idx, policy in enumerate(policies):
        for factor_i, action_i in enumerate(policy[0, :]):
            action_marginals[factor_i][action_i] += q_pi[pol_idx]
    
    action_marginals = utils.norm_dist_obj_arr(action_marginals)

    selected_policy = np.zeros(num_factors)
    for factor_i in range(num_factors):

        # Either you do this:
        if action_selection == 'deterministic':
            selected_policy[factor_i] = select_highest(action_marginals[factor_i])
        elif action_selection == 'stochastic':
            log_marginal_f = spm_log_single(action_marginals[factor_i])
            p_actions = softmax(log_marginal_f * alpha)
            selected_policy[factor_i] = utils.sample(p_actions)

    return selected_policy

def _sample_action_test(q_pi, policies, num_controls, action_selection="deterministic", alpha = 16.0, seed=None):
    """
    Computes the marginal posterior over actions and then samples an action from it, one action per control factor.
    Internal testing version that returns the marginal posterior over actions, and also has a seed argument for reproducibility.

    Parameters
    ----------
    q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy as a 2D array in ``policies[p_idx]``. Shape of ``policies[p_idx]`` 
        is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    num_controls: ``list`` of ``int``
        ``list`` of the dimensionalities of each control state factor.
    action_selection: ``str``, default "deterministic"
        String indicating whether whether the selected action is chosen as the maximum of the posterior over actions,
        or whether it's sampled from the posterior marginal over actions
    alpha: float, default 16.0
        Action selection precision -- the inverse temperature of the softmax that is used to scale the 
        action marginals before sampling. This is only used if ``action_selection`` argument is "stochastic"
    seed: ``int``, default None
        The seed can be set to control the random sampling that occurs when ``action_selection`` is "deterministic" but there are more than one actions with the same maximum posterior probability.


    Returns
    ----------
    selected_policy: 1D ``numpy.ndarray``
        Vector containing the indices of the actions for each control factor
    p_actions: ``numpy.ndarray`` of dtype object
        Marginal posteriors over actions, after softmaxing and scaling with action precision. This distribution will be used to sample actions,
        if``action_selection`` argument is "stochastic"
    """

    num_factors = len(num_controls)

    action_marginals = utils.obj_array_zeros(num_controls)
    
    # weight each action according to its integrated posterior probability under all policies at the current timestep
    for pol_idx, policy in enumerate(policies):
        for factor_i, action_i in enumerate(policy[0, :]):
            action_marginals[factor_i][action_i] += q_pi[pol_idx]
    
    action_marginals = utils.norm_dist_obj_arr(action_marginals)

    selected_policy = np.zeros(num_factors)
    p_actions = utils.obj_array_zeros(num_controls)
    for factor_i in range(num_factors):
        if action_selection == 'deterministic':
            p_actions[factor_i] = action_marginals[factor_i]
            selected_policy[factor_i] = _select_highest_test(p_actions[factor_i], seed=seed)
        elif action_selection == 'stochastic':
            log_marginal_f = spm_log_single(action_marginals[factor_i])
            p_actions[factor_i] = softmax(log_marginal_f * alpha)
            selected_policy[factor_i] = utils.sample(p_actions[factor_i])

    return selected_policy, p_actions

def sample_policy(q_pi, policies, num_controls, action_selection="deterministic", alpha = 16.0):
    """
    Samples a policy from the posterior over policies, taking the action (per control factor) entailed by the first timestep of the selected policy.

    Parameters
    ----------
    q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy as a 2D array in ``policies[p_idx]``. Shape of ``policies[p_idx]`` 
        is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    num_controls: ``list`` of ``int``
        ``list`` of the dimensionalities of each control state factor.
    action_selection: string, default "deterministic"
        String indicating whether whether the selected policy is chosen as the maximum of the posterior over policies,
        or whether it's sampled from the posterior over policies.
    alpha: float, default 16.0
        Action selection precision -- the inverse temperature of the softmax that is used to scale the 
        policy posterior before sampling. This is only used if ``action_selection`` argument is "stochastic"

    Returns
    ----------
    selected_policy: 1D ``numpy.ndarray``
        Vector containing the indices of the actions for each control factor
    """

    num_factors = len(num_controls)

    if action_selection == "deterministic":
        policy_idx = select_highest(q_pi)
    elif action_selection == "stochastic":
        log_qpi = spm_log_single(q_pi)
        p_policies = softmax(log_qpi * alpha)
        policy_idx = utils.sample(p_policies)

    selected_policy = np.zeros(num_factors)
    for factor_i in range(num_factors):
        selected_policy[factor_i] = policies[policy_idx][0, factor_i]

    return selected_policy

def _sample_policy_test(q_pi, policies, num_controls, action_selection="deterministic", alpha = 16.0, seed=None):
    """
    Test version of sampling a policy from the posterior over policies, taking the action (per control factor) entailed by the first timestep of the selected policy.
    This test version also returns the probability distribution over policies, and also has a seed argument for reproducibility.
    Parameters
    ----------
    q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy as a 2D array in ``policies[p_idx]``. Shape of ``policies[p_idx]`` 
        is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    num_controls: ``list`` of ``int``
        ``list`` of the dimensionalities of each control state factor.
    action_selection: string, default "deterministic"
        String indicating whether whether the selected policy is chosen as the maximum of the posterior over policies,
        or whether it's sampled from the posterior over policies.
    alpha: float, default 16.0
        Action selection precision -- the inverse temperature of the softmax that is used to scale the 
        policy posterior before sampling. This is only used if ``action_selection`` argument is "stochastic"
    seed: ``int``, default None
        The seed can be set to control the random sampling that occurs when ``action_selection`` is "deterministic" but there are more than one actions with the same maximum posterior probability.

   
    Returns
    ----------
    selected_policy: 1D ``numpy.ndarray``
        Vector containing the indices of the actions for each control factor
    """

    num_factors = len(num_controls)

    if action_selection == "deterministic":
        p_policies = q_pi
        policy_idx = _select_highest_test(p_policies, seed=seed)
    elif action_selection == "stochastic":
        log_qpi = spm_log_single(q_pi)
        p_policies = softmax(log_qpi * alpha)
        policy_idx = utils.sample(p_policies)

    selected_policy = np.zeros(num_factors)
    for factor_i in range(num_factors):
        selected_policy[factor_i] = policies[policy_idx][0, factor_i]

    return selected_policy, p_policies


def select_highest(options_array):
    """
    Selects the highest value among the provided ones. If the higher value is more than once and they're closer than 1e-5, a random choice is made.
    Parameters
    ----------
    options_array: ``numpy.ndarray``
        The array to examine

    Returns
    -------
    The highest value in the given list
    """
    options_with_idx = np.array(list(enumerate(options_array)))
    same_prob = options_with_idx[
                    abs(options_with_idx[:, 1] - np.amax(options_with_idx[:, 1])) <= 1e-8][:, 0]
    if len(same_prob) > 1:
        # If some of the most likely actions have nearly equal probability, sample from this subset of actions, instead of using argmax
        return int(same_prob[np.random.choice(len(same_prob))])

    return int(same_prob[0])

def _select_highest_test(options_array, seed=None):
    """
    (Test version with seed argument for reproducibility) Selects the highest value among the provided ones. If the higher value is more than once and they're closer than 1e-8, a random choice is made.
    Parameters
    ----------
    options_array: ``numpy.ndarray``
        The array to examine

    Returns
    -------
    The highest value in the given list
    """
    options_with_idx = np.array(list(enumerate(options_array)))
    same_prob = options_with_idx[
                    abs(options_with_idx[:, 1] - np.amax(options_with_idx[:, 1])) <= 1e-8][:, 0]
    if len(same_prob) > 1:
        # If some of the most likely actions have nearly equal probability, sample from this subset of actions, instead of using argmax
        rng = np.random.default_rng(seed)
        return int(same_prob[rng.choice(len(same_prob))])

    return int(same_prob[0])


def backwards_induction(H, B, B_factor_list, threshold, depth):
    """
    Runs backwards induction of reaching a goal state H given a transition model B.
    
    Parameters
    ----------    
    H: ``numpy.ndarray`` of dtype object
       Prior over states
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    B_factor_list: ``list`` of ``list`` of ``int``
        List of lists of hidden state factors each hidden state factor depends on. Each element ``B_factor_list[i]`` is a list of the factor indices that factor i's dynamics depend on.
    threshold: ``float``
        The threshold for pruning transitions that are below a certain probability
    depth: ``int``
        The temporal depth of the backward induction

    Returns
    ----------
    I: ``numpy.ndarray`` of dtype object
        For each state factor, contains a 2D ``numpy.ndarray`` whose element i,j yields the probability 
        of reaching the goal state backwards from state j after i steps.
    """
    # TODO can this be done with arbitrary B_factor_list?
    
    num_factors = len(H)
    I = utils.obj_array(num_factors)
    for factor in range(num_factors):
        I[factor] = np.zeros((depth, H[factor].shape[0]))
        I[factor][0, :] = H[factor]

        bf = factor
        if B_factor_list is not None:
            if len(B_factor_list[factor]) > 1:
                raise ValueError("Backwards induction with factorized transition model not yet implemented")
            bf = B_factor_list[factor][0]

        num_states, _, _ = B[bf].shape
        b = np.zeros((num_states, num_states))
        
        for state in range(num_states):
            for next_state in range(num_states):
                # If there exists an action that allows transitioning 
                # from state to next_state, with probability larger than threshold
                # set b[state, next_state] to 1
                if np.any(B[bf][next_state, state, :] > threshold):
                    b[next_state, state] = 1

        for i in range(1, depth):
            I[factor][i, :] = np.dot(b, I[factor][i-1, :])
            I[factor][i, :] = np.where(I[factor][i, :] > 0.1, 1.0, 0.0)

    return I
    

