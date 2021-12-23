#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

import numpy as np

from pymdp import utils
from pymdp.maths import get_joint_likelihood_seq
from pymdp.algos import run_vanilla_fpi, run_mmp, _run_mmp_testing

VANILLA = "VANILLA"
VMP = "VMP"
MMP = "MMP"
BP = "BP"
EP = "EP"
CV = "CV"

def update_posterior_states_full(
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
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``numpy.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    prev_obs: ``list``
        List of observations over time. Each observation in the list can be an ``int``, a ``list`` of ints, a ``tuple`` of ints, a one-hot vector or an object array of one-hot vectors.
    policies: ``list`` of 2D ``numpy.ndarray``
        List that stores each policy in ``policies[p_idx]``. Shape of ``policies[p_idx]`` is ``(num_timesteps, num_factors)`` where `num_timesteps` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    prior: ``numpy.ndarray`` of dtype object, default ``None``
        If provided, this a ``numpy.ndarray`` of dtype object, with one sub-array per hidden state factor, that stores the prior beliefs about initial states. 
        If ``None``, this defaults to a flat (uninformative) prior over hidden states.
    policy_sep_prior: ``Bool``, default ``True``
        Flag determining whether the prior beliefs from the past are unconditioned on policy, or separated by /conditioned on the policy variable.
    **kwargs: keyword arguments
        Optional keyword arguments for the function ``algos.mmp.run_mmp``

    Returns
    ---------
    qs_seq_pi: ``numpy.ndarray`` of dtype object
        Posterior beliefs over hidden states for each policy. Nesting structure is policies, timepoints, factors,
        where e.g. ``qs_seq_pi[p][t][f]`` stores the marginal belief about factor ``f`` at timepoint ``t`` under policy ``p``.
    F: 1D ``numpy.ndarray``
        Vector of variational free energies for each policy
    """

    num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A, B)
    
    prev_obs = utils.process_observation_seq(prev_obs, num_modalities, num_obs)
   
    lh_seq = get_joint_likelihood_seq(A, prev_obs, num_states)

    if prev_actions is not None:
        prev_actions = np.stack(prev_actions,0)

    qs_seq_pi = utils.obj_array(len(policies))
    F = np.zeros(len(policies)) # variational free energy of policies

    for p_idx, policy in enumerate(policies):

            # get sequence and the free energy for policy
            qs_seq_pi[p_idx], F[p_idx] = run_mmp(
                lh_seq,
                B,
                policy,
                prev_actions=prev_actions,
                prior= prior[p_idx] if policy_sep_prior else prior, 
                **kwargs
            )

    return qs_seq_pi, F

def _update_posterior_states_full_test(
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
    Update posterior over hidden states using marginal message passing (TEST VERSION, with extra returns for benchmarking).

    Parameters
    ----------
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``np.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    B: ``numpy.ndarray`` of dtype object
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    prev_obs: list
        List of observations over time. Each observation in the list can be an ``int``, a ``list`` of ints, a ``tuple`` of ints, a one-hot vector or an object array of one-hot vectors.
    prior: ``numpy.ndarray`` of dtype object, default None
        If provided, this a ``numpy.ndarray`` of dtype object, with one sub-array per hidden state factor, that stores the prior beliefs about initial states. 
        If ``None``, this defaults to a flat (uninformative) prior over hidden states.
    policy_sep_prior: Bool, default True
        Flag determining whether the prior beliefs from the past are unconditioned on policy, or separated by /conditioned on the policy variable.
    **kwargs: keyword arguments
        Optional keyword arguments for the function ``algos.mmp.run_mmp``

    Returns
    --------
    qs_seq_pi: ``numpy.ndarray`` of dtype object
        Posterior beliefs over hidden states for each policy. Nesting structure is policies, timepoints, factors,
        where e.g. ``qs_seq_pi[p][t][f]`` stores the marginal belief about factor ``f`` at timepoint ``t`` under policy ``p``.
    F: 1D ``numpy.ndarray``
        Vector of variational free energies for each policy
    xn_seq_pi: ``numpy.ndarray`` of dtype object
        Posterior beliefs over hidden states for each policy, for each iteration of marginal message passing.
        Nesting structure is policy, iteration, factor, so ``xn_seq_p[p][itr][f]`` stores the ``num_states x infer_len`` 
        array of beliefs about hidden states at different time points of inference horizon.
    vn_seq_pi: `numpy.ndarray`` of dtype object
        Prediction errors over hidden states for each policy, for each iteration of marginal message passing.
        Nesting structure is policy, iteration, factor, so ``vn_seq_p[p][itr][f]`` stores the ``num_states x infer_len`` 
        array of beliefs about hidden states at different time points of inference horizon.
    """

    num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A, B)

    prev_obs = utils.process_observation_seq(prev_obs, num_modalities, num_obs)
    
    lh_seq = get_joint_likelihood_seq(A, prev_obs, num_states)

    if prev_actions is not None:
        prev_actions = np.stack(prev_actions,0)

    qs_seq_pi = utils.obj_array(len(policies))
    xn_seq_pi = utils.obj_array(len(policies))
    vn_seq_pi = utils.obj_array(len(policies))
    F = np.zeros(len(policies)) # variational free energy of policies

    for p_idx, policy in enumerate(policies):

            # get sequence and the free energy for policy
            qs_seq_pi[p_idx], F[p_idx], xn_seq_pi[p_idx], vn_seq_pi[p_idx] = _run_mmp_testing(
                lh_seq,
                B,
                policy,
                prev_actions=prev_actions,
                prior=prior[p_idx] if policy_sep_prior else prior, 
                **kwargs
            )

    return qs_seq_pi, F, xn_seq_pi, vn_seq_pi

def average_states_over_policies(qs_pi, q_pi):
    """
    This function computes a expected posterior over hidden states with respect to the posterior over policies, 
    also known as the 'Bayesian model average of states with respect to policies'.

    Parameters
    ----------
    qs_pi: ``numpy.ndarray`` of dtype object
        Posterior beliefs over hidden states for each policy. Nesting structure is policies, factors,
        where e.g. ``qs_pi[p][f]`` stores the marginal belief about factor ``f`` under policy ``p``.
    q_pi: ``numpy.ndarray`` of dtype object
        Posterior beliefs about policies where ``len(q_pi) = num_policies``

    Returns
    ---------
    qs_bma: ``numpy.ndarray`` of dtype object
        Marginal posterior over hidden states for the current timepoint, 
        averaged across policies according to their posterior probability given by ``q_pi``
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
    FPI or Fixed point iteration. 

    See the following links for details:
    http://www.cs.cmu.edu/~guestrin/Class/10708/recitations/r9/VI-view.pdf, slides 13- 18, and http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.137.221&rep=rep1&type=pdf, slides 24 - 38.
    
    Parameters
    ----------
    A: ``numpy.ndarray`` of dtype object
        Sensory likelihood mapping or 'observation model', mapping from hidden states to observations. Each element ``A[m]`` of
        stores an ``np.ndarray`` multidimensional array for observation modality ``m``, whose entries ``A[m][i, j, k, ...]`` store 
        the probability of observation level ``i`` given hidden state levels ``j, k, ...``
    obs: 1D ``numpy.ndarray``, ``numpy.ndarray`` of dtype object, int or tuple
        The observation (generated by the environment). If single modality, this can be a 1D ``np.ndarray``
        (one-hot vector representation) or an ``int`` (observation index)
        If multi-modality, this can be ``np.ndarray`` of dtype object whose entries are 1D one-hot vectors,
        or a tuple (of ``int``)
    prior: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object, default None
        Prior beliefs about hidden states, to be integrated with the marginal likelihood to obtain
        a posterior distribution. If not provided, prior is set to be equal to a flat categorical distribution (at the level of
        the individual inference functions).
    **kwargs: keyword arguments 
        List of keyword/parameter arguments corresponding to parameter values for the fixed-point iteration
        algorithm ``algos.fpi.run_vanilla_fpi.py``

    Returns
    ----------
    qs: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
        Marginal posterior beliefs over hidden states at current timepoint
    """

    num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A = A)
    
    obs = utils.process_observation(obs, num_modalities, num_obs)

    if prior is not None:
        prior = utils.to_obj_array(prior)

    return run_vanilla_fpi(A, obs, num_obs, num_states, prior, **kwargs)
   
