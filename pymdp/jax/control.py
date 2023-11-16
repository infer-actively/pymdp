#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-an-iterable

import itertools
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Tuple, Optional
from functools import partial
from jax import lax, jit, vmap, nn
from itertools import chain

from pymdp.jax.maths import *
# import pymdp.jax.utils as utils

def get_marginals(q_pi, policies, num_controls):
    """
    Computes the marginal posterior(s) over actions by integrating their posterior probability under the policies that they appear within.

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
    
    Returns
    ----------
    action_marginals: ``list`` of ``jax.numpy.ndarrays``
       List of arrays corresponding to marginal probability of each action possible action
    """
    num_factors = len(num_controls)

    action_marginals = []
    for factor_i in range(num_factors):
        actions = jnp.arange(num_controls[factor_i])[:, None]
        action_marginals.append(jnp.where(actions==policies[:, 0, factor_i], q_pi, 0).sum(-1))
    
    return action_marginals


def sample_action(q_pi, policies, num_controls, action_selection="deterministic", alpha=16.0, rng_key=None):
    """
    Samples an action from posterior marginals, one action per control factor.

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
        String indicating whether whether the selected action is chosen as the maximum of the posterior over actions,
        or whether it's sampled from the posterior marginal over actions
    alpha: float, default 16.0
        Action selection precision -- the inverse temperature of the softmax that is used to scale the 
        action marginals before sampling. This is only used if ``action_selection`` argument is "stochastic"

    Returns
    ----------
    selected_policy: 1D ``numpy.ndarray``
        Vector containing the indices of the actions for each control factor
    """

    marginal = get_marginals(q_pi, policies, num_controls)
    
    if action_selection == 'deterministic':
        selected_policy = jtu.tree_map(lambda x: jnp.argmax(x, -1), marginal)
    elif action_selection == 'stochastic':
        selected_policy = jtu.tree_map( lambda x: random.categorical(rng_key, alpha * log_stable(x)), marginal)
    else:
        raise NotImplementedError

    return jnp.array(selected_policy)


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
        policies[pol_i] = jnp.array(policies[pol_i]).reshape(policy_len, num_factors)

    return jnp.stack(policies)


def update_posterior_policies(policy_matrix, qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, gamma=16.0, use_utility=True, use_states_info_gain=True, use_param_info_gain=False):
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    ## vmap across policies
    compute_G_fixed_states = partial(compute_G_policy, qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies,
                                     use_utility=use_utility, use_states_info_gain=use_states_info_gain, use_param_info_gain=use_param_info_gain)

    # only in the case of policy-dependent qs_inits
    # in_axes_list = (1,) * n_factors
    # all_efe_of_policies = vmap(compute_G_policy, in_axes=(in_axes_list, 0))(qs_init_pi, policy_matrix)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    neg_efe_all_policies = vmap(compute_G_fixed_states)(policy_matrix)

    return nn.softmax(gamma * neg_efe_all_policies), neg_efe_all_policies

def compute_expected_state(qs_prior, B, u_t, B_dependencies=None): 
    """
    Compute posterior over next state, given belief about previous state, transition model and action...
    """
    assert len(u_t) == len(B)  
    qs_next = []
    for B_f, u_f, deps in zip(B, u_t, B_dependencies):
        # qs_next.append( B_f[..., u_f].dot(qs_f) )
        qs_next_f = factor_dot(B_f[...,u_f], qs_prior[deps])
        qs_next.append(qs_next_f)
        
    return qs_next

def compute_expected_state_and_Bs(qs_prior, B, u_t): 
    """
    Compute posterior over next state, given belief about previous state, transition model and action...
    """
    assert len(u_t) == len(B)  
    qs_next = []
    Bs = []
    for qs_f, B_f, u_f in zip(qs_prior, B, u_t):
        qs_next.append( B_f[..., u_f].dot(qs_f) )
        Bs.append(B_f[..., u_f])
    
    return qs_next, Bs

def compute_expected_obs(qs, A, A_dependencies):
    """
    New version of expected observation (computation of Q(o|pi)) that takes into account sparse dependencies between observation
    modalities and hidden state factors
    """
        
    def compute_expected_obs_modality(A_m, m):
        deps = A_dependencies[m]
        relevant_factors = [qs[idx] for idx in deps]
        return factor_dot(A_m, relevant_factors, keep_dims=(0,))

    return jtu.tree_map(compute_expected_obs_modality, A, list(range(len(A))))

def compute_info_gain(qs, qo, A, A_dependencies):
    """
    New version of expected information gain that takes into account sparse dependencies between observation modalities and hidden state factors.
    """

    def compute_info_gain_for_modality(qo_m, A_m, m):
        H_qo = - (qo_m * log_stable(qo_m)).sum()
        H_A_m = - (A_m * log_stable(A_m)).sum(0)
        deps = A_dependencies[m]
        relevant_factors = [qs[idx] for idx in deps]
        qs_H_A_m = factor_dot(H_A_m, relevant_factors)
        return H_qo - qs_H_A_m
    
    info_gains_per_modality = jtu.tree_map(compute_info_gain_for_modality, qo, A, list(range(len(A))))
        
    return jtu.tree_reduce(lambda x,y: x+y, info_gains_per_modality)

# qs_H_A = 0 # expected entropy of the likelihood, under Q(s)
# H_qo = 0 # marginal entropy of Q(o)
# for a, o, deps in zip(A, qo, A_dependencies):
#     relevant_factors = jtu.tree_map(lambda idx: qs[idx], deps)
#     qs_joint_relevant = relevant_factors[0]
#     for q in relevant_factors[1:]:
#         qs_joint_relevant = jnp.expand_dims(qs_joint_relevant, -1) * q
#     H_A_m = -(a * log_stable(a)).sum(0)
#     qs_H_A += (H_A_m * qs_joint_relevant).sum()

#     H_qo -= (o * log_stable(o)).sum()

def compute_expected_utility(qo, C):
    
    util = 0.
    for o_m, C_m in zip(qo, C):
        util += (o_m * C_m).sum()
    
    return util

def calc_pA_info_gain(pA, qo, qs):
    """
    Compute expected Dirichlet information gain about parameters ``pA`` for a given posterior predictive distribution over observations ``qo`` and states ``qs``.

    Parameters
    ----------
    pA: ``numpy.ndarray`` of dtype object
        Dirichlet parameters over observation model (same shape as ``A``)
    qo: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over observations; stores the beliefs about
        observations expected under the policy at some arbitrary time ``t``
    qs: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over hidden states, stores the beliefs about
        hidden states expected under the policy at some arbitrary time ``t``

    Returns
    -------
    infogain_pA: float
        Surprise (about Dirichlet parameters) expected for the pair of posterior predictive distributions ``qo`` and ``qs``
    """

    wA = jtu.tree_map(spm_wnorm, pA)    
    wA_per_modality = jtu.tree_map(lambda wa, pa: wa * (pa > 0.), wA, pA)
    pA_infogain_per_modality = jtu.tree_map(lambda wa, qo: qo.dot(factor_dot(wa, qs)[...,None]), wA_per_modality, qo)
    infogain_pA = jtu.tree_reduce(lambda x, y: x + y, pA_infogain_per_modality)[0]
    return infogain_pA

def calc_pB_info_gain(pB, qs_t, qs_t_minus_1):
    """ Placeholder, not implemented yet """
    # """
    # Compute expected Dirichlet information gain about parameters ``pB`` under a given policy

    # Parameters
    # ----------
    # pB: ``numpy.ndarray`` of dtype object
    #     Dirichlet parameters over transition model (same shape as ``B``)
    # qs_pi: ``list`` of ``numpy.ndarray`` of dtype object
    #     Predictive posterior beliefs over hidden states expected under the policy, where ``qs_pi[t]`` stores the beliefs about
    #     hidden states expected under the policy at time ``t``
    # qs_prev: ``numpy.ndarray`` of dtype object
    #     Posterior over hidden states at beginning of trajectory (before receiving observations)
    # policy: 2D ``numpy.ndarray``
    #     Array that stores actions entailed by a policy over time. Shape is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
    #     depth of the policy and ``num_factors`` is the number of control factors.
    
    # Returns
    # -------
    # infogain_pB: float
    #     Surprise (about dirichlet parameters) expected under the policy in question
    # """

    # n_steps = len(qs_pi)

    # num_factors = len(pB)
    # wB = utils.obj_array(num_factors)
    # for factor, pB_f in enumerate(pB):
    #     wB[factor] = spm_wnorm(pB_f)

    # pB_infogain = 0

    # for t in range(n_steps):
    #     # the 'past posterior' used for the information gain about pB here is the posterior
    #     # over expected states at the timestep previous to the one under consideration
    #     # if we're on the first timestep, we just use the latest posterior in the
    #     # entire action-perception cycle as the previous posterior
    #     if t == 0:
    #         previous_qs = qs_prev
    #     # otherwise, we use the expected states for the timestep previous to the timestep under consideration
    #     else:
    #         previous_qs = qs_pi[t - 1]

    #     # get the list of action-indices for the current timestep
    #     policy_t = policy[t, :]
    #     for factor, a_i in enumerate(policy_t):
    #         wB_factor_t = wB[factor][:, :, int(a_i)] * (pB[factor][:, :, int(a_i)] > 0).astype("float")
    #         pB_infogain -= qs_pi[t][factor].dot(wB_factor_t.dot(previous_qs[factor]))
    return 0.

def compute_G_policy(qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, policy_i, use_utility=True, use_states_info_gain=True, use_param_info_gain=False):
    """ Write a version of compute_G_policy that does the same computations as `compute_G_policy` but using `lax.scan` instead of a for loop. """

    def scan_body(carry, t):

        qs, neg_G = carry

        qs_next = compute_expected_state(qs, B, policy_i[t], B_dependencies)

        qo = compute_expected_obs(qs_next, A, A_dependencies)

        info_gain = compute_info_gain(qs_next, qo, A) if use_states_info_gain else 0.

        utility = compute_expected_utility(qo, C) if use_utility else 0.

        param_info_gain = calc_pA_info_gain(pA, qo, qs_next) if use_param_info_gain else 0.
        param_info_gain += calc_pB_info_gain(pB, qs_next, qs) if use_param_info_gain else 0.

        neg_G += info_gain + utility + param_info_gain

        return (qs_next, neg_G), None

    qs = qs_init
    neg_G = 0.
    final_state, _ = lax.scan(scan_body, (qs, neg_G), jnp.arange(policy_i.shape[0]))
    qs_final, neg_G = final_state
    return neg_G


# if __name__ == '__main__':

#     from jax import random
#     key = random.PRNGKey(1)
#     num_obs = [3, 4]

#     A = [random.uniform(key, shape = (no, 2, 2)) for no in num_obs]
#     B = [random.uniform(key, shape = (2, 2, 2)), random.uniform(key, shape = (2, 2, 2))]
#     C = [log_stable(jnp.array([0.8, 0.1, 0.1])), log_stable(jnp.ones(4)/4)]
#     policy_1 = jnp.array([[0, 1],
#                          [1, 1]])
#     policy_2 = jnp.array([[1, 0],
#                          [0, 0]])
#     policy_matrix = jnp.stack([policy_1, policy_2]) # 2 x 2 x 2 tensor
    
#     qs_init = [jnp.ones(2)/2, jnp.ones(2)/2]
#     neg_G_all_policies = jit(update_posterior_policies)(policy_matrix, qs_init, A, B, C)
#     print(neg_G_all_policies)
