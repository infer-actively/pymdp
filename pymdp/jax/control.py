#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-an-iterable

import itertools
import jax.numpy as jnp
import jax.tree_util as jtu
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


def update_posterior_policies(policy_matrix, qs_init, A, B, C, gamma=16.0):
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    ## vmap across policies
    compute_G_fixed_states = partial(compute_G_policy, qs_init, A, B, C)

    # only in the case of policy-dependent qs_inits
    # in_axes_list = (1,) * n_factors
    # all_efe_of_policies = vmap(compute_G_policy, in_axes=(in_axes_list, 0))(qs_init_pi, policy_matrix)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    neg_efe_all_policies = vmap(compute_G_fixed_states)(policy_matrix)

    return nn.softmax(gamma * neg_efe_all_policies), neg_efe_all_policies

def compute_expected_state(qs_prior, B, u_t): 
    """
    Compute posterior over next state, given belief about previous state, transition model and action...
    """
    assert len(u_t) == len(B)  
    qs_next = []
    for qs_f, B_f, u_f in zip(qs_prior, B, u_t):
        qs_next.append( B_f[..., u_f].dot(qs_f) )
        
    return qs_next

def factor_dot(A, qs):
    """ Dot product of a multidimensional array with `x`.
    
    Parameters
    ----------
    - `x` [1D numpy.ndarray] - either vector or array of arrays
        The alternative array to perform the dot product with
    
    Returns 
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """
    
    dims = list(range(A.ndim - len(qs),len(qs)+A.ndim - len(qs)))

    arg_list = [A, list(range(A.ndim))] + list(chain(*([qs[f],[dims[f]]] for f in range(len(qs))))) + [[0]]

    res = jnp.einsum(*arg_list)

    return res

def compute_expected_obs(qs, A):

    qo = []
    for A_m in A:
        qo.append( factor_dot(A_m, qs) )

    return qo

def compute_info_gain(qs, qo, A):
    
    x = qs[0]
    for q in qs[1:]:
        x = jnp.expand_dims(x, -1) * q

    qs_H_A = 0 # expected entropy of the likelihood, under Q(s)
    H_qo = 0 # marginal entropy of Q(o)
    for a, o in zip(A, qo):
        qs_H_A -= (a * log_stable(a)).sum(0)
        H_qo -= (o * log_stable(o)).sum()
    
    return H_qo - (qs_H_A * x).sum()
    
def compute_expected_utility(qo, C):
    
    util = 0.
    for o_m, C_m in zip(qo, C):
        util += (o_m * C_m).sum()
    
    return util

def compute_G_policy(qs_init, A, B, C, policy_i):

    qs = qs_init
    neg_G = 0.
    for t_step in range(policy_i.shape[0]):

        qs = compute_expected_state(qs, B, policy_i[t_step])

        qo = compute_expected_obs(qs, A)

        info_gain = compute_info_gain(qs, qo, A)
        utility = compute_expected_utility(qo, C)

        # if we're doing scan we'll need some of those control-flow workarounds from lax
        # jnp.where(conditition, f_eval_if_true, 0)
        # calculate pA info gain
        # calculate pB info gain
        
        # Q(s, A) = E_{Q(o)}[D_KL(Q(s|o, \pi) Q(A| o, pi)|| Q(s|pi) Q(A))]

        neg_G += info_gain + utility

    return neg_G


if __name__ == '__main__':

    from jax import random
    key = random.PRNGKey(1)
    num_obs = [3, 4]

    A = [random.uniform(key, shape = (no, 2, 2)) for no in num_obs]
    B = [random.uniform(key, shape = (2, 2, 2)), random.uniform(key, shape = (2, 2, 2))]
    C = [log_stable(jnp.array([0.8, 0.1, 0.1])), log_stable(jnp.ones(4)/4)]
    policy_1 = jnp.array([[0, 1],
                         [1, 1]])
    policy_2 = jnp.array([[1, 0],
                         [0, 0]])
    policy_matrix = jnp.stack([policy_1, policy_2]) # 2 x 2 x 2 tensor
    
    qs_init = [jnp.ones(2)/2, jnp.ones(2)/2]
    neg_G_all_policies = jit(update_posterior_policies)(policy_matrix, qs_init, A, B, C)
    print(neg_G_all_policies)
