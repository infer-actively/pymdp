#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-an-iterable

import itertools
import jax.numpy as jnp
from functools import partial
from jax import lax, vmap, nn
from maths import *
from itertools import chain
# import pymdp.jax.utils as utils

def update_posterior_policies(policy_matrix, qs_init, A, B, log_C, gamma = 16.0):
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    ## vmap across policies
    compute_G_fixed_states = partial(compute_G_policy, qs_init, A, B, log_C)

    # only in the case of policy-dependent qs_inits
    # in_axes_list = (1,) * n_factors
    # all_efe_of_policies = vmap(compute_G_policy, in_axes=(in_axes_list, 0))(qs_init_pi, policy_matrix)

    neg_efe_all_policies = vmap(compute_G_fixed_states)(policy_matrix)
    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)

    # @TODO: convert negative EFE of each policy into a posterior probability

    return nn.softmax(gamma * neg_efe_all_policies), neg_efe_all_policies

def compute_expected_state(qs_prior, B, u_t): 
    """
    Compute posterior over next state, given belief about previous state, transition model and action...
    """  
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

def factor_dot_2(A, qs):
    """ Dot product of a multidimensional array with `x`.
    
    Parameters
    ----------
    - `x` [1D numpy.ndarray] - either vector or array of arrays
        The alternative array to perform the dot product with
    
    Returns 
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """

    x = qs[0]
    for q in qs[1:]:
        x = jnp.expand_dims(x, -1) * q

    joint = A * x
    dim = joint.shape[0]
    return joint.reshape(dim, -1).sum(-1)

def compute_expected_obs(qs, A):

    qo = []
    for A_m in A:
        qo.append( factor_dot(A_m, qs) )
        # qo.append( factor_dot_2(A_m, qs) )

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
    
def compute_expected_utility(qo, log_C):
    
    util = 0.
    for o_m, log_C_m in zip(qo, log_C):
        util += (o_m * log_C_m).sum()
    
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
    log_C = [log_stable(jnp.array([0.8, 0.1, 0.1])), log_stable(jnp.ones(4)/4)]
    policy_1 = jnp.array([[0, 1],
                         [1, 1]])
    policy_2 = jnp.array([[1, 0],
                         [0, 0]])
    policy_matrix = jnp.stack([policy_1, policy_2]) # 2 x 2 x 2 tensor
    
    qs_init = [jnp.ones(2)/2, jnp.ones(2)/2]
    neg_G_all_policies = jit(update_posterior_policies)(policy_matrix, qs_init, A, B, log_C)
    print(neg_G_all_policies)
