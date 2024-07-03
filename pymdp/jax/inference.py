#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

import jax.numpy as jnp
from .algos import run_factorized_fpi, run_mmp, run_vmp
from jax import tree_util as jtu, lax
from jax.experimental.sparse._base import JAXSparse
from jax.experimental import sparse
from jaxtyping import Array, ArrayLike

eps = jnp.finfo('float').eps

def update_posterior_states(
        A, 
        B, 
        obs, 
        past_actions, 
        prior=None, 
        qs_hist=None, 
        A_dependencies=None, 
        B_dependencies=None, 
        num_iter=16, 
        method='fpi'
    ):

    if method == 'fpi' or method == "ovf":
        # format obs to select only last observation
        curr_obs = jtu.tree_map(lambda x: x[-1], obs)
        qs = run_factorized_fpi(A, curr_obs, prior, A_dependencies, num_iter=num_iter)
    else:
        # format B matrices using action sequences here
        # TODO: past_actions can be None
        if past_actions is not None:
            nf = len(B)
            actions_tree = [past_actions[:, i] for i in range(nf)]
            
            # move time steps to the leading axis (leftmost)
            # this assumes that a policy is always specified as the rightmost axis of Bs
            B = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree)
        else:
            B = None

        # outputs of both VMP and MMP should be a list of hidden state factors, where each qs[f].shape = (T, batch_dim, num_states_f)
        if method == 'vmp':
            qs = run_vmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter) 
        if method == 'mmp':
            qs = run_mmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter)
    
    if qs_hist is not None:
        if method == 'fpi' or method == "ovf":
            qs_hist = jtu.tree_map(lambda x, y: jnp.concatenate([x, jnp.expand_dims(y, 0)], 0), qs_hist, qs)
        else:
            #TODO: return entire history of beliefs
            qs_hist = qs
    else:
        if method == 'fpi' or method == "ovf":
            qs_hist = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), qs)
        else:
            qs_hist = qs
    
    return qs_hist

def joint_dist_factor(b: ArrayLike, filtered_qs: list[Array], actions: Array):
    qs_last = filtered_qs[-1]
    qs_filter = filtered_qs[:-1]

    def step_fn(qs_smooth, xs):
        qs_f, action = xs
        time_b = b[..., action]
        qs_j = time_b * qs_f
        norm = qs_j.sum(-1, keepdims=True)
        if isinstance(norm, JAXSparse):
            norm = sparse.todense(norm)
        norm = jnp.where(norm == 0, eps, norm)
        qs_backward_cond = qs_j / norm
        qs_joint = qs_backward_cond * jnp.expand_dims(qs_smooth, -1)
        qs_smooth = qs_joint.sum(-2)
        if isinstance(qs_smooth, JAXSparse):
            qs_smooth = sparse.todense(qs_smooth)
        
        # returns q(s_t), (q(s_t), q(s_t, s_t+1))
        return qs_smooth, (qs_smooth, qs_joint)

    # seq_qs will contain a sequence of smoothed marginals and joints
    _, seq_qs = lax.scan(
        step_fn,
        qs_last,
        (qs_filter, actions),
        reverse=True,
        unroll=2
    )

    # we add the last filtered belief to smoothed beliefs

    qs_smooth_all = jnp.concatenate([seq_qs[0], jnp.expand_dims(qs_last, 0)], 0)
    qs_joint_all = seq_qs[1]
    if isinstance(qs_joint_all, JAXSparse):
        qs_joint_all.shape = (len(actions),) + qs_joint_all.shape
    return qs_smooth_all, qs_joint_all


def smoothing_ovf(filtered_post, B, past_actions):
    assert len(filtered_post) == len(B)
    nf = len(B)  # number of factors

    joint = lambda b, qs, f: joint_dist_factor(b, qs, past_actions[..., f])

    marginals_and_joints = ([], [])
    for b, qs, f in zip(B, filtered_post, list(range(nf))):
        marginals, joints = joint(b, qs, f)
        marginals_and_joints[0].append(marginals)
        marginals_and_joints[1].append(joints)

    return marginals_and_joints


    