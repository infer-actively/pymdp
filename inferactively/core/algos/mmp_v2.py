#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Marginal message passing

__author__: Conor Heins, Beren Millidge, Alexander Tschantz, Brennan Klein
"""

import numpy as np

from inferactively.core import is_arr_of_arr, to_arr_of_arr, spm_dot, spm_norm, softmax


def get_model_dimensions(A, B):
    num_obs = [a.shape[0] for a in A] if is_arr_of_arr(A) else [A.shape[0]]
    num_states = [b.shape[0] for b in B] if is_arr_of_arr(B) else [B.shape[0]]
    num_modalities = len(num_obs)
    num_factors = len(num_states)
    return num_obs, num_states, num_modalities, num_factors


def get_likelihood_seq(A, obs, num_states):
    ll_seq = np.empty(len(obs), dtype=object)
    for t, obs in enumerate(obs):
        ll_t = np.ones(tuple(num_states))
        for modality in range(len(A)):
            ll_t = ll_t * spm_dot(A[modality], obs[obs][modality], obs_mode=True)
        ll_seq[t] = ll_t
    return ll_seq


def run_mmp_v2(
    A, B, prev_obs, policy, prior=None, prev_actions=None, num_iter=10, grad_descent=False, tau=0.25
):
    # windpw
    past_len = len(prev_obs)
    future_len = policy.shape[0]
    infer_len = past_len + future_len + 1
    future_cutoff = past_len + future_len - 1

    # dimensions
    _, num_states, _, num_factors = get_model_dimensions(A, B)
    A = to_arr_of_arr(A)
    B = to_arr_of_arr(B)

    # beliefs
    qs = [np.empty(num_factors, dtype=object) for i in range(infer_len)]
    for t in range(infer_len):
        for f in range(num_factors):
            qs[t][f] = np.ones(num_states[f]) / num_states[f]

    # last message
    qs_T = np.empty(num_factors, dtype=object)
    for f in range(num_factors):
        qs_T[f] = np.zeros(num_states[f])

    # prior
    if prior is None:
        prior = np.empty(num_factors, dtype=object)
        for f in range(num_factors):
            prior[f] = np.ones(num_states[f]) / num_states[f]

    # transposed transition
    trans_B = np.empty(num_factors, dtype=object)
    for f in range(num_factors):
        trans_B[f] = np.zeros_like(B[f])
        for u in range(B[f].shape[2]):
            trans_B[f][:, :, u] = spm_norm(B[f][:, :, u].T)

    # full policy
    if prev_actions is None:
        prev_actions = np.zeros((past_len, policy.shape[1]))
    policy = np.vstack((prev_actions, policy))

    # likelihood sequence
    ll_seq = get_likelihood_seq(A, prev_obs, num_states)
    for n in range(num_iter):
        for t in range(infer_len):
            lnB_past_tensor = np.empty(num_factors, dtype=object)
            for f in range(num_factors):

                # likelihood
                if t < past_len:
                    lnA = np.log(spm_dot(ll_seq[t], qs[t], [f]) + 1e-16)
                else:
                    lnA = np.zeros(num_states[f])

                # past message
                if t == 0:
                    lnB_past = np.log(prior[f] + 1e-16)
                else:
                    past_msg = B[f][:, :, policy[t - 1, f]].dot(qs[t - 1][f])
                    lnB_past = np.log(past_msg + 1e-16)
                lnB_past_tensor[f] = lnB_past

                # future message
                if t >= future_cutoff:
                    lnB_future = qs_T[f]
                else:
                    future_msg = trans_B[f][:, :, int(policy[t, f])].dot(qs[t + 1][f])
                    lnB_future = np.log(future_msg + 1e-16)

                # inference
                if grad_descent:
                    lnqs = np.log(qs[t][f] + 1e-16)
                    coeff = 1 if (t >= future_cutoff) else 2
                    err = (coeff * lnA + lnB_past + lnB_future) - coeff * lnqs
                    err -= err.mean()
                    lns = lnqs + tau * err
                    qs[t][f] = softmax(lns)
                else:
                    qs[t][f] = softmax(lnA + lnB_past + lnB_future)

    return qs, None, None, None
