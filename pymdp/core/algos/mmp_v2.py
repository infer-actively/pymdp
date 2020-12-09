#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Marginal message passing

__author__: Conor Heins, Beren Millidge, Alexander Tschantz, Brennan Klein
"""

import numpy as np

from pymdp.core.utils import to_arr_of_arr, get_model_dimensions
from pymdp.core.maths import spm_dot, spm_norm, softmax


def run_mmp_v2(
    A, B, ll_seq, policy, prev_actions=None, prior=None, num_iter=10, grad_descent=False, tau=0.25
):
    # window
    past_len = len(ll_seq)
    future_len = policy.shape[0]
    infer_len = past_len + future_len + 1
    future_cutoff = past_len + future_len - 1

    # dimensions
    _, num_states, _, num_factors = get_model_dimensions(A, B)
    A = to_arr_of_arr(A)
    B = to_arr_of_arr(B)

    # beliefs
    qs_seq = [np.empty(num_factors, dtype=object) for _ in range(infer_len)]
    for t in range(infer_len):
        for f in range(num_factors):
            qs_seq[t][f] = np.ones(num_states[f]) / num_states[f]

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

    for _ in range(num_iter):
        for t in range(infer_len):
            lnB_past_tensor = np.empty(num_factors, dtype=object)

            for f in range(num_factors):
                # likelihood
                if t < past_len:
                    lnA = np.log(spm_dot(ll_seq[t], qs_seq[t], [f]) + 1e-16)
                else:
                    lnA = np.zeros(num_states[f])

                # past message
                if t == 0:
                    lnB_past = np.log(prior[f] + 1e-16)
                else:
                    past_msg = B[f][:, :, int(policy[t - 1, f])].dot(qs_seq[t - 1][f])
                    lnB_past = np.log(past_msg + 1e-16)
                lnB_past_tensor[f] = lnB_past

                # future message
                if t >= future_cutoff:
                    lnB_future = qs_T[f]
                else:
                    future_msg = trans_B[f][:, :, int(policy[t, f])].dot(qs_seq[t + 1][f])
                    lnB_future = np.log(future_msg + 1e-16)

                # inference
                if grad_descent:
                    lnqs = np.log(qs_seq[t][f] + 1e-16)
                    coeff = 1 if (t >= future_cutoff) else 2
                    err = (coeff * lnA + lnB_past + lnB_future) - coeff * lnqs
                    err -= err.mean()
                    lnqs = lnqs + tau * err
                    qs_seq[t][f] = softmax(lnqs)
                else:
                    qs_seq[t][f] = softmax(lnA + lnB_past + lnB_future)

    return qs_seq
