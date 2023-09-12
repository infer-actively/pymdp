#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

from .algos import run_vanilla_fpi, run_mmp, run_vmp, run_online_filtering

def update_posterior_states(A, B, obs, prior=None, A_dependencies=None, num_iter=16, method='fpi'):

    if method == 'fpi':
        return run_vanilla_fpi(A, obs, prior, num_iter=num_iter)
    if method == 'vmp':
        return run_vmp(A, B, obs, prior, A_dependencies, num_iter=num_iter)
    if method == 'mmp':
        return run_mmp(A, B, obs, prior, A_dependencies, num_iter=num_iter)
    # if method == "ovf":
    #     prior, cond_prev = prior[0], prior[1]
    #     qs, pred, cond = run_online_filtering(A, B, obs, prior, cond_prev, A_dependencies, num_iter=num_iter)
    #     cond = [cond, cond_prev]
    #     return qs, pred, cond
   
