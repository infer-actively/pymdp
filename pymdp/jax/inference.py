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
    if method == "ovf":
        return run_online_filtering(A, B, obs, prior, A_dependencies, num_iter=num_iter)
   
