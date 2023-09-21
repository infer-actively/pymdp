#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

from .algos import run_factorized_fpi, run_mmp, run_vmp

def update_posterior_states(A, B, obs, prior=None, qs_hist=None, A_dependencies=None, num_iter=16, method='fpi'):

    if method == 'fpi' or method == "ovf":
        qs = run_factorized_fpi(A, obs, prior, A_dependencies, num_iter=num_iter)
        return qs_hist.append(qs)
    if method == 'vmp':
        return run_vmp(A, B, obs, prior, A_dependencies, num_iter=num_iter)
    if method == 'mmp':
        return run_mmp(A, B, obs, prior, A_dependencies, num_iter=num_iter)
   
