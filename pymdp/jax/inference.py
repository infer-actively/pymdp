#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

from .algos import run_vanilla_fpi

def update_posterior_states(A, obs, prior=None):

    return run_vanilla_fpi(A, obs, prior)
   
