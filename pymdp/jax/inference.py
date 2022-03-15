#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

import numpy as np

from jax.algos import run_vanilla_fpi, run_mmp, _run_mmp_testing

def update_posterior_states(A, obs, prior=None):

    return run_vanilla_fpi(A, obs,prior)
   
