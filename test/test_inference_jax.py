#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Dimitrije Markovic, Conor Heins
"""

import os
import unittest

import numpy as np
import jax.numpy as jnp

from pymdp.jax.algos import run_vanilla_fpi as fpi_jax
from pymdp.algos import run_vanilla_fpi as fpi_numpy
from pymdp import utils, maths

class TestInferenceJax(unittest.TestCase):

    def test_fixed_point_iteration(self):
        """
        Tests the jax-ified version of mean-field fixed-point iteration against the original numpy version
        """

        ''' Create a random generative model with a desired number/dimensionality of hidden state factors and observation modalities'''

        # fpi_jax throws an error (some broadcasting dimension mistmatch in `fpi_jax`)
        num_states = [2, 2, 5]
        num_obs = [5, 10]

        # fpi_jax executes and returns an answer, but it is numerically incorrect
        # num_states = [2, 2, 2]
        # num_obs = [5, 10]

        # this works and returns the right answer
        # num_states = [4, 4]
        # num_obs = [5, 10, 6]

        # numpy version
        prior = utils.random_single_categorical(num_states)
        A = utils.random_A_matrix(num_obs, num_states)

        obs_idx = [utils.sample(maths.spm_dot(a_m, prior)) for a_m in A]
        obs = utils.process_observation(obs_idx, len(num_obs), num_obs)

        qs_numpy = fpi_numpy(A, obs, num_obs, num_states, prior=prior, num_iter=16, dF=1.0, dF_tol=-1.0) # set dF_tol to negative number so FPI never stops due to convergence

        # jax version
        prior = [jnp.array(prior_f) for prior_f in prior]
        A = [jnp.array(a_m) for a_m in A]
        obs = [jnp.array(o_m) for o_m in obs]

        qs_jax = fpi_jax(A, obs, prior, num_iter=16)

        for f, _ in enumerate(qs_jax):
            self.assertTrue(np.allclose(qs_numpy[f], qs_jax[f]))

if __name__ == "__main__":
    unittest.main()