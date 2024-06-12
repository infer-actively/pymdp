#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Dimitrije Markovic, Conor Heins
"""

import os
import sys
import unittest

import numpy as np
import jax.numpy as jnp

# import the library directly from local source (rather than relying on the library being installed)
# insert the dependency so it's prioritized over an installed variant
sys.path.insert(0, os.path.abspath('../pymdp'))

from pymdp.jax.algos import run_vanilla_fpi as fpi_jax
from pymdp.algos import run_vanilla_fpi as fpi_numpy
from pymdp import utils

class TestInferenceJax(unittest.TestCase):

    def test_fixed_point_iteration_singlestate_singleobs(self):
        """
        Tests the jax-ified version of mean-field fixed-point iteration against the original numpy version.
        In this version there is one hidden state factor and one observation modality
        """

        num_states_list = [
                            [1],
                            [5],
                            [10]
        ]

        num_obs_list = [
                        [5],
                        [1],
                        [2]
        ]

        for (num_states, num_obs) in zip(num_states_list, num_obs_list):

            # numpy version
            prior = utils.random_single_categorical(num_states)
            A = utils.random_A_matrix(num_obs, num_states)

            obs = utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

            qs_numpy = fpi_numpy(A, obs, num_obs, num_states, prior=prior, num_iter=16, dF=1.0, dF_tol=-1.0) # set dF_tol to negative number so numpy version of FPI never stops early due to convergence

            # jax version
            prior = [jnp.array(prior_f) for prior_f in prior]
            A = [jnp.array(a_m) for a_m in A]
            obs = [jnp.array(o_m) for o_m in obs]

            qs_jax = fpi_jax(A, obs, prior, num_iter=16)

            for f, _ in enumerate(qs_jax):
                self.assertTrue(np.allclose(qs_numpy[f], qs_jax[f]))
    
    def test_fixed_point_iteration_singlestate_multiobs(self):
        """
        Tests the jax-ified version of mean-field fixed-point iteration against the original numpy version.
        In this version there is one hidden state factor and multiple observation modalities
        """

        num_states_list = [
                            [1],
                            [5],
                            [10]
        ]

        num_obs_list = [
                        [5, 2],
                        [1, 8, 9],
                        [2, 2, 2]
        ]

        for (num_states, num_obs) in zip(num_states_list, num_obs_list):

            # numpy version
            prior = utils.random_single_categorical(num_states)
            A = utils.random_A_matrix(num_obs, num_states)

            obs = utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

            qs_numpy = fpi_numpy(A, obs, num_obs, num_states, prior=prior, num_iter=16, dF=1.0, dF_tol=-1.0) # set dF_tol to negative number so numpy version of FPI never stops early due to convergence

            # jax version
            prior = [jnp.array(prior_f) for prior_f in prior]
            A = [jnp.array(a_m) for a_m in A]
            obs = [jnp.array(o_m) for o_m in obs]

            qs_jax = fpi_jax(A, obs, prior, num_iter=16)

            for f, _ in enumerate(qs_jax):
                self.assertTrue(np.allclose(qs_numpy[f], qs_jax[f]))
    
    def test_fixed_point_iteration_multistate_singleobs(self):
        """
        Tests the jax-ified version of mean-field fixed-point iteration against the original numpy version.
        In this version there are multiple hidden state factors and a single observation modality
        """

        num_states_list = [
                            [1, 10, 2],
                            [5, 5, 10, 2],
                            [10, 2]
        ]

        num_obs_list = [
                        [5],
                        [1],
                        [10]
        ]

        for (num_states, num_obs) in zip(num_states_list, num_obs_list):

            # numpy version
            prior = utils.random_single_categorical(num_states)
            A = utils.random_A_matrix(num_obs, num_states)

            obs = utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

            qs_numpy = fpi_numpy(A, obs, num_obs, num_states, prior=prior, num_iter=16, dF=1.0, dF_tol=-1.0) # set dF_tol to negative number so numpy version of FPI never stops early due to convergence

            # jax version
            prior = [jnp.array(prior_f) for prior_f in prior]
            A = [jnp.array(a_m) for a_m in A]
            obs = [jnp.array(o_m) for o_m in obs]

            qs_jax = fpi_jax(A, obs, prior, num_iter=16)

            for f, _ in enumerate(qs_jax):
                self.assertTrue(np.allclose(qs_numpy[f], qs_jax[f]))


    def test_fixed_point_iteration_multistate_multiobs(self):
        """
        Tests the jax-ified version of mean-field fixed-point iteration against the original numpy version.
        In this version there are multiple hidden state factors and multiple observation modalities
        """

        ''' Start by creating a collection of random generative models with different 
        cardinalities and dimensionalities of hidden state factors and observation modalities'''

        num_states_list = [ 
                         [2, 2, 5],
                         [2, 2, 2],
                         [4, 4]
        ]

        num_obs_list = [
                        [5, 10],
                        [4, 3, 2],
                        [5, 10, 6]
        ]

        for (num_states, num_obs) in zip(num_states_list, num_obs_list):

            # numpy version
            prior = utils.random_single_categorical(num_states)
            A = utils.random_A_matrix(num_obs, num_states)

            obs = utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

            qs_numpy = fpi_numpy(A, obs, num_obs, num_states, prior=prior, num_iter=16, dF=1.0, dF_tol=-1.0) # set dF_tol to negative number so numpy version of FPI never stops early due to convergence

            # jax version
            prior = [jnp.array(prior_f) for prior_f in prior]
            A = [jnp.array(a_m) for a_m in A]
            obs = [jnp.array(o_m) for o_m in obs]

            qs_jax = fpi_jax(A, obs, prior, num_iter=16)

            for f, _ in enumerate(qs_jax):
                self.assertTrue(np.allclose(qs_numpy[f], qs_jax[f]))
    
    def test_fixed_point_iteration_index_observations(self):
        """
        Tests the jax-ified version of mean-field fixed-point iteration against the original NumPy version.
        In this version there are multiple hidden state factors and multiple observation modalities.

        Test the jax version with index-based observations (not one-hots)
        """

        ''' Start by creating a collection of random generative models with different 
        cardinalities and dimensionalities of hidden state factors and observation modalities'''

        num_states_list = [ 
                         [2, 2, 5],
                         [2, 2, 2],
                         [4, 4]
        ]

        num_obs_list = [
                        [5, 10],
                        [4, 3, 2],
                        [5, 10, 6]
        ]

        for (num_states, num_obs) in zip(num_states_list, num_obs_list):

            # numpy version
            prior = utils.random_single_categorical(num_states)
            A = utils.random_A_matrix(num_obs, num_states)

            obs = utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

            qs_numpy = fpi_numpy(A, obs, num_obs, num_states, prior=prior, num_iter=16, dF=1.0, dF_tol=-1.0) # set dF_tol to negative number so numpy version of FPI never stops early due to convergence

            obs_idx = []
            for ob in obs:
                obs_idx.append(np.where(ob)[0][0])
            
            # jax version
            prior = [jnp.array(prior_f) for prior_f in prior]
            A = [jnp.array(a_m) for a_m in A]
            # obs = [jnp.array(o_m) for o_m in obs]

            qs_jax = fpi_jax(A, obs_idx, prior, num_iter=16, distr_obs=False)

            for f, _ in enumerate(qs_jax):
                self.assertTrue(np.allclose(qs_numpy[f], qs_jax[f]))

if __name__ == "__main__":
    unittest.main()