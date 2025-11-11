#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Dimitrije Markovic, Conor Heins
"""

import unittest

import numpy as np
from jax import numpy as jnp, random as jr

from pymdp.algos import run_vanilla_fpi as fpi_jax
from pymdp.utils import random_factorized_categorical, random_A_array

from pymdp.legacy.algos import run_vanilla_fpi as fpi_numpy
from pymdp.legacy import utils

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

        keys = jr.split(jr.PRNGKey(42), len(num_states_list)*2).reshape((len(num_states_list), 2, 2))
        for (keys_per_element, num_states, num_obs) in zip(keys, num_states_list, num_obs_list):
            
            # jax arrays
            prior_jax = random_factorized_categorical(keys_per_element[0], num_states)
            A_jax = random_A_array(keys_per_element[1], num_obs, num_states)

            # numpy arrays
            prior = utils.obj_array_from_list(prior_jax)
            A_np = utils.obj_array_from_list(A_jax)

            obs = utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

            qs_numpy = fpi_numpy(A_np, obs, num_obs, num_states, prior=prior, num_iter=16, dF=1.0, dF_tol=-1.0) # set dF_tol to negative number so numpy version of FPI never stops early due to convergence

            obs = [jnp.array(o_m) for o_m in obs]
            qs_jax = fpi_jax(A_jax, obs, prior_jax, num_iter=16)

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

        keys = jr.split(jr.PRNGKey(43), len(num_states_list)*2).reshape((len(num_states_list), 2, 2))
        for (keys_per_element, num_states, num_obs) in zip(keys, num_states_list, num_obs_list):

            # jax arrays
            prior_jax = random_factorized_categorical(keys_per_element[0], num_states)
            A_jax = random_A_array(keys_per_element[1], num_obs, num_states)

            # numpy arrays
            prior = utils.obj_array_from_list(prior_jax)
            A_np = utils.obj_array_from_list(A_jax)

            obs = utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

            qs_numpy = fpi_numpy(A_np, obs, num_obs, num_states, prior=prior, num_iter=16, dF=1.0, dF_tol=-1.0) # set dF_tol to negative number so numpy version of FPI never stops early due to convergence

            obs = [jnp.array(o_m) for o_m in obs]

            qs_jax = fpi_jax(A_jax, obs, prior_jax, num_iter=16)

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

        keys = jr.split(jr.PRNGKey(44), len(num_states_list)*2).reshape((len(num_states_list), 2, 2))
        for (keys_per_element, num_states, num_obs) in zip(keys, num_states_list, num_obs_list):
            
            # jax arrays
            prior_jax = random_factorized_categorical(keys_per_element[0], num_states)
            A_jax = random_A_array(keys_per_element[1], num_obs, num_states)

            # numpy version
            prior = utils.obj_array_from_list(prior_jax)
            A_np = utils.obj_array_from_list(A_jax)

            obs = utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

            qs_numpy = fpi_numpy(A_np, obs, num_obs, num_states, prior=prior, num_iter=16, dF=1.0, dF_tol=-1.0) # set dF_tol to negative number so numpy version of FPI never stops early due to convergence

            obs = [jnp.array(o_m) for o_m in obs]
            qs_jax = fpi_jax(A_jax, obs, prior_jax, num_iter=16)

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

        keys = jr.split(jr.PRNGKey(45), len(num_states_list)*2).reshape((len(num_states_list), 2, 2))
        for (keys_per_element, num_states, num_obs) in zip(keys, num_states_list, num_obs_list):
            
            # jax arrays
            prior_jax = random_factorized_categorical(keys_per_element[0], num_states)
            A_jax = random_A_array(keys_per_element[1], num_obs, num_states)
                
            # numpy arrays
            prior = utils.obj_array_from_list(prior_jax)
            A_np = utils.obj_array_from_list(A_jax)

            obs = utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

            qs_numpy = fpi_numpy(A_np, obs, num_obs, num_states, prior=prior, num_iter=16, dF=1.0, dF_tol=-1.0) # set dF_tol to negative number so numpy version of FPI never stops early due to convergence

            # jax version
            obs = [jnp.array(o_m) for o_m in obs]
            qs_jax = fpi_jax(A_jax, obs, prior_jax, num_iter=16)

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

        keys = jr.split(jr.PRNGKey(46), len(num_states_list)*2).reshape((len(num_states_list), 2, 2))
        for (keys_per_element, num_states, num_obs) in zip(keys, num_states_list, num_obs_list):
            
            # jax arrays
            A_jax = random_A_array(keys_per_element[1], num_obs, num_states)
            prior_jax = random_factorized_categorical(keys_per_element[0], num_states)

            # numpy arrays
            prior = utils.obj_array_from_list(prior_jax)
            A_np = utils.obj_array_from_list(A_jax)

            obs = utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

            qs_numpy = fpi_numpy(A_np, obs, num_obs, num_states, prior=prior, num_iter=16, dF=1.0, dF_tol=-1.0) # set dF_tol to negative number so numpy version of FPI never stops early due to convergence

            obs_idx = []
            for ob in obs:
                obs_idx.append(np.where(ob)[0][0])
            
            qs_jax = fpi_jax(A_jax, obs_idx, prior_jax, num_iter=16, distr_obs=False)

            for f, _ in enumerate(qs_jax):
                self.assertTrue(np.allclose(qs_numpy[f], qs_jax[f]))

if __name__ == "__main__":
    unittest.main()
