#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein
"""

import os
import unittest

import numpy as np

from pymdp import utils, maths
from pymdp import inference

class TestInference(unittest.TestCase):

    def test_update_posterior_states(self):
        """
        Tests the refactored version of `update_posterior_states`
        """

        '''Test with single hidden state factor and single observation modality'''

        num_states = [3]
        num_obs = [3]       

        prior = utils.random_single_categorical(num_states)

        A = utils.to_obj_array(maths.softmax(np.eye(num_states[0]) * 0.1))

        obs_idx = 1
        obs = utils.onehot(obs_idx, num_obs[0])

        qs_out = inference.update_posterior_states(A, obs, prior=prior)
        qs_validation =  maths.softmax(maths.spm_log_single(A[0][obs_idx,:]) + maths.spm_log_single(prior[0]))

        self.assertTrue(np.isclose(qs_validation, qs_out[0]).all())

        '''Try single modality inference where the observation is passed in as an int'''
        qs_out_2 = inference.update_posterior_states(A, obs_idx, prior=prior)
        self.assertTrue(np.isclose(qs_out_2[0], qs_out[0]).all())

        '''Try single modality inference where the observation is a one-hot stored in an object array'''
        qs_out_3 = inference.update_posterior_states(A, utils.to_obj_array(obs), prior=prior)
        self.assertTrue(np.isclose(qs_out_3[0], qs_out[0]).all())

        '''Test with multiple hidden state factors and single observation modality'''

        num_states = [3, 4]
        num_obs = [3]

        prior = utils.random_single_categorical(num_states)

        A = utils.random_A_matrix(num_obs, num_states)

        obs_idx = 1
        obs = utils.onehot(obs_idx, num_obs[0])

        qs_out = inference.update_posterior_states(A, obs, prior=prior, num_iter = 1)

        # validate with a quick n' dirty implementation of FPI

        # initialize posterior and log prior
        qs_valid_init = utils.obj_array_uniform(num_states)
        log_prior = maths.spm_log_obj_array(prior)

        qs_valid_final = utils.obj_array(len(num_states))

        log_likelihood = maths.spm_log_single(maths.get_joint_likelihood(A, obs, num_states))

        num_factors = len(num_states)

        qs_valid_init_all = qs_valid_init[0]
        for factor in range(num_factors-1):
            qs_valid_init_all = qs_valid_init_all[...,None]*qs_valid_init[factor+1]
        LL_tensor = log_likelihood * qs_valid_init_all

        factor_ids = range(num_factors)

        for factor, qs_f in enumerate(qs_valid_init):
            ax2sum = tuple(set(factor_ids) - set([factor])) # which axes to sum out
            qL = LL_tensor.sum(axis = ax2sum) / qs_f
            qs_valid_final[factor] = maths.softmax(qL + log_prior[factor])

        for factor, qs_f_valid in enumerate(qs_valid_final):
            self.assertTrue(np.isclose(qs_f_valid, qs_out[factor]).all())

        '''Test with multiple hidden state factors and multiple observation modalities, for two different kinds of observation input formats'''
        
        num_states = [3, 4]
        num_obs = [3, 3, 5]

        prior = utils.random_single_categorical(num_states)

        A = utils.random_A_matrix(num_obs, num_states)

        obs_index_tuple = tuple([np.random.randint(obs_dim) for obs_dim in num_obs])

        qs_out1 = inference.update_posterior_states(A, obs_index_tuple, prior=prior)

        obs_onehots = utils.obj_array(len(num_obs))
        for g in range(len(num_obs)):
            obs_onehots[g] = utils.onehot(obs_index_tuple[g], num_obs[g])

        qs_out2 = inference.update_posterior_states(A, obs_onehots, prior=prior)

        for factor in range(len(num_states)):
            self.assertTrue(np.isclose(qs_out1[factor], qs_out2[factor]).all())

    def test_update_posterior_states_factorized_single_factor(self):
        """
        Tests the version of `update_posterior_states` where an `mb_dict` is provided as an argument to factorize
        the fixed-point iteration (FPI) algorithm. Single factor version.
        """
        num_states = [3]
        num_obs = [3]       

        prior = utils.random_single_categorical(num_states)

        A = utils.to_obj_array(maths.softmax(np.eye(num_states[0]) * 0.1))

        obs_idx = 1
        obs = utils.onehot(obs_idx, num_obs[0])

        mb_dict = {'A_factor_list': [[0]],
                    'A_modality_list': [[0]]}

        qs_out = inference.update_posterior_states_factorized(A, obs, num_obs, num_states, mb_dict, prior=prior)
        qs_validation =  maths.softmax(maths.spm_log_single(A[0][obs_idx,:]) + maths.spm_log_single(prior[0]))

        self.assertTrue(np.isclose(qs_validation, qs_out[0]).all())

        '''Try single modality inference where the observation is passed in as an int'''
        qs_out_2 = inference.update_posterior_states_factorized(A, obs_idx, num_obs, num_states, mb_dict, prior=prior)
        self.assertTrue(np.isclose(qs_out_2[0], qs_out[0]).all())

        '''Try single modality inference where the observation is a one-hot stored in an object array'''
        qs_out_3 = inference.update_posterior_states_factorized(A, utils.to_obj_array(obs),num_obs, num_states, mb_dict, prior=prior)
        self.assertTrue(np.isclose(qs_out_3[0], qs_out[0]).all())

    def test_update_posterior_states_factorized(self):
        """
        Tests the version of `update_posterior_states` where an `mb_dict` is provided as an argument to factorize
        the fixed-point iteration (FPI) algorithm.
        """

        num_states = [3, 4]
        num_obs = [3, 3, 5]

        prior = utils.random_single_categorical(num_states)

        obs_index_tuple = tuple([np.random.randint(obs_dim) for obs_dim in num_obs])

        mb_dict = {'A_factor_list': [[0], [1], [0, 1]],
                    'A_modality_list': [[0, 2], [1, 2]]}
        
        A_reduced = utils.random_A_matrix(num_obs, num_states, A_factor_list=mb_dict['A_factor_list'])

        qs_out = inference.update_posterior_states_factorized(A_reduced, obs_index_tuple, num_obs, num_states, mb_dict, prior=prior)

        A_full = utils.initialize_empty_A(num_obs, num_states)
        for m, A_m in enumerate(A_full):
            other_factors = list(set(range(len(num_states))) - set(mb_dict['A_factor_list'][m])) # list of the factors that modality `m` does not depend on

            # broadcast or tile the reduced A matrix (`A_reduced`) along the dimensions of corresponding to `other_factors`
            expanded_dims = [num_obs[m]] + [1 if f in other_factors else ns for (f, ns) in enumerate(num_states)]
            tile_dims = [1] + [ns if f in other_factors else 1 for (f, ns) in enumerate(num_states)]
            A_full[m] = np.tile(A_reduced[m].reshape(expanded_dims), tile_dims)
        
        qs_validation = inference.update_posterior_states(A_full, obs_index_tuple, prior=prior)

        for qs_f_val, qs_f_out in zip(qs_validation, qs_out):
            self.assertTrue(np.isclose(qs_f_val, qs_f_out).all())
    
    def test_update_posterior_states_factorized_noVFE_compute(self):
        """
        Tests the version of `update_posterior_states` where an `mb_dict` is provided as an argument to factorize
        the fixed-point iteration (FPI) algorithm.

        In this version, we always run the total number of iterations because we don't compute the variational free energy over the course of convergence/optimization.
        """

        num_states = [3, 4]
        num_obs = [3, 3, 5]

        prior = utils.random_single_categorical(num_states)

        obs_index_tuple = tuple([np.random.randint(obs_dim) for obs_dim in num_obs])

        mb_dict = {'A_factor_list': [[0], [1], [0, 1]],
                    'A_modality_list': [[0, 2], [1, 2]]}
        
        A_reduced = utils.random_A_matrix(num_obs, num_states, A_factor_list=mb_dict['A_factor_list'])

        qs_out = inference.update_posterior_states_factorized(A_reduced, obs_index_tuple, num_obs, num_states, mb_dict, prior=prior, compute_vfe=False)

        A_full = utils.initialize_empty_A(num_obs, num_states)
        for m, A_m in enumerate(A_full):
            other_factors = list(set(range(len(num_states))) - set(mb_dict['A_factor_list'][m])) # list of the factors that modality `m` does not depend on

            # broadcast or tile the reduced A matrix (`A_reduced`) along the dimensions of corresponding to `other_factors`
            expanded_dims = [num_obs[m]] + [1 if f in other_factors else ns for (f, ns) in enumerate(num_states)]
            tile_dims = [1] + [ns if f in other_factors else 1 for (f, ns) in enumerate(num_states)]
            A_full[m] = np.tile(A_reduced[m].reshape(expanded_dims), tile_dims)
        
        qs_validation = inference.update_posterior_states(A_full, obs_index_tuple, prior=prior, compute_vfe=False)

        for qs_f_val, qs_f_out in zip(qs_validation, qs_out):
            self.assertTrue(np.isclose(qs_f_val, qs_f_out).all())
    

if __name__ == "__main__":
    unittest.main()