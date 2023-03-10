#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests for factorized version of variational fixed point iteration (FPI or "Vanilla FPI")
__author__: Conor Heins
"""

import os
import unittest

import numpy as np

from pymdp import utils, maths
from pymdp.algos import run_vanilla_fpi, run_vanilla_fpi_factorized

class TestFPI(unittest.TestCase):

    def test_factorized_fpi_one_factor_one_modality(self):
        """
        Test the sparsified version of `run_vanilla_fpi`, named `run_vanilla_fpi_factorized`
        with single hidden state factor and single observation modality.
        """

        num_states = [3]
        num_obs = [3]       

        prior = utils.random_single_categorical(num_states)

        A = utils.to_obj_array(maths.softmax(np.eye(num_states[0]) * 0.1))

        obs_idx = np.random.choice(num_obs[0])
        obs = utils.onehot(obs_idx, num_obs[0])

        mb_dict = {'A_factor_list': [[0]],
                    'A_modality_list': [[0]]}

        qs_out = run_vanilla_fpi_factorized(A, obs, num_obs, num_states, mb_dict, prior=prior)[0]
        qs_validation_1 = run_vanilla_fpi(A, obs, num_obs, num_states, prior=prior)[0]
        qs_validation_2 = maths.softmax(maths.spm_log_single(A[0][obs_idx,:]) + maths.spm_log_single(prior[0]))

        self.assertTrue(np.isclose(qs_validation_1, qs_out).all())
        self.assertTrue(np.isclose(qs_validation_2, qs_out).all())
    
    def test_factorized_fpi_one_factor_multi_modality(self):
        """
        Test the sparsified version of `run_vanilla_fpi`, named `run_vanilla_fpi_factorized`
        with single hidden state factor and multiple observation modalities.
        """

        num_states = [3]
        num_obs = [3, 2]

        prior = utils.random_single_categorical(num_states)

        A = utils.random_A_matrix(num_obs, num_states)

        obs = utils.obj_array(len(num_obs))
        for m, obs_dim in enumerate(num_obs):
            obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

        mb_dict = {'A_factor_list': [[0], [0]],
                    'A_modality_list': [[0, 1]]}
        
        qs_out = run_vanilla_fpi_factorized(A, obs, num_obs, num_states, mb_dict, prior=prior)[0]
        qs_validation = run_vanilla_fpi(A, obs, num_obs, num_states, prior=prior)[0]

        self.assertTrue(np.isclose(qs_validation, qs_out).all())
    
    def test_factorized_fpi_multi_factor_one_modality(self):
        """
        Test the sparsified version of `run_vanilla_fpi`, named `run_vanilla_fpi_factorized`
        with multiple hidden state factors and one observation modality.
        """

        num_states = [4, 5]
        num_obs = [3]

        prior = utils.random_single_categorical(num_states)

        A = utils.random_A_matrix(num_obs, num_states)

        obs_idx = np.random.choice(num_obs[0])
        obs = utils.onehot(obs_idx, num_obs[0])

        mb_dict = {'A_factor_list': [[0, 1]],
                    'A_modality_list': [[0], [0]]}
        
        qs_out = run_vanilla_fpi_factorized(A, obs, num_obs, num_states, mb_dict, prior=prior)
        qs_validation = run_vanilla_fpi(A, obs, num_obs, num_states, prior=prior)

        for qs_f_val, qs_f_out in zip(qs_validation, qs_out):
            self.assertTrue(np.isclose(qs_f_val, qs_f_out).all())


if __name__ == "__main__":
    unittest.main()