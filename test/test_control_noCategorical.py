#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein
"""

import os
import unittest

import numpy as np

from pymdp.core import utils, maths
from pymdp.core import control

class TestControl(unittest.TestCase):

    def test_get_expected_states(self):
        """
        Tests the refactored (Categorical-less) version of `get_expected_states`
        """

        '''Test with single hidden state factor and single timestep'''

        num_states = [3]
        num_controls = [3]

        qs = utils.obj_array_uniform(num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        qs_pi = [control.get_expected_states(qs, B, policy) for policy in policies]

        factor_idx = 0
        t_idx = 0

        for p_idx in range(len(policies)):
            self.assertTrue((qs_pi[p_idx][t_idx][factor_idx] == B[factor_idx][:,:,policies[p_idx][t_idx,factor_idx]].dot(qs[factor_idx])).all())

        '''Test with single hidden state factor and multiple-timesteps'''

        num_states = [3]
        num_controls = [3]

        qs = utils.obj_array_uniform(num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        policies = control.construct_policies(num_states, num_controls, policy_len=2)

        qs_pi = [control.get_expected_states(qs, B, policy) for policy in policies]

        for p_idx in range(len(policies)):
            for t_idx in range(2):
                if t_idx == 0:
                    self.assertTrue((qs_pi[p_idx][t_idx][factor_idx] == B[factor_idx][:,:,policies[p_idx][t_idx,factor_idx]].dot(qs[factor_idx])).all())
                else:
                    self.assertTrue((qs_pi[p_idx][t_idx][factor_idx] == B[factor_idx][:,:,policies[p_idx][t_idx,factor_idx]].dot(qs_pi[p_idx][t_idx-1][factor_idx])).all())
       
        '''Test with multiple hidden state factors and single timestep'''

        num_states = [3, 3]
        num_controls = [3, 1]

        num_factors = len(num_states)

        qs = utils.obj_array_uniform(num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        qs_pi = [control.get_expected_states(qs, B, policy) for policy in policies]

        t_idx = 0

        for p_idx in range(len(policies)):
            for factor_idx in range(num_factors):
                self.assertTrue((qs_pi[p_idx][t_idx][factor_idx] == B[factor_idx][:,:,policies[p_idx][t_idx,factor_idx]].dot(qs[factor_idx])).all())

        '''Test with multiple hidden state factors and multiple timesteps'''

        num_states = [3, 3]
        num_controls = [3, 3]

        num_factors = len(num_states)

        qs = utils.obj_array_uniform(num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        policies = control.construct_policies(num_states, num_controls, policy_len=3)

        qs_pi = [control.get_expected_states(qs, B, policy) for policy in policies]

        for p_idx in range(len(policies)):
            for t_idx in range(3):
                for factor_idx in range(num_factors):
                    if t_idx == 0:
                        self.assertTrue((qs_pi[p_idx][t_idx][factor_idx] == B[factor_idx][:,:,policies[p_idx][t_idx,factor_idx]].dot(qs[factor_idx])).all())
                    else:
                        self.assertTrue((qs_pi[p_idx][t_idx][factor_idx] == B[factor_idx][:,:,policies[p_idx][t_idx,factor_idx]].dot(qs_pi[p_idx][t_idx-1][factor_idx])).all())

    def test_get_expected_states_and_obs(self):
        """
        Tests the refactored (Categorical-less) versions of `get_expected_states` and `get_expected_obs` together
        """

        '''Test with single observation modality, single hidden state factor and single timestep'''

        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        qs = utils.obj_array_uniform(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)
        
        factor_idx = 0
        modality_idx = 0
        t_idx = 0

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)

            # validation qs_pi
            qs_pi_valid = B[factor_idx][:,:,policies[idx][t_idx,factor_idx]].dot(qs[factor_idx])

            self.assertTrue((qs_pi[t_idx][factor_idx] == qs_pi_valid).all())

            qo_pi = control.get_expected_obs(qs_pi, A)

            # validation qo_pi
            qo_pi_valid = maths.spm_dot(A[modality_idx],utils.to_arr_of_arr(qs_pi_valid))

            self.assertTrue((qo_pi[t_idx][modality_idx] == qo_pi_valid).all())

if __name__ == "__main__":
    unittest.main()
