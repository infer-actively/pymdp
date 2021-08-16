#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein
"""

import os
import unittest

import numpy as np

from pymdp import utils, maths
from pymdp import control

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

        '''Test with multiple observation modalities, multiple hidden state factors and single timestep'''

        num_obs = [3, 3]
        num_states = [3, 3]
        num_controls = [3, 2]

        num_factors = len(num_states)
        num_modalities = len(num_obs)

        qs = utils.obj_array_uniform(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)
        
        t_idx = 0

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)

            for factor_idx in range(num_factors):
                # validation qs_pi
                qs_pi_valid = B[factor_idx][:,:,policies[idx][t_idx,factor_idx]].dot(qs[factor_idx])
                self.assertTrue((qs_pi[t_idx][factor_idx] == qs_pi_valid).all())

            qo_pi = control.get_expected_obs(qs_pi, A)

            for modality_idx in range(num_modalities):
                # validation qo_pi
                qo_pi_valid = maths.spm_dot(A[modality_idx],qs_pi[t_idx])
                self.assertTrue((qo_pi[t_idx][modality_idx] == qo_pi_valid).all())
        
        '''Test with multiple observation modalities, multiple hidden state factors and multiple timesteps'''

        num_obs = [3, 3]
        num_states = [3, 3]
        num_controls = [3, 2]

        num_factors = len(num_states)
        num_modalities = len(num_obs)

        qs = utils.obj_array_uniform(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        policies = control.construct_policies(num_states, num_controls, policy_len=3)
        
        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)

            for t_idx in range(3):
                for factor_idx in range(num_factors):
                    # validation qs_pi
                    if t_idx == 0:
                        qs_pi_valid = B[factor_idx][:,:,policies[idx][t_idx,factor_idx]].dot(qs[factor_idx])
                    else:
                        qs_pi_valid = B[factor_idx][:,:,policies[idx][t_idx,factor_idx]].dot(qs_pi[t_idx-1][factor_idx])

                    self.assertTrue((qs_pi[t_idx][factor_idx] == qs_pi_valid).all())

            qo_pi = control.get_expected_obs(qs_pi, A)

            for t_idx in range(3):
                for modality_idx in range(num_modalities):
                    # validation qo_pi
                    qo_pi_valid = maths.spm_dot(A[modality_idx],qs_pi[t_idx])
                    self.assertTrue((qo_pi[t_idx][modality_idx] == qo_pi_valid).all())

    def test_expected_utility(self):
        """
        Test for the expected utility function, for a simple single factor generative model 
        where there are imbalances in the preferences for different outcomes. Test for both single
        timestep policy horizons and multiple timestep policy horizons (planning)
        """

        '''1-step policies'''
        num_states = [2]
        num_controls = [2]

        qs = utils.random_single_categorical(num_states)
        B = utils.construct_controllable_B(num_states, num_controls)

        # Single timestep
        n_step = 1
        policies = control.construct_policies(num_states, num_controls, policy_len=n_step)

        # Single observation modality
        num_obs = [2]

        # Create noiseless identity A matrix
        A = utils.to_arr_of_arr(np.eye(num_obs[0]))

        # Create imbalance in preferences for observations
        C = utils.to_arr_of_arr(utils.onehot(1, num_obs[0]))
        
        # Compute expected utility of policies
        expected_utilities = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states(qs, B, policy)
            qo_pi = control.get_expected_obs(qs_pi, A)
            expected_utilities[idx] += control.calc_expected_utility(qo_pi, C)

        self.assertGreater(expected_utilities[1], expected_utilities[0])

        '''3-step policies'''
        # One policy entails going to state 0 two times in a row, and then state 2 at the end
        # Another policy entails going to state 1 three times in a row

        num_states = [3]
        num_controls = [3]

        qs = utils.random_single_categorical(num_states)
        B = utils.construct_controllable_B(num_states, num_controls)

        policies = [np.array([0, 0, 2]).reshape(-1, 1), np.array([1, 1, 1]).reshape(-1, 1)]

        # single observation modality
        num_obs = [3]

        # create noiseless identity A matrix
        A = utils.to_arr_of_arr(np.eye(num_obs[0]))

        # create imbalance in preferences for observations
        # This test is designed to illustrate the emergence of planning by
        # using the time-integral of the expected free energy.
        # Even though the first observation (index 0) is the most preferred, the policy
        # that frequents this observation the most is actually not optimal, because that policy
        # terminates in a less preferred state by timestep 3.
        C = utils.to_arr_of_arr(np.array([1.2, 1.0, 0.55]))

        expected_utilities = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states(qs, B, policy)
            qo_pi = control.get_expected_obs(qs_pi, A)
            expected_utilities[idx] += control.calc_expected_utility(qo_pi, C)

        self.assertGreater(expected_utilities[1], expected_utilities[0])
    
    def test_state_info_gain(self):
        """
        Test the states_info_gain function. 
        Function is tested by manipulating uncertainty in the likelihood matrices (A or B)
        in a ways that alternatively change the resolvability of uncertainty
        This is done with A) an imprecise expected state and a precise sensory mapping, 
        and B) high ambiguity and imprecise sensory mapping.
        """

        num_states = [2]
        num_controls = [2]

        # start with a precise initial state
        qs = utils.to_arr_of_arr(utils.onehot(0, num_states[0]))

        '''Case study 1: Uncertain states, unambiguous observations'''
        # add some uncertainty into the consequences of the second policy, which
        # leads to increased epistemic value of observations, in case of pursuing
        # that policy -- this of course depends on a precise observation likelihood model
        B = utils.construct_controllable_B(num_states, num_controls)
        B[0][:, :, 1] = maths.softmax(B[0][:, :, 1]) # "noise-ify" the consequences of the 1-th action

        # single timestep
        n_step = 1
        policies = control.construct_policies(num_states, num_controls, policy_len=n_step)

        # single observation modality
        num_obs = [2]

        # create noiseless identity A matrix
        A = utils.to_arr_of_arr(np.eye(num_obs[0]))

        state_info_gains = np.zeros(len(policies)) # store the Bayesian surprise / epistemic values of states here (AKA state info gain)
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states(qs, B, policy)
            state_info_gains[idx] += control.calc_states_info_gain(A, qs_pi)
        self.assertGreater(state_info_gains[1], state_info_gains[0])

        '''Case study 2: Uncertain states, ambiguous observations (for all states)'''
        # now we 'undo' the epistemic bonus of the second policy by making the A matrix
        # totally ambiguous; thus observations cannot resolve uncertainty about hidden states.
        # In this case, uncertainty in the posterior beliefs induced by Policy 1 doesn't tip the balance
        # of epistemic value, because uncertainty is irresolveable either way.
        A = utils.obj_array_uniform([ [num_obs[0], num_states[0]] ])

        state_info_gains = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states(qs, B, policy)
            state_info_gains[idx] += control.calc_states_info_gain(A, qs_pi)
        self.assertEqual(state_info_gains[0], state_info_gains[1])

        '''Case study 2: Uncertain states, ambiguous observations (for particular states)'''

        # create noiseless identity A matrix
        A = utils.to_arr_of_arr(np.eye(num_obs[0]))

        # add some uncertainty into the consequences of the both policies
        B = utils.construct_controllable_B(num_states, num_controls)

        B[0][:, :, 0] = maths.softmax(B[0][:, :, 0]*2.0) # "noise-ify" the consequences of the 0-th action, but to a lesser extent than the 1-th action
        B[0][:, :, 1] = maths.softmax(B[0][:, :, 1]) # "noise-ify" the consequences of the 1-th action

        # Although in the presence of a precise likelihood mapping, 
        # Policy 1 would be preferred (due to higher resolve-able uncertainty, introduced by a noisier action-dependent B matrix),
        # if the expected observation likelihood of being in state 1 (the most likely consequence of Policy 1) is not precise, then 
        # Policy 0 (which has more probability loaded over state 0) will have more resolveable uncertainty, due to the
        # higher precision of the A matrix over that column (column 0, which is identity). Even though the expected density over states
        # is less noisy for policy 0.
        A[0][:,1] = maths.softmax(A[0][:,1]) 

        state_info_gains = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states(qs, B, policy)
            state_info_gains[idx] += control.calc_states_info_gain(A, qs_pi)
        self.assertGreater(state_info_gains[1], state_info_gains[0])
    
    def test_pA_info_gain(self):

        """
        Test the pA_info_gain function. Demonstrates operation
        by manipulating shape of the Dirichlet priors over likelihood parameters
        (pA), which affects information gain for different expected observations
        """

        num_states = [2]
        num_controls = [2]

        # start with a precise initial state
        qs = utils.to_arr_of_arr(utils.onehot(0, num_states[0]))

        B = utils.construct_controllable_B(num_states, num_controls)

        # single timestep
        n_step = 1
        policies = control.construct_policies(num_states, num_controls, policy_len=n_step)

        # single observation modality
        num_obs = [2]

        # create noiseless identity A matrix
        A = utils.to_arr_of_arr(np.eye(num_obs[0]))

        # create prior over dirichlets such that there is a skew
        # in the parameters about the likelihood mapping from the
        # second hidden state (index 1) to observations, such that
        # Observation 0 is believed to be more likely than the other, conditioned on State 1.
        # Therefore sampling observations conditioned on State 1 would afford high info gain
        # about parameters, for that part of the likelhood distribution.

        pA = utils.obj_array_ones([ [num_obs[0], num_states[0]]] )
        pA[0][0, 1] += 1.0

        pA_info_gains = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states(qs, B, policy)
            qo_pi = control.get_expected_obs(qs_pi, A)
            pA_info_gains[idx] += control.calc_pA_info_gain(pA, qo_pi, qs_pi)

        self.assertGreater(pA_info_gains[1], pA_info_gains[0])
    
    def test_pB_info_gain(self):
        """
        Test the pB_info_gain function. Demonstrates operation
        by manipulating shape of the Dirichlet priors over likelihood parameters
        (pB), which affects information gain for different states
        """
        num_states = [2]
        num_controls = [2]

        # start with a precise initial state
        qs = utils.to_arr_of_arr(utils.onehot(0, num_states[0]))

        B = utils.construct_controllable_B(num_states, num_controls)

        pB = utils.obj_array_ones([ [num_states[0], num_states[0], num_controls[0]] ])

        # create prior over dirichlets such that there is a skew
        # in the parameters about the likelihood mapping from the
        # hidden states to hidden states under the second action,
        # such that hidden state 0 is considered to be more likely than the other,
        # given the action in question
        # Therefore taking that action would yield an expected state that afford
        # high information gain about that part of the likelihood distribution.
        #
        pB[0][0, :, 1] += 1.0

        # single timestep
        n_step = 1
        policies = control.construct_policies(num_states, num_controls, policy_len=n_step)

        pB_info_gains = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states(qs, B, policy)
            pB_info_gains[idx] += control.calc_pB_info_gain(pB, qs_pi, qs, policy)
        self.assertGreater(pB_info_gains[1], pB_info_gains[0])

    def test_update_posterior_policies_utility(self):
        """
        Tests the refactored (Categorical-less) version of `update_posterior_policies`, using only the expected utility component of the expected free energy
        """

        '''Test with single observation modality, single hidden state factor and single timestep'''

        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        qs = utils.obj_array_uniform(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        C = utils.obj_array_zeros(num_obs)
        C[0][0] = 1.0  
        C[0][1] = -2.0  

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = True,
            use_states_info_gain = False,
            use_param_info_gain = False,
            pA=None,
            pB=None,
            gamma=16.0
        )

        factor_idx = 0
        modality_idx = 0
        t_idx = 0

        efe_valid = np.zeros(len(policies))

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)
            
            qo_pi = control.get_expected_obs(qs_pi, A)

            lnC = maths.spm_log_single(maths.softmax(C[modality_idx][:, np.newaxis]))
            efe_valid[idx] += qo_pi[t_idx][modality_idx].dot(lnC)
        
        q_pi_valid = maths.softmax(efe_valid * 16.0)

        self.assertTrue(np.allclose(efe, efe_valid))
        self.assertTrue(np.allclose(q_pi, q_pi_valid))

        '''Test with multiple observation modalities, multiple hidden state factors and single timestep'''

        num_obs = [3, 3]
        num_states = [3, 2]
        num_controls = [3, 1]

        qs = utils.random_single_categorical(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        C = utils.obj_array_zeros(num_obs)
        C[0][0] = 1.0  
        C[0][1] = -2.0
        C[1][2] = 4.0  

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = True,
            use_states_info_gain = False,
            use_param_info_gain = False,
            pA=None,
            pB=None,
            gamma=16.0
        )

        t_idx = 0

        efe_valid = np.zeros(len(policies))

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)
            qo_pi = control.get_expected_obs(qs_pi, A)

            for modality_idx in range(len(A)):
                lnC = maths.spm_log_single(maths.softmax(C[modality_idx][:, np.newaxis]))
                efe_valid[idx] += qo_pi[t_idx][modality_idx].dot(lnC)
        
        q_pi_valid = maths.softmax(efe_valid * 16.0)

        self.assertTrue(np.allclose(efe, efe_valid))
        self.assertTrue(np.allclose(q_pi, q_pi_valid))

        '''Test with multiple observation modalities, multiple hidden state factors and multiple timesteps'''

        num_obs = [3, 3]
        num_states = [3, 2]
        num_controls = [3, 1]

        qs = utils.random_single_categorical(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        C = utils.obj_array_zeros(num_obs)
        C[0][0] = 1.0  
        C[0][1] = -2.0
        C[1][2] = 4.0  

        policies = control.construct_policies(num_states, num_controls, policy_len=3)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = True,
            use_states_info_gain = False,
            use_param_info_gain = False,
            pA=None,
            pB=None,
            gamma=16.0
        )

        efe_valid = np.zeros(len(policies))

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)
            qo_pi = control.get_expected_obs(qs_pi, A)

            for t_idx in range(3):
                for modality_idx in range(len(A)):
                    lnC = maths.spm_log_single(maths.softmax(C[modality_idx][:, np.newaxis]))
                    efe_valid[idx] += qo_pi[t_idx][modality_idx].dot(lnC)
        
        q_pi_valid = maths.softmax(efe_valid * 16.0)

        self.assertTrue(np.allclose(efe, efe_valid))
        self.assertTrue(np.allclose(q_pi, q_pi_valid))

    def test_update_posterior_policies_states_infogain(self):
        """
        Tests the refactored (Categorical-less) version of `update_posterior_policies`, using only the information gain (about states) component of the expected free energy
        """

        '''Test with single observation modality, single hidden state factor and single timestep'''

        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        qs = utils.obj_array_uniform(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        C = utils.obj_array_zeros(num_obs)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = False,
            use_states_info_gain = True,
            use_param_info_gain = False,
            pA=None,
            pB=None,
            gamma=16.0
        )

        factor_idx = 0
        modality_idx = 0
        t_idx = 0

        efe_valid = np.zeros(len(policies))

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)
            
            efe_valid[idx] += maths.spm_MDP_G(A, qs_pi[0])
        
        q_pi_valid = maths.softmax(efe_valid * 16.0)

        self.assertTrue(np.allclose(efe, efe_valid))
        self.assertTrue(np.allclose(q_pi, q_pi_valid))

        '''Test with multiple observation modalities, multiple hidden state factors and single timestep'''

        num_obs = [3, 3]
        num_states = [3, 2]
        num_controls = [3, 1]

        qs = utils.random_single_categorical(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        C = utils.obj_array_zeros(num_obs)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = False,
            use_states_info_gain = True,
            use_param_info_gain = False,
            pA=None,
            pB=None,
            gamma=16.0
        )

        t_idx = 0

        efe_valid = np.zeros(len(policies))

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)

            efe_valid[idx] += maths.spm_MDP_G(A, qs_pi[0])
        
        q_pi_valid = maths.softmax(efe_valid * 16.0)

        self.assertTrue(np.allclose(efe, efe_valid))
        self.assertTrue(np.allclose(q_pi, q_pi_valid))

        '''Test with multiple observation modalities, multiple hidden state factors and multiple timesteps'''

        num_obs = [3, 3]
        num_states = [3, 2]
        num_controls = [3, 1]

        qs = utils.random_single_categorical(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        C = utils.obj_array_zeros(num_obs) 

        policies = control.construct_policies(num_states, num_controls, policy_len=3)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = False,
            use_states_info_gain = True,
            use_param_info_gain = False,
            pA=None,
            pB=None,
            gamma=16.0
        )

        efe_valid = np.zeros(len(policies))

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)

            for t_idx in range(3):
                efe_valid[idx] += maths.spm_MDP_G(A, qs_pi[t_idx])
    
        q_pi_valid = maths.softmax(efe_valid * 16.0)

        self.assertTrue(np.allclose(efe, efe_valid))
        self.assertTrue(np.allclose(q_pi, q_pi_valid))

    def test_update_posterior_policies_pA_infogain(self):
        """
        Tests the refactored (Categorical-less) version of `update_posterior_policies`, using only the information gain (about likelihood parameters) component of the expected free energy
        """

        '''Test with single observation modality, single hidden state factor and single timestep'''

        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        qs = utils.obj_array_uniform(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])

        B = utils.random_B_matrix(num_states, num_controls)
        C = utils.obj_array_zeros(num_obs)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = False,
            use_states_info_gain = False,
            use_param_info_gain = True,
            pA=pA,
            pB=None,
            gamma=16.0
        )

        factor_idx = 0
        modality_idx = 0
        t_idx = 0

        efe_valid = np.zeros(len(policies))

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)
            qo_pi = control.get_expected_obs(qs_pi, A)
            
            efe_valid[idx] += control.calc_pA_info_gain(pA, qo_pi, qs_pi)
        
        q_pi_valid = maths.softmax(efe_valid * 16.0)

        self.assertTrue(np.allclose(efe, efe_valid))
        self.assertTrue(np.allclose(q_pi, q_pi_valid))

        '''Test with multiple observation modalities, multiple hidden state factors and single timestep'''

        num_obs = [3, 3]
        num_states = [3, 2]
        num_controls = [3, 1]

        qs = utils.random_single_categorical(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])

        B = utils.random_B_matrix(num_states, num_controls)
        C = utils.obj_array_zeros(num_obs)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = False,
            use_states_info_gain = False,
            use_param_info_gain = True,
            pA=pA,
            pB=None,
            gamma=16.0
        )

        t_idx = 0

        efe_valid = np.zeros(len(policies))

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)
            qo_pi = control.get_expected_obs(qs_pi, A)
            
            efe_valid[idx] += control.calc_pA_info_gain(pA, qo_pi, qs_pi)
        
        q_pi_valid = maths.softmax(efe_valid * 16.0)

        self.assertTrue(np.allclose(efe, efe_valid))
        self.assertTrue(np.allclose(q_pi, q_pi_valid))

        '''Test with multiple observation modalities, multiple hidden state factors and multiple timesteps'''

        num_obs = [3, 3]
        num_states = [3, 2]
        num_controls = [3, 1]

        qs = utils.random_single_categorical(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])

        B = utils.random_B_matrix(num_states, num_controls)
        C = utils.obj_array_zeros(num_obs) 

        policies = control.construct_policies(num_states, num_controls, policy_len=3)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = False,
            use_states_info_gain = False,
            use_param_info_gain = True,
            pA=pA,
            pB=None,
            gamma=16.0
        )

        efe_valid = np.zeros(len(policies))

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)
            qo_pi = control.get_expected_obs(qs_pi, A)

            efe_valid[idx] += control.calc_pA_info_gain(pA, qo_pi, qs_pi)
    
        q_pi_valid = maths.softmax(efe_valid * 16.0)

        self.assertTrue(np.allclose(efe, efe_valid))
        self.assertTrue(np.allclose(q_pi, q_pi_valid))
    
    def test_update_posterior_policies_pB_infogain(self):
        """
        Tests the refactored (Categorical-less) version of `update_posterior_policies`, using only the information gain (about transition likelihood parameters) component of the expected free energy
        """

        '''Test with single observation modality, single hidden state factor and single timestep'''

        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        qs = utils.obj_array_uniform(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        C = utils.obj_array_zeros(num_obs)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = False,
            use_states_info_gain = False,
            use_param_info_gain = True,
            pA=None,
            pB=pB,
            gamma=16.0
        )

        factor_idx = 0
        modality_idx = 0
        t_idx = 0

        efe_valid = np.zeros(len(policies))

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)
            
            efe_valid[idx] += control.calc_pB_info_gain(pB, qs_pi, qs, policy)
        
        q_pi_valid = maths.softmax(efe_valid * 16.0)

        self.assertTrue(np.allclose(efe, efe_valid))
        self.assertTrue(np.allclose(q_pi, q_pi_valid))

        '''Test with multiple observation modalities, multiple hidden state factors and single timestep'''

        num_obs = [3, 3]
        num_states = [3, 2]
        num_controls = [3, 1]

        qs = utils.random_single_categorical(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])
        
        C = utils.obj_array_zeros(num_obs)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = False,
            use_states_info_gain = False,
            use_param_info_gain = True,
            pA=None,
            pB=pB,
            gamma=16.0
        )

        t_idx = 0

        efe_valid = np.zeros(len(policies))

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)
            
            efe_valid[idx] += control.calc_pB_info_gain(pB, qs_pi, qs, policy)
        
        q_pi_valid = maths.softmax(efe_valid * 16.0)

        self.assertTrue(np.allclose(efe, efe_valid))
        self.assertTrue(np.allclose(q_pi, q_pi_valid))

        '''Test with multiple observation modalities, multiple hidden state factors and multiple timesteps'''

        num_obs = [3, 3]
        num_states = [3, 2]
        num_controls = [3, 1]

        qs = utils.random_single_categorical(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        C = utils.obj_array_zeros(num_obs) 

        policies = control.construct_policies(num_states, num_controls, policy_len=3)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = False,
            use_states_info_gain = False,
            use_param_info_gain = True,
            pA=None,
            pB=pB,
            gamma=16.0
        )

        efe_valid = np.zeros(len(policies))

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)

            efe_valid[idx] += control.calc_pB_info_gain(pB, qs_pi, qs, policy)
    
        q_pi_valid = maths.softmax(efe_valid * 16.0)

        self.assertTrue(np.allclose(efe, efe_valid))
        self.assertTrue(np.allclose(q_pi, q_pi_valid))

    def test_sample_action(self):
        """
        Tests the refactored (Categorical-less) version of `sample_action`
        """

        '''Test with single observation modality, single hidden state factor and single timestep'''

        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        qs = utils.obj_array_uniform(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        C = utils.obj_array_zeros(num_obs)
        C[0][0] = 1.0  
        C[0][1] = -2.0

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = True,
            use_states_info_gain = True,
            use_param_info_gain = True,
            pA=pA,
            pB=pB,
            gamma=16.0
        )

        factor_idx = 0
        modality_idx = 0
        t_idx = 0

        chosen_action = control.sample_action(q_pi, policies, num_controls, action_selection="deterministic")
        sampled_action = control.sample_action(q_pi, policies, num_controls, action_selection="stochastic", alpha = 1.0)

        self.assertEqual(chosen_action.shape, sampled_action.shape)

        '''Test with multiple observation modalities, multiple hidden state factors and single timestep'''

        num_obs = [3, 4]
        num_states = [3, 4]
        num_controls = [3, 3]

        qs = utils.random_single_categorical(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        C = utils.obj_array_zeros(num_obs)
        C[0][0] = 1.0  
        C[0][1] = -2.0
        C[1][3] = 3.0

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = True,
            use_states_info_gain = True,
            use_param_info_gain = True,
            pA=pA,
            pB=pB,
            gamma=16.0
        )

        chosen_action = control.sample_action(q_pi, policies, num_controls, action_selection="deterministic")
        sampled_action = control.sample_action(q_pi, policies, num_controls, action_selection="stochastic", alpha = 1.0)

        self.assertEqual(chosen_action.shape, sampled_action.shape)

        '''Test with multiple observation modalities, multiple hidden state factors and multiple timesteps'''

        num_obs = [3, 4]
        num_states = [3, 4]
        num_controls = [3, 3]

        qs = utils.random_single_categorical(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        C = utils.obj_array_zeros(num_obs)
        C[0][0] = 1.0  
        C[0][1] = -2.0
        C[1][3] = 3.0

        policies = control.construct_policies(num_states, num_controls, policy_len=3)

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = True,
            use_states_info_gain = True,
            use_param_info_gain = True,
            pA=pA,
            pB=pB,
            gamma=16.0
        )

        chosen_action = control.sample_action(q_pi, policies, num_controls, action_selection="deterministic")
        sampled_action = control.sample_action(q_pi, policies, num_controls, action_selection="stochastic", alpha = 1.0)

        self.assertEqual(chosen_action.shape, sampled_action.shape)


        '''Single observation modality, single (controllable) hidden state factor, 3-step policies. Using utility only'''
        # One policy entails going to state 0 two times in a row, and then state 2 at the end
        # Another policy entails going to state 1 three times in a row

        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        qs = utils.random_single_categorical(num_states)
        B = utils.construct_controllable_B(num_states, num_controls)

        policies = [np.array([0, 0, 2]).reshape(-1, 1), np.array([1, 1, 1]).reshape(-1, 1)]

        # create noiseless identity A matrix
        A = utils.to_arr_of_arr(np.eye(num_obs[0]))

        # create imbalance in preferences for observations
        # This test is designed to illustrate the emergence of planning by
        # using the time-integral of the expected free energy.
        # Even though the first observation (index 0) is the most preferred, the policy
        # that frequents this observation the most is actually not optimal, because that policy
        # terminates in a less preferred state by timestep 3.
        C = utils.to_arr_of_arr(np.array([1.2, 1.0, 0.55]))

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility = True,
            use_states_info_gain = False,
            use_param_info_gain = False,
            pA=None,
            pB=None,
            gamma=16.0
        )

        chosen_action = control.sample_action(q_pi, policies, num_controls, action_selection="deterministic")
        self.assertEqual(int(chosen_action[0]), 1)


if __name__ == "__main__":
    unittest.main()
