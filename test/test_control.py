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

    def test_get_expected_states_interactions_single_factor(self):
        """
        Test the new version of `get_expected_states` that includes `B` array inter-factor dependencies, in case a of trivial single factor
        """
        
        num_states = [3]
        num_controls = [3]

        B_factor_list = [[0]]
        B_factor_control_list = [[0]]

        qs = utils.random_single_categorical(num_states)
        B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list, B_factor_control_list=B_factor_control_list)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        qs_pi_0 = control.get_expected_states_interactions(qs, B, B_factor_list, B_factor_control_list, policies[0])
        
        self.assertTrue(np.allclose(qs_pi_0[0][0], B[0][:,:,policies[0][0,0]].dot(qs[0])))

    def test_get_expected_states_interactions_multi_factor(self):
        """
        Test the new version of `get_expected_states` that includes `B` array inter-factor dependencies, 
        in the case where there are two hidden state factors: one that depends on itself and another that depends on both itself and the other factor.
        """
        
        num_states = [3, 4]
        num_controls = [3, 2]

        B_factor_list = [[0], [0, 1]]
        B_factor_control_list = [[0], [1]]

        qs = utils.random_single_categorical(num_states)
        B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list, B_factor_control_list=B_factor_control_list)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        qs_pi_0 = control.get_expected_states_interactions(qs, B, B_factor_list, B_factor_control_list, policies[0])

        self.assertTrue(np.allclose(qs_pi_0[0][0], B[0][:,:,policies[0][0,0]].dot(qs[0])))

        qs_next_validation = (B[1][..., policies[0][0,1]] * maths.spm_cross(qs)[None,...]).sum(axis=(1,2)) # how to compute equivalent of `spm_dot(B[...,past_action], qs)`
        self.assertTrue(np.allclose(qs_pi_0[0][1], qs_next_validation))
    
    def test_get_expected_states_interactions_multi_factor_independent(self):
        """
        Test the new version of `get_expected_states` that includes `B` array inter-factor dependencies, 
        in the case where there are multiple hidden state factors, but they all only depend on themselves
        """
        
        num_states = [3, 4, 5, 6]
        num_controls = [1, 2, 5, 3]

        B_factor_list = [[f] for f in range(len(num_states))] # each factor only depends on itself
        B_factor_control_list = [[f] for f in range(len(num_states))] # each factor only depends on its own action

        qs = utils.random_single_categorical(num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        qs_pi_0 = control.get_expected_states_interactions(qs, B, B_factor_list, B_factor_control_list, policies[0])

        qs_pi_0_validation = control.get_expected_states(qs, B, policies[0])

        for qs_f, qs_val_f in zip(qs_pi_0[0], qs_pi_0_validation[0]):
            self.assertTrue(np.allclose(qs_f, qs_val_f))
    
    def test_get_expected_states_interactions_multi_factor_single_action(self):
        """
        Test the new version of `get_expected_states` that includes `B` array inter-factor dependencies, 
        in the case where there are two hidden state factors: one that depends on itself and another that depends on both itself and the other factor,
        and both states depend on the same action.
        """
        
        num_states = [3, 4]
        num_controls = [5]

        B_factor_list = [[0], [0, 1]]
        B_factor_control_list = [[0], [0]]

        qs = utils.random_single_categorical(num_states)
        B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list, B_factor_control_list=B_factor_control_list)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        qs_pi_0 = control.get_expected_states_interactions(qs, B, B_factor_list, B_factor_control_list, policies[0])

        self.assertTrue(np.allclose(qs_pi_0[0][0], B[0][:,:,policies[0][0,0]].dot(qs[0])))

        qs_next_validation = (B[1][..., policies[0][0,0]] * maths.spm_cross(qs)[None,...]).sum(axis=(1,2)) # how to compute equivalent of `spm_dot(B[...,past_action], qs)`
        self.assertTrue(np.allclose(qs_pi_0[0][1], qs_next_validation))

    def test_get_expected_states_interactions_multi_factor_multi_action(self):
        """
        Test the new version of `get_expected_states` that includes `B` array inter-factor dependencies, 
        in the case where there are two hidden state factors: one that depends on itself and another that depends on both itself and the other factor,
        and one state depends on two actions.
        """
        
        num_states = [3, 4]
        num_controls = [5, 6]

        B_factor_list = [[0], [0, 1]]
        B_factor_control_list = [[0, 1], [0]]

        qs = utils.random_single_categorical(num_states)
        B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list, B_factor_control_list=B_factor_control_list)
        
        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        qs_pi_0 = control.get_expected_states_interactions(qs, B, B_factor_list, B_factor_control_list, policies[0])

        self.assertTrue(np.allclose(qs_pi_0[0][0], B[0][:,:,policies[0][0,0],policies[0][0,1]].dot(qs[0])))

        qs_next_validation = (B[1][..., policies[0][0,0]] * maths.spm_cross(qs)[None,...]).sum(axis=(1,2)) # how to compute equivalent of `spm_dot(B[...,past_action], qs)`
        self.assertTrue(np.allclose(qs_pi_0[0][1], qs_next_validation))

    def test_get_expected_obs_factorized(self):
        """
        Test the new version of `get_expected_obs` that includes sparse dependencies of `A` array on hidden state factors (not all observation modalities depend on all hidden state factors)
        """

        """ Case 1, where all modalities depend on all hidden state factors """

        num_states = [3, 4]
        num_obs = [3, 4]

        A_factor_list = [[0, 1], [0, 1]]

        qs = utils.random_single_categorical(num_states)
        A = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_factor_list)

        qo_test = control.get_expected_obs_factorized([qs], A, A_factor_list) # need to wrap `qs` in list because `get_expected_obs_factorized` expects a list of `qs` (representing multiple timesteps)
        qo_val = control.get_expected_obs([qs], A) # need to wrap `qs` in list because `get_expected_obs` expects a list of `qs` (representing multiple timesteps)

        for qo_m, qo_val_m in zip(qo_test[0], qo_val[0]): # need to extract first index of `qo_test` and `qo_val` because `get_expected_obs_factorized` returns a list of `qo` (representing multiple timesteps)
            self.assertTrue(np.allclose(qo_m, qo_val_m))
        
        """ Case 2, where some modalities depend on some hidden state factors """

        num_states = [3, 4]
        num_obs = [3, 4]

        A_factor_list = [[0], [0, 1]]

        qs = utils.random_single_categorical(num_states)
        A_reduced = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_factor_list)

        qo_test = control.get_expected_obs_factorized([qs], A_reduced, A_factor_list) # need to wrap `qs` in list because `get_expected_obs_factorized` expects a list of `qs` (representing multiple timesteps)

        A_full = utils.initialize_empty_A(num_obs, num_states)
        for m, A_m in enumerate(A_full):
            other_factors = list(set(range(len(num_states))) - set(A_factor_list[m])) # list of the factors that modality `m` does not depend on

            # broadcast or tile the reduced A matrix (`A_reduced`) along the dimensions of corresponding to `other_factors`
            expanded_dims = [num_obs[m]] + [1 if f in other_factors else ns for (f, ns) in enumerate(num_states)]
            tile_dims = [1] + [ns if f in other_factors else 1 for (f, ns) in enumerate(num_states)]
            A_full[m] = np.tile(A_reduced[m].reshape(expanded_dims), tile_dims)
        
        qo_val = control.get_expected_obs([qs], A_full) # need to wrap `qs` in list because `get_expected_obs` expects a list of `qs` (representing multiple timesteps)
        
        for qo_m, qo_val_m in zip(qo_test[0], qo_val[0]): # need to extract first index of `qo_test` and `qo_val` because `get_expected_obs_factorized` returns a list of `qo` (representing multiple timesteps)
            self.assertTrue(np.allclose(qo_m, qo_val_m))

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
            qo_pi_valid = maths.spm_dot(A[modality_idx],utils.to_obj_array(qs_pi_valid))

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
        A = utils.to_obj_array(np.eye(num_obs[0]))

        # Create imbalance in preferences for observations
        C = utils.to_obj_array(utils.onehot(1, num_obs[0]))
        
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
        A = utils.to_obj_array(np.eye(num_obs[0]))

        # create imbalance in preferences for observations
        # This test is designed to illustrate the emergence of planning by
        # using the time-integral of the expected free energy.
        # Even though the first observation (index 0) is the most preferred, the policy
        # that frequents this observation the most is actually not optimal, because that policy
        # terminates in a less preferred state by timestep 3.
        C = utils.to_obj_array(np.array([1.2, 1.0, 0.55]))

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
        qs = utils.to_obj_array(utils.onehot(0, num_states[0]))

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
        A = utils.to_obj_array(np.eye(num_obs[0]))

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
        A = utils.to_obj_array(np.eye(num_obs[0]))

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

    def test_state_info_gain_factorized(self):
        """ 
        Unit test the `calc_states_info_gain_factorized` function by qualitatively checking that in the T-Maze (contextual bandit)
        example, the state info gain is higher for the policy that leads to visiting the cue, which is higher than state info gain
        for visiting the bandit arm, which in turn is higher than the state info gain for the policy that leads to staying in the start state.
        """

        num_states = [2, 3]  
        num_obs = [3, 3, 3]
        num_controls = [1, 3]

        A_factor_list = [[0, 1], [0, 1], [1]] 
        
        A = utils.obj_array(len(num_obs))
        for m, obs in enumerate(num_obs):
            lagging_dimensions = [ns for i, ns in enumerate(num_states) if i in A_factor_list[m]]
            modality_shape = [obs] + lagging_dimensions
            A[m] = np.zeros(modality_shape)
            if m == 0:
                A[m][:, :, 0] = np.ones( (num_obs[m], num_states[0]) ) / num_obs[m]
                A[m][:, :, 1] = np.ones( (num_obs[m], num_states[0]) ) / num_obs[m]
                A[m][:, :, 2] = np.array([[0.9, 0.1], [0.0, 0.0], [0.1, 0.9]]) # cue statistics
            if m == 1:
                A[m][2, :, 0] = np.ones(num_states[0])
                A[m][0:2, :, 1] = np.array([[0.6, 0.4], [0.6, 0.4]]) # bandit statistics (mapping between reward-state (first hidden state factor) and rewards (Good vs Bad))
                A[m][2, :, 2] = np.ones(num_states[0])
            if m == 2:
                A[m] = np.eye(obs)

        qs_start = utils.obj_array_uniform(num_states)
        qs_start[1] = np.array([1., 0., 0.]) # agent believes it's in the start state

        state_info_gain_visit_start = 0.
        for m, A_m in enumerate(A):
            if len(A_factor_list[m]) == 1:
                qs_that_matter = utils.to_obj_array(qs_start[A_factor_list[m]])
            else:
                qs_that_matter = qs_start[A_factor_list[m]]
            state_info_gain_visit_start += control.calc_states_info_gain(A_m, [qs_that_matter])

        qs_arm = utils.obj_array_uniform(num_states)
        qs_arm[1] = np.array([0., 1., 0.]) # agent believes it's in the arm-visiting state

        state_info_gain_visit_arm = 0.
        for m, A_m in enumerate(A):
            if len(A_factor_list[m]) == 1:
                qs_that_matter = utils.to_obj_array(qs_arm[A_factor_list[m]])
            else:
                qs_that_matter = qs_arm[A_factor_list[m]]
            state_info_gain_visit_arm += control.calc_states_info_gain(A_m, [qs_that_matter])

        qs_cue = utils.obj_array_uniform(num_states)
        qs_cue[1] = np.array([0., 0., 1.]) # agent believes it's in the cue-visiting state

        state_info_gain_visit_cue = 0.
        for m, A_m in enumerate(A):
            if len(A_factor_list[m]) == 1:
                qs_that_matter = utils.to_obj_array(qs_cue[A_factor_list[m]])
            else:
                qs_that_matter = qs_cue[A_factor_list[m]]
            state_info_gain_visit_cue += control.calc_states_info_gain(A_m, [qs_that_matter])
        
        self.assertGreater(state_info_gain_visit_arm, state_info_gain_visit_start)
        self.assertGreater(state_info_gain_visit_cue, state_info_gain_visit_arm)

    # def test_neg_ambiguity_modality_sum(self):
    #     """
    #     Test that the negativity ambiguity function is the same when computed using the full (unfactorized) joint distribution over observations and hidden state factors vs. when computed for each modality separately and summed together.
    #     """

    #     num_states = [10, 20, 10, 10]
    #     num_obs = [2, 25, 10, 8]

    #     qs = utils.random_single_categorical(num_states)
    #     A = utils.random_A_matrix(num_obs, num_states)

    #     neg_ambig_full = maths.spm_calc_neg_ambig(A, qs) # need to wrap `qs` in a list because the function expects a list of policy-conditioned posterior beliefs (corresponding to each timestep)
    #     neg_ambig_by_modality = 0.
    #     for m, A_m in enumerate(A):
    #         neg_ambig_by_modality += maths.spm_calc_neg_ambig(A_m, qs)
        
    #     self.assertEqual(neg_ambig_full, neg_ambig_by_modality)
    
    # def test_entropy_modality_sum(self):
    #     """
    #     Test that the negativity ambiguity function is the same when computed using the full (unfactorized) joint distribution over observations and hidden state factors vs. when computed for each modality separately and summed together.
    #     """

    #     num_states = [10, 20, 10, 10]
    #     num_obs = [2, 25, 10, 8]

    #     qs = utils.random_single_categorical(num_states)
    #     A = utils.random_A_matrix(num_obs, num_states)

    #     H_full = maths.spm_calc_qo_entropy(A, qs) # need to wrap `qs` in a list because the function expects a list of policy-conditioned posterior beliefs (corresponding to each timestep)
    #     H_by_modality = 0.
    #     for m, A_m in enumerate(A):
    #         H_by_modality += maths.spm_calc_qo_entropy(A_m, qs)
        
    #     self.assertEqual(H_full, H_by_modality)

    def test_pA_info_gain(self):
        """
        Test the pA_info_gain function. Demonstrates operation
        by manipulating shape of the Dirichlet priors over likelihood parameters
        (pA), which affects information gain for different expected observations
        """

        num_states = [2]
        num_controls = [2]

        # start with a precise initial state
        qs = utils.to_obj_array(utils.onehot(0, num_states[0]))

        B = utils.construct_controllable_B(num_states, num_controls)

        # single timestep
        n_step = 1
        policies = control.construct_policies(num_states, num_controls, policy_len=n_step)

        # single observation modality
        num_obs = [2]

        # create noiseless identity A matrix
        A = utils.to_obj_array(np.eye(num_obs[0]))

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

        """ Test the factorized version of the pA_info_gain function. """
        pA_info_gains_fac = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states(qs, B, policy)
            qo_pi = control.get_expected_obs_factorized(qs_pi, A, A_factor_list=[[0]])
            pA_info_gains_fac[idx] += control.calc_pA_info_gain_factorized(pA, qo_pi, qs_pi, A_factor_list=[[0]])

        self.assertTrue(np.allclose(pA_info_gains_fac,  pA_info_gains))  
    
    def test_pB_info_gain(self):
        """
        Test the pB_info_gain function. Demonstrates operation
        by manipulating shape of the Dirichlet priors over likelihood parameters
        (pB), which affects information gain for different states
        """
        num_states = [2]
        num_controls = [2]

        # start with a precise initial state
        qs = utils.to_obj_array(utils.onehot(0, num_states[0]))

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

        B_factor_list = [[0]]
        B_factor_control_list = [[0]]
        pB_info_gains_interactions = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states_interactions(qs, B, B_factor_list, B_factor_control_list, policy)
            pB_info_gains_interactions[idx] += control.calc_pB_info_gain_interactions(pB, qs_pi, qs, B_factor_list, B_factor_control_list, policy)
        self.assertTrue(np.allclose(pB_info_gains_interactions, pB_info_gains))
    
    """TODO: currently just testing the function can run. need to properly test the infogain value, but shouldn't be any different from single action"""
    def test_pB_info_gain_multi_action(self):
        """
        Test the pB_info_gain function. Demonstrates operation
        by manipulating shape of the Dirichlet priors over likelihood parameters
        (pB), which affects information gain for different states.
        Multi action version smoke test
        """
        num_states = [4, 5]
        num_controls = [2, 3]
        B_factor_list = [[0], [1]]
        B_factor_control_list = [[0, 1], [1]]

        # start with a precise initial state
        qs = np.array([utils.onehot(0, d) for d in num_states], dtype=object)

        B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list, B_factor_control_list=B_factor_control_list)

        pB = np.array([np.zeros_like(a) for a in B], dtype=object)

        # create prior over dirichlets such that there is a skew
        # in the parameters about the likelihood mapping from the
        # hidden states to hidden states under the second action,
        # such that hidden state 0 is considered to be more likely than the other,
        # given the action in question
        # Therefore taking that action would yield an expected state that afford
        # high information gain about that part of the likelihood distribution.
        #
        # pB[0][0, :, 1] += 1.0

        # single timestep
        n_step = 1
        policies = control.construct_policies(num_states, num_controls, policy_len=n_step)

        # pB_info_gains = np.zeros(len(policies))
        # for idx, policy in enumerate(policies):
        #     qs_pi = control.get_expected_states(qs, B, policy)
        #     pB_info_gains[idx] += control.calc_pB_info_gain(pB, qs_pi, qs, policy)
        # self.assertGreater(pB_info_gains[1], pB_info_gains[0])

        pB_info_gains_interactions = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states_interactions(qs, B, B_factor_list, B_factor_control_list, policy)
            pB_info_gains_interactions[idx] += control.calc_pB_info_gain_interactions(pB, qs_pi, qs, B_factor_list, B_factor_control_list, policy)
        # self.assertTrue(np.allclose(pB_info_gains_interactions, pB_info_gains))

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
    
    def test_temporal_C_matrix(self):
        """ Unit-tests for preferences that change over time """

        '''Test with single observation modality, single hidden state factor and single timestep, and non-temporal C vector'''

        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        qs = utils.obj_array_uniform(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        C = utils.obj_array_zeros(num_obs)
        C[0][0] = 1.0  
        C[0][1] = -2.0  

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

        factor_idx = 0
        modality_idx = 0
        t_idx = 0

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

        '''Test with single observation modality, single hidden state factor and single timestep, and temporal C vector'''

        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        qs = utils.obj_array_uniform(num_states)
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        C = utils.obj_array_zeros([(3,3)])
        C[0][0,:] = np.array([1.0, 2.0, 0.0])
        C[0][1,:] = np.array([-2.0, -1.0, 0.0])  

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

        factor_idx = 0
        modality_idx = 0
        t_idx = 0

        efe_valid = np.zeros(len(policies))

        for idx, policy in enumerate(policies):

            qs_pi = control.get_expected_states(qs, B, policy)
            qo_pi = control.get_expected_obs(qs_pi, A)

            for t_idx in range(3):
                for modality_idx in range(len(A)):
                    lnC = maths.spm_log_single(maths.softmax(C[modality_idx][:, t_idx]))
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
        C = utils.obj_array(len(num_obs))

        # C vectors for modalities 0 is time-dependent
        C[0] = np.random.rand(3, 3) 

        # C vectors for modalities 1 is time-independent
        C[1] = np.random.rand(3)

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
                    if modality_idx == 0:
                        lnC = maths.spm_log_single(maths.softmax(C[modality_idx][:, t_idx]))
                    elif modality_idx == 1:
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
    
    def test_update_posterior_policies_factorized(self):
        """ 
        Test new update_posterior_policies_factorized function, just to make sure it runs through and outputs correct shapes
        """

        num_obs = [3, 3]
        num_states = [3, 2]
        num_controls = [3, 2]

        A_factor_list = [[0, 1], [1]]
        B_factor_list = [[0], [0, 1]]
        B_factor_control_list = [[0], [1]]

        qs = utils.random_single_categorical(num_states)
        A = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_factor_list)
        B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list)
        C = utils.obj_array_zeros(num_obs)

        policies = control.construct_policies(num_states, num_controls, policy_len=1)

        q_pi, efe = control.update_posterior_policies_factorized(
            qs,
            A,
            B,
            C,
            A_factor_list,
            B_factor_list,
            B_factor_control_list,
            policies,
            use_utility = True,
            use_states_info_gain = True,
            gamma=16.0
        )

        self.assertEqual(len(q_pi), len(policies))
        self.assertEqual(len(efe), len(policies))

        chosen_action = control.sample_action(q_pi, policies, num_controls, action_selection="deterministic")

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
        sampled_action = control.sample_action(q_pi, policies, num_controls, action_selection="stochastic", alpha=1.0)

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
        sampled_action = control.sample_action(q_pi, policies, num_controls, action_selection="stochastic", alpha=1.0)

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
        sampled_action = control.sample_action(q_pi, policies, num_controls, action_selection="stochastic", alpha=1.0)

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
        A = utils.to_obj_array(np.eye(num_obs[0]))

        # create imbalance in preferences for observations
        # This test is designed to illustrate the emergence of planning by
        # using the time-integral of the expected free energy.
        # Even though the first observation (index 0) is the most preferred, the policy
        # that frequents this observation the most is actually not optimal, because that policy
        # terminates in a less preferred state by timestep 3.
        C = utils.to_obj_array(np.array([1.2, 1.0, 0.55]))

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

    def test_sample_policy(self):
        """
        Tests the action selection function where policies are sampled directly from posterior over policies `q_pi`
        """

        num_states = [3, 2]
        num_controls = [3, 2]

        policies = control.construct_policies(num_states, num_controls, policy_len=3)

        q_pi = utils.norm_dist(np.random.rand(len(policies)))
        best_policy = policies[np.argmax(q_pi)]

        selected_policy = control.sample_policy(q_pi, policies, num_controls)

        for factor_ii in range(len(num_controls)):
            self.assertEqual(selected_policy[factor_ii], best_policy[0,factor_ii])
        
        selected_policy_stochastic = control.sample_policy(q_pi, policies, num_controls, action_selection="stochastic",
                                                           alpha=1.0)
        self.assertEqual(selected_policy_stochastic.shape, selected_policy.shape)
        
    def test_update_posterior_policies_withE_vector(self):
        """
        Test update posterior policies in the case that there is a prior over policies
        """

        """ Construct an explicit example where policy 0 is preferred based on utility,
        but action 2 also gets a bump in probability because of prior over policies
        """
        num_obs = [3]
        num_states = [3]
        num_controls = [3]

        qs = utils.to_obj_array(utils.onehot(0, num_states[0]))
        A = utils.to_obj_array(np.eye(num_obs[0]))
        B = utils.construct_controllable_B(num_states, num_controls)
        
        C = utils.to_obj_array(np.array([1.5, 1.0, 1.0]))

        D = utils.to_obj_array(utils.onehot(0, num_states[0]))
        E = np.array([0.05, 0.05, 0.9])

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
            E = E,
            gamma=16.0
        )

        self.assertGreater(q_pi[0], q_pi[1])
        self.assertGreater(q_pi[2], q_pi[1])
    
    def test_stochastic_action_unidimensional_control(self):
        """
        Test stochastic action sampling in case that one of the control states is one-dimensional.
        Due to a call to probabilities.squeeze() in an earlier version of utils.sample(), this was throwing an
        error due to the inability to use np.random.multinomial on an array with undefined length (an 'unsized' array)
        """
        
        num_states = [2, 2]
        num_controls = [2, 1]
        policies = control.construct_policies(num_states, num_controls = num_controls, policy_len=1)
        q_pi = utils.norm_dist(np.random.rand(len(policies)))
        sampled_action = control.sample_action(q_pi, policies, num_controls, action_selection="stochastic")
        self.assertEqual(sampled_action[1], 0)

        sampled_action = control.sample_action(q_pi, policies, num_controls, action_selection="deterministic")
        self.assertEqual(sampled_action[1], 0)
    
    def test_deterministic_action_sampling_equal_value(self):
        """
        Test `deterministic` action sampling in the case that multiple actions have the same probability. 
        Desired behavior is that actions are randomly sampled from the subset of total actions that have the highest (but equal) probability.
        """

        num_states = [3]
        num_controls = [3]
        policies = control.construct_policies(num_states, num_controls = num_controls, policy_len=1)
        q_pi = np.array([0.4, 0.4, 0.2])

        seeds = [1923, 48323]

        sampled_action = control._sample_action_test(q_pi, policies, num_controls, action_selection="deterministic", seed=seeds[0])
        self.assertEqual(sampled_action[0], 0)

        sampled_action = control._sample_action_test(q_pi, policies, num_controls, action_selection="deterministic", seed=seeds[1])
        self.assertEqual(sampled_action[0], 1)
    
    def test_deterministic_policy_selection_equal_value(self):
        """
        Test `deterministic` action sampling in the case that multiple actions have the same probability. 
        Desired behavior is that actions are randomly sampled from the subset of total actions that have the highest (but equal) probability.
        """

        num_states = [3]
        num_controls = [3]
        policies = control.construct_policies(num_states, num_controls = num_controls, policy_len=1)
        q_pi = np.array([0.1, 0.45, 0.45])

        seeds = [1923, 48323]

        sampled_action = control._sample_policy_test(q_pi, policies, num_controls, action_selection="deterministic", seed=seeds[0])
        self.assertEqual(sampled_action[0], 1)

        sampled_action = control._sample_policy_test(q_pi, policies, num_controls, action_selection="deterministic", seed=seeds[1])
        self.assertEqual(sampled_action[0], 2)

if __name__ == "__main__":
    unittest.main()
