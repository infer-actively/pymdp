
import unittest

import numpy as np
from pymdp import utils, maths, learning

class TestLearning(unittest.TestCase):

    def test_update_pA_single_factor_all(self):
        """
        Test for updating prior Dirichlet parameters over sensory likelihood (pA)
        in the case that all observation modalities are updated and the generative model 
        has a single hidden state factor
        """
        num_states = [3]
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        '''single observation modality'''
        num_obs = [4]
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])
        
        observation = utils.sample(maths.spm_dot(A[0], qs))

        pA_updated = learning.update_likelihood_dirichlet(
            pA, A, observation, qs, lr=l_rate, modalities="all")
        validation_pA = pA[0] + l_rate * maths.spm_cross(utils.onehot(observation, num_obs[0]), qs)
        self.assertTrue(np.all(pA_updated[0] == validation_pA))

        '''multiple observation modalities'''
        num_obs = [3, 4]
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])
        
        observation = [utils.sample(maths.spm_dot(A_m, qs)) for A_m in A]

        pA_updated = learning.update_likelihood_dirichlet(
            pA, A, observation, qs, lr=l_rate, modalities="all")

        for modality, obs_dim in enumerate(num_obs):
            update = maths.spm_cross(utils.onehot(observation[modality], obs_dim), qs)
            validation_pA = pA[modality] + l_rate * update
            self.assertTrue(np.all(pA_updated[modality] == validation_pA))

    def test_update_pA_single_factor_one_modality(self):
        """
        Test for updating prior Dirichlet parameters over sensory likelihood (pA)
        in the case that ONE observation modalities is updated and the generative model 
        has a single hidden state factor
        """

        num_states = [3]
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        '''multiple observation modalities'''
        num_obs = [3, 4]

        modality_to_update = [np.random.randint(len(num_obs))]
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])
        
        observation = [utils.sample(maths.spm_dot(A_m, qs)) for A_m in A]

        pA_updated = learning.update_likelihood_dirichlet(
            pA, A, observation, qs, lr=l_rate, modalities=modality_to_update)

        for modality, obs_dim in enumerate(num_obs):
            if modality in modality_to_update:
                update = maths.spm_cross(utils.onehot(observation[modality], obs_dim), qs)
                validation_pA = pA[modality] + l_rate * update
            else:
                validation_pA = pA[modality]
            self.assertTrue(np.all(pA_updated[modality] == validation_pA))
    
    def test_update_pA_single_factor_some_modalities(self):
        """
        Test for updating prior Dirichlet parameters over sensory likelihood (pA)
        in the case that some observation modalities are updated and the generative model 
        has a single hidden state factor
        """
        num_states = [3]
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0
        
        '''multiple observation modalities'''
        num_obs = [3, 4, 5]
        modalities_to_update = [0, 2]
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])

        observation = [utils.sample(maths.spm_dot(A_m, qs)) for A_m in A]

        pA_updated = learning.update_likelihood_dirichlet(
            pA, A, observation, qs, lr=l_rate, modalities=modalities_to_update)

        for modality, obs_dim in enumerate(num_obs):
            if modality in modalities_to_update:
                update = maths.spm_cross(utils.onehot(observation[modality], obs_dim), qs)
                validation_pA = pA[modality] + l_rate * update
            else:
                validation_pA = pA[modality]
            self.assertTrue(np.all(pA_updated[modality] == validation_pA))
    
    def test_update_pA_multi_factor_all(self):
        """
        Test for updating prior Dirichlet parameters over sensory likelihood (pA)
        in the case that all observation modalities are updated and the generative model 
        has multiple hidden state factors
        """
        num_states = [2, 6]
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        # single observation modality
        num_obs = [4]
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])
        observation = [utils.sample(maths.spm_dot(A_m, qs)) for A_m in A]
        pA_updated = learning.update_likelihood_dirichlet(
            pA, A, observation, qs, lr=l_rate, modalities="all")
        update = maths.spm_cross(utils.onehot(observation[0], num_obs[0]), qs)
        validation_pA = pA[0] + l_rate * update
        self.assertTrue(np.all(pA_updated[0] == validation_pA))

        # multiple observation modalities
        num_obs = [3, 4]
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])
        observation = [utils.sample(maths.spm_dot(A_m, qs)) for A_m in A]
        pA_updated = learning.update_likelihood_dirichlet(
            pA, A, observation, qs, lr=l_rate, modalities="all")
        for modality, obs_dim in enumerate(num_obs):
            update = maths.spm_cross(utils.onehot(observation[modality], obs_dim), qs)
            validation_pA = pA[modality] + l_rate * update
            self.assertTrue(np.all(pA_updated[modality] == validation_pA))
    
    def test_update_pA_multi_factor_one_modality(self):
        """
        Test for updating prior Dirichlet parameters over sensory likelihood (pA)
        in the case that ONE observation modalities is updated and the generative model 
        has multiple hidden state factors
        """
        num_states = [2, 6]
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        # multiple observation modalities
        num_obs = [3, 4]
        modality_to_update = [np.random.randint(len(num_obs))]
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])
        observation = [utils.sample(maths.spm_dot(A_m, qs)) for A_m in A]
        pA_updated = learning.update_likelihood_dirichlet(
            pA, A, observation, qs, lr=l_rate, modalities=modality_to_update)

        for modality, obs_dim in enumerate(num_obs):
            if modality in modality_to_update:
                update = maths.spm_cross(utils.onehot(observation[modality], obs_dim), qs)
                validation_pA = pA[modality] + l_rate * update
            else:
                validation_pA = pA[modality]
            self.assertTrue(np.all(pA_updated[modality] == validation_pA))
    
    def test_update_pA_multi_factor_some_modalities(self):
        """
        Test for updating prior Dirichlet parameters over sensory likelihood (pA)
        in the case that SOME observation modalities are updated and the generative model 
        has multiple hidden state factors
        """
        num_states = [2, 6]
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        # multiple observation modalities
        num_obs = [3, 4, 5]
        modalities_to_update = [0, 2]
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])
        observation = [utils.sample(maths.spm_dot(A_m, qs)) for A_m in A]
        pA_updated = learning.update_likelihood_dirichlet(
            pA, A, observation, qs, lr=l_rate, modalities=modalities_to_update)

        for modality, obs_dim in enumerate(num_obs):
            if modality in modalities_to_update:
                update = maths.spm_cross(utils.onehot(observation[modality], obs_dim), qs)
                validation_pA = pA[modality] + l_rate * update
            else:
                validation_pA = pA[modality]
            self.assertTrue(np.all(pA_updated[modality] == validation_pA))
    
    def test_update_pB_single_factor_no_actions(self):
        """
        Test for updating prior Dirichlet parameters over transition likelihood (pB)
        in the case that the one and only hidden state factor is updated, and there 
        are no actions.
        """

        num_states = [3]
        num_controls = [1]  # this is how we encode the fact that there aren't any actions
        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated = learning.update_transition_dirichlet(
            pB, B, action, qs, qs_prev, lr=l_rate, factors="all"
        )

        validation_pB = utils.obj_array_ones([B_f.shape for B_f in B])
        validation_pB[0][:, :, 0] += (
            l_rate * maths.spm_cross(qs[0], qs_prev[0]) * (B[0][:, :, action[0]] > 0)
        )
        self.assertTrue(np.all(pB_updated[0] == validation_pB[0]))
    
    def test_update_pB_single_factor_with_actions(self):
        """
        Test for updating prior Dirichlet parameters over transition likelihood (pB)
        in the case that the one and only hidden state factor is updated, and there 
        are actions.
        """

        num_states = [3]
        num_controls = [3]
        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated = learning.update_transition_dirichlet(
            pB, B, action, qs, qs_prev, lr=l_rate, factors="all"
        )

        validation_pB = utils.obj_array_ones([B_f.shape for B_f in B])
        validation_pB[0][:, :, action[0]] += (
            l_rate * maths.spm_cross(qs[0], qs_prev[0]) * (B[0][:, :, action[0]] > 0)
        )
        self.assertTrue(np.all(pB_updated[0] == validation_pB[0]))

    def test_update_pB_multi_factor_no_actions_all_factors(self):
        """
        Test for updating prior Dirichlet parameters over transition likelihood (pB)
        in the case that there are mulitple hidden state factors, and there 
        are no actions. All factors are updated.
        """

        num_states = [3, 4]
        num_controls = [1, 1]
        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated = learning.update_transition_dirichlet(
            pB, B, action, qs, qs_prev, lr=l_rate, factors="all"
        )

        validation_pB = utils.obj_array_ones([B_f.shape for B_f in B])

        for factor, action_i in enumerate(action):
            validation_pB[factor][:, :, action_i] += (
                l_rate
                * maths.spm_cross(qs[factor], qs_prev[factor])
                * (B[factor][:, :, action_i] > 0)
            )
            self.assertTrue(np.all(pB_updated[factor] == validation_pB[factor]))
    
    def test_update_pB_multi_factor_no_actions_one_factor(self):
        """
        Test for updating prior Dirichlet parameters over transition likelihood (pB)
        in the case that there are mulitple hidden state factors, and there 
        are no actions. One factor is updated
        """

        num_states = [3, 4]
        num_controls = [1, 1]
        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        factors_to_update = [np.random.randint(len(num_states))]

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated = learning.update_transition_dirichlet(
            pB, B, action, qs, qs_prev, lr=l_rate, factors=factors_to_update
        )

        validation_pB = utils.obj_array_ones([B_f.shape for B_f in B])

        for factor, action_i in enumerate(action):

            if factor in factors_to_update:
                validation_pB[factor][:, :, action_i] += (
                    l_rate
                    * maths.spm_cross(qs[factor], qs_prev[factor])
                    * (B[factor][:, :, action_i] > 0)
                )
            self.assertTrue(np.all(pB_updated[factor] == validation_pB[factor]))
    
    def test_update_pB_multi_factor_no_actions_some_factors(self):
        """
        Test for updating prior Dirichlet parameters over transition likelihood (pB)
        in the case that there are mulitple hidden state factors, and there 
        are no actions. Some factors are updated.
        """

        num_states = [3, 4, 5]
        num_controls = [1, 1, 1]
        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        factors_to_update = [0, 2]

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated = learning.update_transition_dirichlet(
            pB, B, action, qs, qs_prev, lr=l_rate, factors=factors_to_update
        )

        validation_pB = utils.obj_array_ones([B_f.shape for B_f in B])

        for factor, action_i in enumerate(action):

            if factor in factors_to_update:
                validation_pB[factor][:, :, action_i] += (
                    l_rate
                    * maths.spm_cross(qs[factor], qs_prev[factor])
                    * (B[factor][:, :, action_i] > 0)
                )
            self.assertTrue(np.all(pB_updated[factor] == validation_pB[factor]))
    
    def test_update_pB_multi_factor_with_actions_all_factors(self):
        """
        Test for updating prior Dirichlet parameters over transition likelihood (pB)
        in the case that there are mulitple hidden state factors, and there 
        are actions. All factors are updated
        """

        num_states = [3, 4, 5]
        num_controls = [2, 3, 4]
        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated = learning.update_transition_dirichlet(
            pB, B, action, qs, qs_prev, lr=l_rate, factors="all"
        )

        validation_pB = utils.obj_array_ones([B_f.shape for B_f in B])

        for factor, action_i in enumerate(action):

            validation_pB[factor][:, :, action_i] += (
                l_rate
                * maths.spm_cross(qs[factor], qs_prev[factor])
                * (B[factor][:, :, action_i] > 0)
            )
            self.assertTrue(np.all(pB_updated[factor] == validation_pB[factor]))
    
    def test_update_pB_multi_factor_with_actions_one_factor(self):
        """
        Test for updating prior Dirichlet parameters over transition likelihood (pB)
        in the case that there are mulitple hidden state factors, and there 
        are actions. One factor is updated
        """

        num_states = [3, 4, 5]
        num_controls = [2, 3, 4]
        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        factors_to_update = [np.random.randint(len(num_states))]

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated = learning.update_transition_dirichlet(
            pB, B, action, qs, qs_prev, lr=l_rate, factors=factors_to_update
        )

        validation_pB = utils.obj_array_ones([B_f.shape for B_f in B])

        for factor, action_i in enumerate(action):
            
            if factor in factors_to_update:
                validation_pB[factor][:, :, action_i] += (
                    l_rate
                    * maths.spm_cross(qs[factor], qs_prev[factor])
                    * (B[factor][:, :, action_i] > 0)
                )
            self.assertTrue(np.all(pB_updated[factor] == validation_pB[factor]))
    
    def test_update_pB_multi_factor_with_actions_some_factors(self):
        """
        Test for updating prior Dirichlet parameters over transition likelihood (pB)
        in the case that there are mulitple hidden state factors, and there 
        are actions. Some factors are updated
        """

        num_states = [3, 4, 5]
        num_controls = [2, 3, 4]
        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        factors_to_update = [0, 1]

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated = learning.update_transition_dirichlet(
            pB, B, action, qs, qs_prev, lr=l_rate, factors=factors_to_update
        )

        validation_pB = utils.obj_array_ones([B_f.shape for B_f in B])

        for factor, action_i in enumerate(action):
            
            if factor in factors_to_update:
                validation_pB[factor][:, :, action_i] += (
                    l_rate
                    * maths.spm_cross(qs[factor], qs_prev[factor])
                    * (B[factor][:, :, action_i] > 0)
                )
            self.assertTrue(np.all(pB_updated[factor] == validation_pB[factor]))

    def test_update_pB_multi_factor_some_controllable_some_factors(self):
        """
        Test for updating prior Dirichlet parameters over transition likelihood (pB)
        in the case that there are mulitple hidden state factors, some of which 
        are controllable. Some factors are updated.
        """

        num_states = [3, 4, 5]
        num_controls = [2, 1, 1]
        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        factors_to_update = [0, 1]

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated = learning.update_transition_dirichlet(
            pB, B, action, qs, qs_prev, lr=l_rate, factors=factors_to_update
        )

        validation_pB = utils.obj_array_ones([B_f.shape for B_f in B])

        for factor, action_i in enumerate(action):
            
            if factor in factors_to_update:
                validation_pB[factor][:, :, action_i] += (
                    l_rate
                    * maths.spm_cross(qs[factor], qs_prev[factor])
                    * (B[factor][:, :, action_i] > 0)
                )
            self.assertTrue(np.all(pB_updated[factor] == validation_pB[factor]))
        
if __name__ == "__main__":
    unittest.main()
