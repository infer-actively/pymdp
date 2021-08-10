
import unittest

import numpy as np
from pymdp.core import utils, maths, learning

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

if __name__ == "__main__":
    unittest.main()
