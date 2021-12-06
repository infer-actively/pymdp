import unittest

import numpy as np
from pymdp import utils, maths, learning

from copy import deepcopy

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

        pA_updated = learning.update_obs_likelihood_dirichlet(
            pA, A, observation, qs, lr=l_rate, modalities="all")
        validation_pA = pA[0] + l_rate * maths.spm_cross(utils.onehot(observation, num_obs[0]), qs)
        self.assertTrue(np.all(pA_updated[0] == validation_pA))

        '''multiple observation modalities'''
        num_obs = [3, 4]
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])
        
        observation = [utils.sample(maths.spm_dot(A_m, qs)) for A_m in A]

        pA_updated = learning.update_obs_likelihood_dirichlet(
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

        pA_updated = learning.update_obs_likelihood_dirichlet(
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

        pA_updated = learning.update_obs_likelihood_dirichlet(
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
        pA_updated = learning.update_obs_likelihood_dirichlet(
            pA, A, observation, qs, lr=l_rate, modalities="all")
        update = maths.spm_cross(utils.onehot(observation[0], num_obs[0]), qs)
        validation_pA = pA[0] + l_rate * update
        self.assertTrue(np.all(pA_updated[0] == validation_pA))

        # multiple observation modalities
        num_obs = [3, 4]
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])
        observation = [utils.sample(maths.spm_dot(A_m, qs)) for A_m in A]
        pA_updated = learning.update_obs_likelihood_dirichlet(
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
        pA_updated = learning.update_obs_likelihood_dirichlet(
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
        pA_updated = learning.update_obs_likelihood_dirichlet(
            pA, A, observation, qs, lr=l_rate, modalities=modalities_to_update)

        for modality, obs_dim in enumerate(num_obs):
            if modality in modalities_to_update:
                update = maths.spm_cross(utils.onehot(observation[modality], obs_dim), qs)
                validation_pA = pA[modality] + l_rate * update
            else:
                validation_pA = pA[modality]
            self.assertTrue(np.all(pA_updated[modality] == validation_pA))
    
    def test_update_pA_diff_observation_formats(self):
        """
        Test for updating prior Dirichlet parameters over sensory likelihood (pA)
        in the case that observation is stored in various formats
        """

        num_states = [2, 6]
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        # multiple observation modalities
        num_obs = [3, 4, 5]
        modalities_to_update = "all"
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.obj_array_ones([A_m.shape for A_m in A])

        observation_list = [0, 3, 2]

        pA_updated_1 = learning.update_obs_likelihood_dirichlet(
            pA, A, observation_list, qs, lr=l_rate, modalities=modalities_to_update)

        observation_tuple = (0, 3, 2)

        pA_updated_2 = learning.update_obs_likelihood_dirichlet(
            pA, A, observation_tuple, qs, lr=l_rate, modalities=modalities_to_update)
        
        observation_obj_array = utils.process_observation((0, 3, 2), len(num_obs), num_obs)

        pA_updated_3 = learning.update_obs_likelihood_dirichlet(
            pA, A, observation_obj_array, qs, lr=l_rate, modalities=modalities_to_update)

        for modality, _ in enumerate(num_obs):
            
            self.assertTrue(np.allclose(pA_updated_1[modality], pA_updated_2[modality]))
            self.assertTrue(np.allclose(pA_updated_1[modality], pA_updated_3[modality]))
       
        # now do the same for case of single modality

        num_states = [2, 6]
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        num_obs = [3]
        modalities_to_update = "all"
        A = utils.random_A_matrix(num_obs, num_states)
        pA = utils.dirichlet_like(A)

        observation_int = 2

        pA_updated_1 = learning.update_obs_likelihood_dirichlet(
            pA, A, observation_int, qs, lr=l_rate, modalities=modalities_to_update)

        observation_onehot = utils.onehot(2, num_obs[0])

        pA_updated_2 = learning.update_obs_likelihood_dirichlet(
            pA, A, observation_onehot, qs, lr=l_rate, modalities=modalities_to_update)
        
        self.assertTrue(np.allclose(pA_updated_1[0], pA_updated_2[0]))


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

        pB_updated = learning.update_state_likelihood_dirichlet(
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

        pB_updated = learning.update_state_likelihood_dirichlet(
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

        pB_updated = learning.update_state_likelihood_dirichlet(
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

        pB_updated = learning.update_state_likelihood_dirichlet(
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

        pB_updated = learning.update_state_likelihood_dirichlet(
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

        pB_updated = learning.update_state_likelihood_dirichlet(
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

        pB_updated = learning.update_state_likelihood_dirichlet(
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

        pB_updated = learning.update_state_likelihood_dirichlet(
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

        pB_updated = learning.update_state_likelihood_dirichlet(
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
    
    def test_update_pD(self):
        """
        Test updating prior Dirichlet parameters over initial hidden states (pD). 
        Tests different cases
        1. Multiple vs. single hidden state factor
        2. One factor vs. several factors vs. all factors learned
        """

        # 1. Single hidden state factor
        num_states = [3]

        pD = utils.dirichlet_like(utils.random_single_categorical(num_states), scale = 0.5)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        pD_test= learning.update_state_prior_dirichlet(
                        pD, qs, lr=l_rate, factors="all"
                    )
        
        for factor in range(len(num_states)):
            pD_validation_f = pD[factor].copy()
            idx = pD_validation_f > 0
            pD_validation_f[idx] += l_rate * qs[factor][idx]
            self.assertTrue(np.allclose(pD_test[factor], pD_validation_f))

        # 2. Multiple hidden state factors
        num_states = [3, 4, 5]

        pD = utils.dirichlet_like(utils.random_single_categorical(num_states), scale = 0.5)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        pD_test= learning.update_state_prior_dirichlet(
                        pD, qs, lr=l_rate, factors="all"
                    )
        
        for factor in range(len(num_states)):
            pD_validation_f = pD[factor].copy()
            idx = pD_validation_f > 0
            pD_validation_f[idx] += l_rate * qs[factor][idx]
            self.assertTrue(np.allclose(pD_test[factor], pD_validation_f))
        
        # 3. Multiple hidden state factors, only some learned
        num_states = [3, 4, 5]

        factors_to_learn = [0, 2]

        pD = utils.dirichlet_like(utils.random_single_categorical(num_states), scale = 0.5)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        pD_test= learning.update_state_prior_dirichlet(
                        pD, qs, lr=l_rate, factors=factors_to_learn
                    )
        
        pD_validation = deepcopy(pD)

        for factor in range(len(num_states)):

            if factor in factors_to_learn:
                idx = pD_validation[factor] > 0
                pD_validation[factor][idx] += l_rate * qs[factor][idx]

            self.assertTrue(np.allclose(pD_test[factor], pD_validation[factor]))
    
    def test_prune_prior(self):
        """
        Test removing hidden state factor levels and/or observation levels from the priors vectors
        of a generative model, using the `_prune_prior` function of the `learning` module
        """

        """ Test 1a. Testing `_prune_prior()` in case of a single hidden state factor/modality """

        num_levels_total = 4 # this could either be 4 hidden state levels, or 4 observation levels
        test_prior = utils.random_single_categorical([num_levels_total])[0]

        levels_to_remove = [2]

        reduced_prior = learning._prune_prior(test_prior, levels_to_remove)

        self.assertTrue(len(reduced_prior) == (num_levels_total - len(levels_to_remove)))
        self.assertTrue(utils.is_normalized(reduced_prior))

        """ Test 1b. Testing `_prune_prior()` in case of multiple hidden state factors/modalities """

        num_levels_total = [4, 5] # this could either be 4 hidden state levels, or 4 observation levels
        test_prior = utils.random_single_categorical(num_levels_total)

        levels_to_remove = [ [2, 3], []]

        reduced_prior = learning._prune_prior(test_prior, levels_to_remove)

        for f, ns in enumerate(num_levels_total):
            self.assertTrue(len(reduced_prior[f]) == (ns - len(levels_to_remove[f])))
            self.assertTrue(utils.is_normalized(reduced_prior[f]))
        
        """ Test 1c. Testing `_prune_prior()` in case of multiple hidden state factors/modalities, and where you're removing all the levels of a particular factor """

        num_levels_total = [4, 5] # this could either be 4 hidden state levels, or 4 observation levels
        test_prior = utils.random_single_categorical(num_levels_total)

        levels_to_remove = [ [2, 3], list(range(5))]

        reduced_prior = learning._prune_prior(test_prior, levels_to_remove)

        self.assertTrue(len(reduced_prior[0]) == (num_levels_total[0] - len(levels_to_remove[0])))
        self.assertTrue(utils.is_normalized(reduced_prior[0]))
    
        self.assertTrue(len(reduced_prior) == (len(levels_to_remove)-1))
    
    def test_prune_likelihoods(self):
        """
        Test removing hidden state factor levels and/or observation levels from the likelihood arrays 
        of a generative model, using the `_prune_A` and `_prune_B` functions of the `learning` module
        """

        """ Test 1a. Testing `_prune_A()` in case of a single hidden state factor/modality """

        A = utils.random_A_matrix([5], [4])[0]

        obs_levels_to_prune = [2, 3]
        state_levels_to_prune = [1, 3]

        A_pruned = learning._prune_A(A, obs_levels_to_prune, state_levels_to_prune)

        expected_shape = (A.shape[0] - len(obs_levels_to_prune), A.shape[1] - len(state_levels_to_prune))
        self.assertTrue(A_pruned.shape == expected_shape)
        self.assertTrue(utils.is_normalized(A_pruned))

        """ Test 1b. Testing `_prune_A()` in case of a single hidden state factor/modality, where hidden state levels aren't pruned at all """

        A = utils.random_A_matrix([5], [4])[0]

        obs_levels_to_prune = [2, 3]
        state_levels_to_prune = []

        A_pruned = learning._prune_A(A, obs_levels_to_prune, state_levels_to_prune)

        expected_shape = (A.shape[0] - len(obs_levels_to_prune), A.shape[1] - len(state_levels_to_prune))
        self.assertTrue(A_pruned.shape == expected_shape)
        self.assertTrue(utils.is_normalized(A_pruned))

        """ Test 1c. Testing `_prune_A()` in case of a single hidden state factor/modality, where observation levels aren't pruned at all """
        
        A = utils.random_A_matrix([5], [4])[0]

        obs_levels_to_prune = []
        state_levels_to_prune = [2,3]

        A_pruned = learning._prune_A(A, obs_levels_to_prune, state_levels_to_prune)

        expected_shape = (A.shape[0] - len(obs_levels_to_prune), A.shape[1] - len(state_levels_to_prune))
        self.assertTrue(A_pruned.shape == expected_shape)
        self.assertTrue(utils.is_normalized(A_pruned))

        """ Test 1d. Testing `_prune_A()` in case of a multiple hidden state factors/modalities """

        num_obs = [3, 4, 5]
        num_states = [2, 10, 4]
        A = utils.random_A_matrix(num_obs, num_states)

        obs_levels_to_prune = [ [0, 2], [], [1, 2, 3]]
        state_levels_to_prune = [[], [5,6,7,8], [1]]

        A_pruned = learning._prune_A(A, obs_levels_to_prune, state_levels_to_prune)

        expected_lagging_dimensions = []
        for f, ns in enumerate(num_states):
            expected_lagging_dimensions.append(ns - len(state_levels_to_prune[f]))
        for m, no in enumerate(num_obs):
            expected_shape = (no - len(obs_levels_to_prune[m]),) + tuple(expected_lagging_dimensions)
            self.assertTrue(A_pruned[m].shape == expected_shape)
            self.assertTrue(utils.is_normalized(A_pruned[m]))
        
        """ Test 2a. Testing `_prune_B()` in case of a single hidden state factor / control state factor """

        B = utils.random_B_matrix([4], [3])[0]

        state_levels_to_prune = [1, 3]
        action_levels_to_prune = [0, 1]

        B_pruned = learning._prune_B(B, state_levels_to_prune, action_levels_to_prune)

        expected_shape = (B.shape[0] - len(state_levels_to_prune), B.shape[1] - len(state_levels_to_prune), B.shape[2] - len(action_levels_to_prune))
        self.assertTrue(B_pruned.shape == expected_shape)
        self.assertTrue(utils.is_normalized(B_pruned))

        """ Test 2b. Testing `_prune_B()` in case of a single hidden state factor, where control state levels aren't pruned at all """

        B = utils.random_B_matrix([4], [3])[0]

        state_levels_to_prune = [1, 3]
        action_levels_to_prune = []

        B_pruned = learning._prune_B(B, state_levels_to_prune, action_levels_to_prune)

        expected_shape = (B.shape[0] - len(state_levels_to_prune), B.shape[1] - len(state_levels_to_prune), B.shape[2] - len(action_levels_to_prune))
        self.assertTrue(B_pruned.shape == expected_shape)
        self.assertTrue(utils.is_normalized(B_pruned))

        """ Test 1c. Testing `_prune_B()` in case of a single hidden state factor, where hidden state levels aren't pruned at all """
        
        B = utils.random_B_matrix([4], [3])[0]

        state_levels_to_prune = []
        action_levels_to_prune = [0]

        B_pruned = learning._prune_B(B, state_levels_to_prune, action_levels_to_prune)

        expected_shape = (B.shape[0] - len(state_levels_to_prune), B.shape[1] - len(state_levels_to_prune), B.shape[2] - len(action_levels_to_prune))
        self.assertTrue(B_pruned.shape == expected_shape)
        self.assertTrue(utils.is_normalized(B_pruned))
        
        """ Test 1d. Testing `_prune_B()` in case of a multiple hidden state factors """

        num_states = [2, 10, 4]
        num_controls = [5, 3, 4]
        B = utils.random_B_matrix(num_states, num_controls)

        state_levels_to_prune = [ [0, 1], [], [1, 2, 3]]
        action_levels_to_prune = [[], [0, 1], [1]]

        B_pruned = learning._prune_B(B, state_levels_to_prune, action_levels_to_prune)

        for f, ns in enumerate(num_states):
            expected_shape = (ns - len(state_levels_to_prune[f]), ns - len(state_levels_to_prune[f]), num_controls[f] - len(action_levels_to_prune[f]))
            self.assertTrue(B_pruned[f].shape == expected_shape)
            self.assertTrue(utils.is_normalized(B_pruned[f]))

    

        
if __name__ == "__main__":
    unittest.main()
