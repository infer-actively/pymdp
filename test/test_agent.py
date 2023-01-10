#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

import os
import unittest

import numpy as np
from copy import deepcopy

from pymdp.agent import Agent
from pymdp import utils, maths
from pymdp import inference, control, learning
from pymdp.default_models import generate_grid_world_transitions

class TestAgent(unittest.TestCase):
    
    def test_agent_init_without_control_fac_idx(self):
        """
        Initialize instance of the agent class and pass in a custom `control_fac_idx`
        """

        num_obs = [2, 4]
        num_states = [2, 2]
        num_controls = [2, 2]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = Agent(A=A, B=B)

        self.assertEqual(agent.num_controls[0], 2)
        self.assertEqual(agent.num_controls[1], 2)

        self.assertEqual([0, 1], agent.control_fac_idx)

    def test_reset_agent_VANILLA(self):
        """
        Ensure the `reset` method of Agent() using the new refactor is working as intended, 
        using the `VANILLA` argument to `inference_algo` 
        """

        num_obs = [2, 4]
        num_states = [2, 3]
        num_controls = [2, 3]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = Agent(A=A, B=B, inference_algo = "VANILLA")

        init_qs = utils.obj_array_uniform(agent.num_states)
        self.assertTrue(all( [ (agent.qs[f] == init_qs[f]).all() for f in range(agent.num_factors)]) )
        self.assertTrue(agent.curr_timestep == 0)

    def test_reset_agent_MMP_wBMA(self):
        """
        Ensure the `reset` method of Agent() using the new refactor is working as intended, 
        using the `MMP` argument to `inference_algo`, and `use_BMA` equal to True
        """

        num_obs = [2, 4]
        num_states = [2, 3]
        num_controls = [2, 3]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = Agent(A=A, B=B, inference_algo = "MMP", use_BMA=True)

        self.assertEqual(len(agent.latest_belief), agent.num_factors)
        self.assertTrue(all( [ (agent.latest_belief[f] == agent.D[f]).all() for f in range(agent.num_factors)]) )
        self.assertTrue(agent.curr_timestep == 0)
    
    def test_reset_agent_MMP_wPSP(self):
        """
        Ensure the `reset` method of Agent() using the new refactor is working as intended, 
        using the `MMP` argument to `inference_algo`, and `policy-separated prior` equal to True
        """

        num_obs = [2, 4]
        num_states = [2, 3]
        num_controls = [2, 3]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = Agent(A=A, B=B, inference_algo = "MMP", use_BMA = False, policy_sep_prior = True)

        self.assertEqual(len(agent.qs[0]) - agent.inference_horizon - 1, agent.policy_len)
    
    def test_agent_infer_states(self):
        """
        Test `infer_states` method of the Agent() class
        """

        ''' VANILLA method (fixed point iteration) with one hidden state factor and one observation modality '''
        num_obs = [5]
        num_states = [3]
        num_controls = [1]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = Agent(A=A, B=B, inference_algo = "VANILLA")

        o = tuple([np.random.randint(obs_dim) for obs_dim in num_obs])
        qs_out = agent.infer_states(o)

        qs_validation = inference.update_posterior_states(A, o, prior=agent.D)

        for f in range(len(num_states)):
            self.assertTrue(np.isclose(qs_validation[f], qs_out[f]).all())

        ''' VANILLA method (fixed point iteration) with multiple hidden state factors and multiple observation modalities '''
        num_obs = [2, 4]
        num_states = [2, 3]
        num_controls = [2, 3]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = Agent(A=A, B=B, inference_algo = "VANILLA")

        o = tuple([np.random.randint(obs_dim) for obs_dim in num_obs])
        qs_out = agent.infer_states(o)

        qs_validation = inference.update_posterior_states(A, o, prior=agent.D)

        for f in range(len(num_states)):
            self.assertTrue(np.isclose(qs_validation[f], qs_out[f]).all())

        ''' Marginal message passing inference with one hidden state factor and one observation modality '''
        num_obs = [5]
        num_states = [3]
        num_controls = [1]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = Agent(A=A, B=B, inference_algo = "MMP")

        o = tuple([np.random.randint(obs_dim) for obs_dim in num_obs])
        qs_pi_out = agent.infer_states(o)

        policies = control.construct_policies(num_states, num_controls, policy_len = 1)

        qs_pi_validation, _ = inference.update_posterior_states_full(A, B, [o], policies, prior = agent.D, policy_sep_prior = False)

        for p_idx in range(len(policies)):
            for f in range(len(num_states)):
                self.assertTrue(np.isclose(qs_pi_validation[p_idx][0][f], qs_pi_out[p_idx][0][f]).all())

        ''' Marginal message passing inference with multiple hidden state factors and multiple observation modalities '''
        num_obs = [2, 4]
        num_states = [2, 2]
        num_controls = [2, 2]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls) 

        planning_horizon = 3
        backwards_horizon = 1
        agent = Agent(A=A, B=B, inference_algo="MMP", policy_len=planning_horizon, inference_horizon=backwards_horizon)
        o = [0, 2]
        qs_pi_out = agent.infer_states(o)

        policies = control.construct_policies(num_states, num_controls, policy_len = planning_horizon)

        qs_pi_validation, _ = inference.update_posterior_states_full(A, B, [o], policies, prior = agent.D, policy_sep_prior = False)

        for p_idx in range(len(policies)):
            for t in range(planning_horizon+backwards_horizon):
                for f in range(len(num_states)):
                    self.assertTrue(np.isclose(qs_pi_validation[p_idx][t][f], qs_pi_out[p_idx][t][f]).all())

    def test_mmp_active_inference(self):
        """
        Tests to make sure whole active inference loop works (with various past and future
        inference/policy horizons).
        """

        num_obs = [3, 2]
        num_states = [4, 3]
        num_controls = [1, 3]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        C = utils.obj_array_zeros(num_obs)
        C[1][0] = 1.0  
        C[1][1] = -2.0  

        agent = Agent(A=A, B=B, C=C, control_fac_idx=[1], inference_algo="MMP", policy_len=2, inference_horizon=3)

        T = 10

        for t in range(T):

            o = [np.random.randint(num_ob) for num_ob in num_obs] # just randomly generate observations at each timestep, no generative process
            qx = agent.infer_states(o)
            agent.infer_policies()
            action = agent.sample_action()
        
        self.assertEqual(len(agent.prev_obs), T)
        self.assertEqual(len(agent.prev_actions), T)

    def test_agent_with_B_learning_vanilla(self):
        """ Unit test for updating prior Dirichlet parameters over transition model (pB) with the ``Agent`` class,
        in the case that you're using "vanilla" inference mode.
        """

        # 3 x 3, 2-dimensional grid world
        num_obs = [9]
        num_states = [9]
        num_controls = [4]

        A = utils.obj_array_zeros([ [num_obs[0], num_states[0]] ])
        A[0] = np.eye(num_obs[0])

        action_labels = ["LEFT", "DOWN", "RIGHT", "UP"]

        # flat transition prior
        pB = utils.obj_array_ones([[num_states[0], num_states[0], num_controls[0]]])
        B = utils.norm_dist_obj_arr(pB)
        
        # instantiate the agent
        learning_rate_pB = 2.0
        agent = Agent(A = A, B = B, pB = pB, inference_algo="VANILLA", save_belief_hist = True, action_selection="stochastic", lr_pB= learning_rate_pB)

        # get some true transition dynamics
        true_transition_matrix = generate_grid_world_transitions(action_labels, num_rows = 3, num_cols = 3)

        # time horizon
        T = 10
        next_state = 0

        for t in range(T):
            
            prev_state = next_state
            o = [prev_state] 
            qx = agent.infer_states(o)
            agent.infer_policies()
            agent.sample_action()

            # sample the next state given the true transition dynamics and the sampled action
            next_state = utils.sample(true_transition_matrix[:,prev_state,int(agent.action[0])])

            if t > 0:
                
                # compute the predicted update to the action-conditioned slice of qB
                predicted_update = agent.pB[0][:,:,int(agent.action[0])] + learning_rate_pB * maths.spm_cross(agent.qs_hist[-1][0], agent.qs_hist[-2][0])
                qB = agent.update_B(qs_prev = agent.qs_hist[-2]) # update qB using the agent function

                # check if the predicted update and the actual update are the same
                self.assertTrue(np.allclose(predicted_update, qB[0][:,:,int(agent.action[0])]))
    
    def test_agent_with_D_learning_vanilla(self):
        """
        Test updating prior Dirichlet parameters over initial hidden states (pD) with the ``Agent`` class,
        in the case that you're using "vanilla" inference mode.
        """

        num_obs = [2, 4]
        num_states = [2, 3]
        num_controls = [1, 1] # HMM mode
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        pD = utils.dirichlet_like(utils.random_single_categorical(num_states))

        # 1. Test that the updating works when `save_belief_hist` is True, and you don't need to pass in the beliefs about first hidden states
        agent = Agent(A=A, B=B, pD = pD, inference_algo = "VANILLA", save_belief_hist=True)

        T = 10

        qs_history = []

        for t in range(T):

            # get some random hidden state distribution
            p_states = utils.random_single_categorical(num_states)
            observation = [utils.sample(maths.spm_dot(A_g, p_states)) for A_g in A]

            qs_t = agent.infer_states(observation)

            qs_history.append(qs_t)

        pD_validation = learning.update_state_prior_dirichlet(agent.pD, qs_history[0], agent.lr_pD, factors = agent.factors_to_learn)

        pD_test = agent.update_D()

        for factor in range(len(num_states)):

            self.assertTrue(np.allclose(pD_test[factor], pD_validation[factor]))
        
        # 2. Test that the updating works when `save_belief_hist` is False, and you do have to pass in the beliefs about first hidden states
        agent = Agent(A=A, B=B, pD = pD, inference_algo = "VANILLA", save_belief_hist=False)

        T = 10

        qs_history = []

        for t in range(T):

            # get some random hidden state distribution
            p_states = utils.random_single_categorical(num_states)
            observation = [utils.sample(maths.spm_dot(A_g, p_states)) for A_g in A]

            qs_t = agent.infer_states(observation)

            qs_history.append(qs_t)

        pD_validation = learning.update_state_prior_dirichlet(agent.pD, qs_history[0], agent.lr_pD, factors = agent.factors_to_learn)

        pD_test = agent.update_D(qs_t0=qs_history[0])

        for factor in range(len(num_states)):

            self.assertTrue(np.allclose(pD_test[factor], pD_validation[factor]))
        
        # 3. Same as test #1, except with learning on only certain hidden state factors. Also passed in a different learning rate
        agent = Agent(A=A, B=B, pD = pD, lr_pD = 2.0, inference_algo = "VANILLA", save_belief_hist=True, factors_to_learn= [0])

        T = 10

        qs_history = []

        for t in range(T):

            # get some random hidden state distribution
            p_states = utils.random_single_categorical(num_states)
            observation = [utils.sample(maths.spm_dot(A_g, p_states)) for A_g in A]

            qs_t = agent.infer_states(observation)

            qs_history.append(qs_t)

        pD_validation = learning.update_state_prior_dirichlet(agent.pD, qs_history[0], agent.lr_pD, factors = agent.factors_to_learn)

        pD_test = agent.update_D()

        for factor in range(len(num_states)):

            self.assertTrue(np.allclose(pD_test[factor], pD_validation[factor]))
    
    def test_agent_with_D_learning_MMP(self):
        """
        Test updating prior Dirichlet parameters over initial hidden states (pD) with the agent class,
        in the case that you're using MMP inference and various combinations of Bayesian model averaging at the edge of the inference horizon vs. other possibilities
        """

        num_obs = [2]
        num_states = [2, 3, 3]
        num_controls = [2, 3, 1]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)
        pD = utils.dirichlet_like(utils.random_single_categorical(num_states))

        # 1. Using Bayesian model average over hidden states at the edge of the inference horizon
        agent = Agent(A=A, B=B, pD = pD, inference_algo = "MMP", use_BMA = True, policy_sep_prior = False, inference_horizon = 10, save_belief_hist = True)

        T = 10

        for t in range(T):

            # get some random hidden state distribution
            p_states = utils.random_single_categorical(num_states)
            observation = [utils.sample(maths.spm_dot(A_g, p_states)) for A_g in A]

            agent.infer_states(observation)
            agent.infer_policies()
            agent.sample_action()

        qs_t0 = agent.latest_belief

        pD_validation = learning.update_state_prior_dirichlet(pD, qs_t0, agent.lr_pD, factors = agent.factors_to_learn)

        pD_test = agent.update_D()

        for factor in range(len(num_states)):

            self.assertTrue(np.allclose(pD_test[factor], pD_validation[factor]))

        # 2. Using policy-conditioned prior over hidden states at the edge of the inference horizon
        agent = Agent(A=A, B=B, pD = pD, inference_algo = "MMP", use_BMA = False, policy_sep_prior = True, inference_horizon = 10, save_belief_hist = True)

        T = 10

        q_pi_hist = []
        for t in range(T):

            # get some random hidden state distribution
            p_states = utils.random_single_categorical(num_states)
            observation = [utils.sample(maths.spm_dot(A_g, p_states)) for A_g in A]

            qs_final = agent.infer_states(observation)
            q_pi, _ = agent.infer_policies()
            q_pi_hist.append(q_pi)
            agent.sample_action()

        # get beliefs about policies at the time at the beginning of the inference horizon
        
        qs_pi_t0 = utils.obj_array(len(agent.policies))
        for p_i in range(len(qs_pi_t0)):
            qs_pi_t0[p_i] = qs_final[p_i][0]

        qs_t0 = inference.average_states_over_policies(qs_pi_t0,q_pi_hist[0]) # beliefs about hidden states at the first timestep of the inference horizon
     
        pD_validation = learning.update_state_prior_dirichlet(pD, qs_t0, agent.lr_pD, factors = agent.factors_to_learn)

        pD_test = agent.update_D()

        for factor in range(len(num_states)):

            self.assertTrue(np.allclose(pD_test[factor], pD_validation[factor]))
    
    def test_agent_with_input_alpha(self):
        """
        Test for passing in `alpha` (action sampling precision parameter) as argument to `agent.Agent()` constructor.
        Test two cases to make sure alpha scaling is working properly, by comparing entropies of action marginals 
        after computing posterior over actions in cases where alpha is passed in as two different values to the Agent constructor.
        """

        num_obs = [2]
        num_states = [2]
        num_controls = [2]
        A = utils.to_obj_array(np.eye(num_obs[0], num_states[0]))
        B = utils.construct_controllable_B(num_states, num_controls)
        C = utils.to_obj_array(np.array([1.01, 1.0]))

        agent = Agent(A=A, B=B, C=C, alpha = 16.0, action_selection = "stochastic")
        agent.infer_states(([0]))
        agent.infer_policies()
        chosen_action, p_actions1 = agent._sample_action_test()

        agent = Agent(A=A, B=B, C=C, alpha = 0.0, action_selection = "stochastic")
        agent.infer_states(([0]))
        agent.infer_policies()
        chosen_action, p_actions2 = agent._sample_action_test()

        entropy_p_actions1 = -maths.spm_log_single(p_actions1[0]).dot(p_actions1[0])
        entropy_p_actions2 = -maths.spm_log_single(p_actions2[0]).dot(p_actions2[0])

        self.assertGreater(entropy_p_actions2, entropy_p_actions1)
    
    def test_agent_with_sampling_mode(self):
        """
        Test for passing in `sampling_mode` argument to `agent.Agent()` constructor, which determines whether you sample
        from posterior marginal over actions ("marginal", default) or posterior over policies ("full")
        """

        num_obs = [2]
        num_states = [2]
        num_controls = [2]
        A = utils.to_obj_array(np.eye(num_obs[0], num_states[0]))
        B = utils.construct_controllable_B(num_states, num_controls)
        C = utils.to_obj_array(np.array([1.01, 1.0]))

        agent = Agent(A=A, B=B, C=C, alpha = 16.0, sampling_mode = "full", action_selection = "stochastic")
        agent.infer_states(([0]))
        agent.infer_policies()
        chosen_action, p_policies = agent._sample_action_test()
        self.assertEqual(len(p_policies), len(agent.q_pi))

        agent = Agent(A=A, B=B, C=C, alpha = 16.0, sampling_mode = "full", action_selection = "deterministic")
        agent.infer_states(([0]))
        agent.infer_policies()
        chosen_action, p_policies = agent._sample_action_test()
        self.assertEqual(len(p_policies), len(agent.q_pi))

        agent = Agent(A=A, B=B, C=C, alpha = 16.0, sampling_mode = "marginal", action_selection = "stochastic")
        agent.infer_states(([0]))
        agent.infer_policies()
        chosen_action, p_actions = agent._sample_action_test()
        self.assertEqual(len(p_actions[0]), num_controls[0])

    def test_agent_with_stochastic_action_unidimensional_control(self):
        """
        Test stochastic action sampling in case that one of the control states is one-dimensional, within the agent
        method `sample_action()`.
        Due to a call to probabilities.squeeze() in an earlier version of utils.sample(), this was throwing an
        error due to the inability to use np.random.multinomial on an array with undefined length (an 'unsized' array)
        """
        
        num_obs = [2]
        num_states = [2, 2]
        num_controls = [2, 1]

        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = Agent(A=A, B=B, action_selection = "stochastic")
        agent.infer_policies()
        chosen_action = agent.sample_action()
        self.assertEqual(chosen_action[1], 0)

        agent = Agent(A=A, B=B, action_selection = "deterministic")
        agent.infer_policies()
        chosen_action = agent.sample_action()
        self.assertEqual(chosen_action[1], 0)
    
    def test_agent_distributional_obs(self):

        ''' VANILLA method (fixed point iteration) with one hidden state factor and one observation modality '''
        num_obs = [5]
        num_states = [3]
        num_controls = [1]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = Agent(A=A, B=B, inference_algo = "VANILLA")

        p_o = utils.obj_array_zeros(num_obs) # use a distributional observation
        # @NOTE: `utils.obj_array_from_list` will make a nested list of object arrays if you only put in a list with one vector!!! Makes me think we should remove utils.obj_array_from_list potentially
        p_o[0] = A[0][:,0]
        qs_out = agent.infer_states(p_o, distr_obs=True)

        qs_validation = inference.update_posterior_states(A, p_o, prior=agent.D)

        for f in range(len(num_states)):
            self.assertTrue(np.isclose(qs_validation[f], qs_out[f]).all())

        ''' VANILLA method (fixed point iteration) with multiple hidden state factors and multiple observation modalities '''
        num_obs = [2, 4]
        num_states = [2, 3]
        num_controls = [2, 3]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = Agent(A=A, B=B, inference_algo = "VANILLA")

        p_o = utils.obj_array_from_list([A[0][:,0,0], A[1][:,1,1]]) # use a distributional observation
        qs_out = agent.infer_states(p_o, distr_obs=True)

        qs_validation = inference.update_posterior_states(A, p_o, prior=agent.D)

        for f in range(len(num_states)):
            self.assertTrue(np.isclose(qs_validation[f], qs_out[f]).all())

        ''' Marginal message passing inference with one hidden state factor and one observation modality '''
        num_obs = [5]
        num_states = [3]
        num_controls = [1]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = Agent(A=A, B=B, inference_algo = "MMP")

        p_o = utils.obj_array_zeros(num_obs) # use a distributional observation
        # @NOTE: `utils.obj_array_from_list` will make a nested list of object arrays if you only put in a list with one vector!!! Makes me think we should remove utils.obj_array_from_list potentially
        p_o[0] = A[0][:,0]
        qs_pi_out = agent.infer_states(p_o, distr_obs=True)

        policies = control.construct_policies(num_states, num_controls, policy_len = 1)

        qs_pi_validation, _ = inference.update_posterior_states_full(A, B, [p_o], policies, prior = agent.D, policy_sep_prior = False)

        for p_idx in range(len(policies)):
            for f in range(len(num_states)):
                self.assertTrue(np.isclose(qs_pi_validation[p_idx][0][f], qs_pi_out[p_idx][0][f]).all())
        
        ''' Marginal message passing inference with multiple hidden state factors and multiple observation modalities '''
        num_obs = [2, 4]
        num_states = [2, 2]
        num_controls = [2, 2]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls) 

        planning_horizon = 3
        backwards_horizon = 1
        agent = Agent(A=A, B=B, inference_algo="MMP", policy_len=planning_horizon, inference_horizon=backwards_horizon)
        p_o = utils.obj_array_from_list([A[0][:,0,0], A[1][:,1,1]]) # use a distributional observation
        qs_pi_out = agent.infer_states(p_o, distr_obs=True)

        policies = control.construct_policies(num_states, num_controls, policy_len = planning_horizon)

        qs_pi_validation, _ = inference.update_posterior_states_full(A, B, [p_o], policies, prior = agent.D, policy_sep_prior = False)

        for p_idx in range(len(policies)):
            for t in range(planning_horizon+backwards_horizon):
                for f in range(len(num_states)):
                    self.assertTrue(np.isclose(qs_pi_validation[p_idx][t][f], qs_pi_out[p_idx][t][f]).all())


        


if __name__ == "__main__":
    unittest.main()

    