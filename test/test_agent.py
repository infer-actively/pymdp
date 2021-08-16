#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

import os
import unittest

import numpy as np

from pymdp.agent import Agent
from pymdp import utils
from pymdp import inference, control

class TestAgent(unittest.TestCase):
   
    def test_agent_init_with_control_fac_idx(self):
        """
        Initialize instance of the agent class and pass in a custom `control_fac_idx`
        """

        num_obs = [2, 4]
        num_states = [2, 2]
        num_controls = [2, 2]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = Agent(A=A, B=B, control_fac_idx=[1])

        self.assertEqual(agent.num_controls[0], 1)
        self.assertEqual(agent.num_controls[1], 2)
    
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

        ''' Marginal message passing inference with multiple hidden state factors and multiple observation modalities '''
        num_obs = [5]
        num_states = [3]
        num_controls = [1]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_controls)

        agent = Agent(A=A, B=B, inference_algo = "MMP")

        o = tuple([np.random.randint(obs_dim) for obs_dim in num_obs])
        qs_pi_out = agent.infer_states(o)

        policies = control.construct_policies(num_states, num_controls, policy_len = 1)

        qs_pi_validation, _ = inference.update_posterior_states_v2(A, B, [o], policies, prior = agent.D, policy_sep_prior = False)

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

        qs_pi_validation, _ = inference.update_posterior_states_v2(A, B, [o], policies, prior = agent.D, policy_sep_prior = False)

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

if __name__ == "__main__":
    unittest.main()

    