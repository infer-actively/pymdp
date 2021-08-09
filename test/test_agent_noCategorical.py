#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

import os
import unittest

import numpy as np

from pymdp.agent import Agent
from pymdp.core import utils
from pymdp.core import inference

class TestAgent_noCat(unittest.TestCase):
   
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

        ''' MMP method with one hidden state factor and one observation modality '''


if __name__ == "__main__":
    unittest.main()

    