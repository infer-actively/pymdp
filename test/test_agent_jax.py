#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Dimitrije Markovic, Conor Heins
"""

import unittest

import numpy as np
import jax.numpy as jnp
from jax import vmap, nn, random
import jax.tree_util as jtu
import math as pymath

from pymdp.legacy import utils
from pymdp.agent import Agent
from pymdp.maths import compute_log_likelihood_single_modality
from pymdp.utils import norm_dist
from equinox import Module

class TestAgentJax(unittest.TestCase):

    def test_vmappable_agent_methods(self):

        dim, N = 5, 10
        sampling_key = random.PRNGKey(1)

        class BasicAgent(Module):
            A: jnp.ndarray
            B: jnp.ndarray 
            qs: jnp.ndarray

            def __init__(self, A, B, qs=None):
                self.A = A
                self.B = B
                self.qs = jnp.ones((N, dim))/dim if qs is None else qs
            
            @vmap
            def infer_states(self, obs):
                qs = nn.softmax(compute_log_likelihood_single_modality(obs, self.A))
                return qs, BasicAgent(self.A, self.B, qs=qs)

        A_key, B_key, obs_key, test_key = random.split(sampling_key, 4)

        all_A = vmap(norm_dist)(random.uniform(A_key, shape = (N, dim, dim)))
        all_B = vmap(norm_dist)(random.uniform(B_key, shape = (N, dim, dim)))
        all_obs = vmap(nn.one_hot, (0, None))(random.choice(obs_key, dim, shape = (N,)), dim)

        my_agent = BasicAgent(all_A, all_B)

        all_qs, my_agent = my_agent.infer_states(all_obs)

        assert all_qs.shape == my_agent.qs.shape
        self.assertTrue(jnp.allclose(all_qs, my_agent.qs))

        # validate that the method broadcasted properly
        for id_to_check in range(N):
            validation_qs = nn.softmax(compute_log_likelihood_single_modality(all_obs[id_to_check], all_A[id_to_check]))
            self.assertTrue(jnp.allclose(validation_qs, all_qs[id_to_check]))

    def test_agent_complex_action(self):
        """
        Test that an instance of the `Agent` class can be initialized and run with complex action dependency
        """
        np.random.seed(1)
        num_obs = [5, 4, 4]
        num_states = [2, 3, 1]
        num_controls = [2, 3, 2]

        A_factor_list = [[0], [0, 1], [0, 1, 2]]
        B_factor_list = [[0], [0, 1], [1, 2]]
        B_factor_control_list = [[], [0, 1], [0, 2]]
        A = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_factor_list)
        B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list, B_factor_control_list=B_factor_control_list)
        
        agent = Agent(
            A, B, 
            A_dependencies=A_factor_list, 
            B_dependencies=B_factor_list, 
            B_action_dependencies=B_factor_control_list,
            num_controls=num_controls,
            sampling_mode="full",
        )

        # dummy history
        action = agent.policies[np.random.randint(0, len(agent.policies))]
        observation = [np.random.randint(0, d, size=(1, 1)) for d in agent.num_obs]
        qs_hist = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), agent.D)

        prior, _ = agent.update_empirical_prior(action, qs_hist)
        qs = agent.infer_states(observation, prior)

        q_pi, G = agent.infer_policies(qs)
        action = agent.sample_action(q_pi)
        action_multi = agent.decode_multi_actions(action)
        action_reconstruct = agent.encode_multi_actions(action_multi)
        
        self.assertTrue(action_multi.shape[-1] == len(agent.num_controls))
        self.assertTrue(jnp.allclose(action, action_reconstruct))


if __name__ == "__main__":
    unittest.main()       








    
