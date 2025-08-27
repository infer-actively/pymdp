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

    def test_no_desired_batch_no_batched_input_construction(self):
        """
        Tests for the case where the user wants no batch size, and they pass in tensors with no batch size
        """

        num_obs = [2, 3, 4]
        num_states = [4, 5, 2]
        num_controls = [2, 3, 1]

        A_factor_list = [[0], [0, 1], [0, 1, 2]]
        B_factor_list = [[0], [0, 1], [1, 2]]

        A = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_factor_list)
        B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list, B_factor_control_list=None)

        agent = Agent(
            A, B, 
            A_dependencies=A_factor_list, 
            B_dependencies=B_factor_list, 
            num_controls=num_controls,
        )

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_factor_list[m]])
            self.assertTrue(agent.A[m].shape == (1, num_obs[m]) + num_states_m)
            self.assertTrue(agent.C[m].shape == (1, num_obs[m]))

        for f in range(len(num_states)):
            num_states_f = tuple([num_states[f] for f in B_factor_list[f]])
            self.assertTrue(agent.B[f].shape == (1, num_states[f]) + num_states_f + (num_controls[f],))
            self.assertTrue(agent.D[f].shape == (1, num_states[f]))
        
        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)
        self.assertTrue(agent.num_controls == num_controls)
        
        # Test the version with complex action dependencies
        num_obs = [2, 3]
        num_states = [4, 5, 2]
        num_controls = [2, 3, 2]
        A_factor_list = [[0, 1], [1]]
        B_factor_list = [[0], [0, 1, 2], [2]]
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

        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_factor_list[m]])
            self.assertTrue(agent.A[m].shape == (1, num_obs[m]) + num_states_m)
            self.assertTrue(agent.C[m].shape == (1, num_obs[m]))

        num_controls_flattened = []
        for f,action_dependency in enumerate(B_factor_control_list):
            if action_dependency == []:
                num_controls_f_flattened = 1
            else:
                num_controls_f_flattened = pymath.prod([num_controls[d] for d in action_dependency])

            num_controls_flattened.append(num_controls_f_flattened)
            num_states_f = tuple([num_states[f] for f in B_factor_list[f]])
            self.assertTrue(agent.B[f].shape == (1, num_states[f]) + num_states_f + (num_controls_f_flattened,))
            self.assertTrue(agent.D[f].shape == (1, num_states[f]))
        self.assertTrue(agent.num_controls == num_controls_flattened)

            
    def test_desired_batch_no_batched_input_construction(self):
        """
        Tests for the case where the user wants > 1 batch size, and they pass in tensors with no batch size
        """
        num_obs = [2, 3, 4]
        num_states = [4, 5, 2]
        num_controls = [2, 3, 1]
        desired_batch_size = 3

        A_factor_list = [[0], [0, 1], [0, 1, 2]]
        B_factor_list = [[0], [0, 1], [1, 2]]

        A = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_factor_list)
        B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list, B_factor_control_list=None)

        agent = Agent(
            A, B, 
            A_dependencies=A_factor_list, 
            B_dependencies=B_factor_list, 
            num_controls=num_controls,
            batch_size=desired_batch_size,
        )

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_factor_list[m]])
            self.assertTrue(agent.A[m].shape == (desired_batch_size, num_obs[m]) + num_states_m)
            self.assertTrue(agent.C[m].shape == (desired_batch_size, num_obs[m]))

        for f in range(len(num_states)):
            num_states_f = tuple([num_states[f] for f in B_factor_list[f]])
            self.assertTrue(agent.B[f].shape == (desired_batch_size, num_states[f]) + num_states_f + (num_controls[f],))
            self.assertTrue(agent.D[f].shape == (desired_batch_size, num_states[f]))
        
        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)
        self.assertTrue(agent.num_controls == num_controls)

        # Test the version with complex action dependencies
        num_obs = [2, 3]
        num_states = [4, 5, 2]
        num_controls = [2, 3, 2]
        A_factor_list = [[0, 1], [1]]
        B_factor_list = [[0], [0, 1, 2], [2]]
        B_factor_control_list = [[], [0, 1], [0, 2]]
        desired_batch_size = 3


        A = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_factor_list)
        B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list, B_factor_control_list=B_factor_control_list)

        agent = Agent(
            A, B, 
            A_dependencies=A_factor_list, 
            B_dependencies=B_factor_list, 
            B_action_dependencies=B_factor_control_list,
            num_controls=num_controls,
            sampling_mode="full",
            batch_size=desired_batch_size,
        )

        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_factor_list[m]])
            self.assertTrue(agent.A[m].shape == (desired_batch_size, num_obs[m]) + num_states_m)
            self.assertTrue(agent.C[m].shape == (desired_batch_size, num_obs[m]))

        num_controls_flattened = []
        for f,action_dependency in enumerate(B_factor_control_list):
            if action_dependency == []:
                num_controls_f_flattened = 1
            else:
                num_controls_f_flattened = pymath.prod([num_controls[d] for d in action_dependency])

            num_controls_flattened.append(num_controls_f_flattened)
            num_states_f = tuple([num_states[f] for f in B_factor_list[f]])
            self.assertTrue(agent.B[f].shape == (desired_batch_size, num_states[f]) + num_states_f + (num_controls_f_flattened,))
            self.assertTrue(agent.D[f].shape == (desired_batch_size, num_states[f]))
        self.assertTrue(agent.num_controls == num_controls_flattened)

    def test_desired_batch_and_batched_input_construction(self):
        """
        Tests for the case where the user wants >= 1 batch size, and they pass in tensors with that correct >= 1 batch size
        """
        num_obs = [2, 3, 4]
        num_states = [4, 5, 2]
        num_controls = [2, 3, 1]

        A_factor_list = [[0], [0, 1], [0, 1, 2]]
        B_factor_list = [[0], [0, 1], [1, 2]]

        A_single = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_factor_list)
        B_single = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list, B_factor_control_list=None)
        A = [a[None,...] for a in A_single]
        B = [b[None,...] for b in B_single]

        agent = Agent(
            A, B, 
            A_dependencies=A_factor_list, 
            B_dependencies=B_factor_list, 
            num_controls=num_controls,
            batch_size=1,
        )

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_factor_list[m]])
            self.assertTrue(agent.A[m].shape == (1,) + (num_obs[m],) + num_states_m)
            self.assertTrue(agent.C[m].shape == (1,) + (num_obs[m],))

        for f in range(len(num_states)):
            num_states_f = tuple([num_states[f] for f in B_factor_list[f]])
            self.assertTrue(agent.B[f].shape == (1,) + (num_states[f],) + num_states_f + (num_controls[f],))
            self.assertTrue(agent.D[f].shape == (1,) + (num_states[f],))
        
        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)
        self.assertTrue(agent.num_controls == num_controls)
        
        desired_batch_size = 3
        A = [jnp.broadcast_to(a, (desired_batch_size,) + a.shape) for a in A_single]
        B = [jnp.broadcast_to(b, (desired_batch_size,) + b.shape) for b in B_single]
        agent = Agent(
            A, B, 
            A_dependencies=A_factor_list, 
            B_dependencies=B_factor_list, 
            num_controls=num_controls,
            batch_size=desired_batch_size,
        )

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_factor_list[m]])
            self.assertTrue(agent.A[m].shape == (desired_batch_size,) + (num_obs[m],) + num_states_m)
            self.assertTrue(agent.C[m].shape == (desired_batch_size,) + (num_obs[m],))

        for f in range(len(num_states)):
            num_states_f = tuple([num_states[f] for f in B_factor_list[f]])
            self.assertTrue(agent.B[f].shape == (desired_batch_size,) + (num_states[f],) + num_states_f + (num_controls[f],))
            self.assertTrue(agent.D[f].shape == (desired_batch_size,) + (num_states[f],))
        
        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)
        self.assertTrue(agent.num_controls == num_controls)


        ### Now test the version with complex action dependencies
        num_obs = [2, 3]
        num_states = [4, 5, 2]
        num_controls = [2, 3, 2]
        A_factor_list = [[0, 1], [1]]
        B_factor_list = [[0], [0, 1, 2], [2]]
        B_factor_control_list = [[], [0, 1], [0, 2]]

        A_single = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_factor_list)
        B_single = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list, B_factor_control_list=B_factor_control_list)
        A = [a[None,...] for a in A_single]
        B = [b[None,...] for b in B_single]

        agent = Agent(
            A, B, 
            A_dependencies=A_factor_list, 
            B_dependencies=B_factor_list, 
            B_action_dependencies=B_factor_control_list,
            num_controls=num_controls,
            sampling_mode="full",
            batch_size=1,
        )

        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_factor_list[m]])
            self.assertTrue(agent.A[m].shape == (1, num_obs[m]) + num_states_m)
            self.assertTrue(agent.C[m].shape == (1, num_obs[m]))

        num_controls_flattened = []
        for f,action_dependency in enumerate(B_factor_control_list):
            if action_dependency == []:
                num_controls_f_flattened = 1
            else:
                num_controls_f_flattened = pymath.prod([num_controls[d] for d in action_dependency])

            num_controls_flattened.append(num_controls_f_flattened)
            num_states_f = tuple([num_states[f] for f in B_factor_list[f]])
            self.assertTrue(agent.B[f].shape == (1, num_states[f]) + num_states_f + (num_controls_f_flattened,))
            self.assertTrue(agent.D[f].shape == (1, num_states[f]))
        self.assertTrue(agent.num_controls == num_controls_flattened)

        desired_batch_size = 3
        A = [jnp.broadcast_to(a, (desired_batch_size,) + a.shape) for a in A_single]
        B = [jnp.broadcast_to(b, (desired_batch_size,) + b.shape) for b in B_single]

        agent = Agent(
            A, B, 
            A_dependencies=A_factor_list, 
            B_dependencies=B_factor_list, 
            B_action_dependencies=B_factor_control_list,
            num_controls=num_controls,
            sampling_mode="full",
            batch_size=desired_batch_size,
        )

        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_factor_list[m]])
            self.assertTrue(agent.A[m].shape == (desired_batch_size, num_obs[m]) + num_states_m)
            self.assertTrue(agent.C[m].shape == (desired_batch_size, num_obs[m]))

        num_controls_flattened = []
        for f,action_dependency in enumerate(B_factor_control_list):
            if action_dependency == []:
                num_controls_f_flattened = 1
            else:
                num_controls_f_flattened = pymath.prod([num_controls[d] for d in action_dependency])

            num_controls_flattened.append(num_controls_f_flattened)
            num_states_f = tuple([num_states[f] for f in B_factor_list[f]])
            self.assertTrue(agent.B[f].shape == (desired_batch_size, num_states[f]) + num_states_f + (num_controls_f_flattened,))
            self.assertTrue(agent.D[f].shape == (desired_batch_size, num_states[f]))
        self.assertTrue(agent.num_controls == num_controls_flattened)


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
    
    def test_agent_validate_normalization_ok(self):
        """
        Agent should construct without errors when A and B are normalized
        """
        num_obs = [3, 4]
        num_states = [4, 5]
        num_controls = [1, 2]
        A_deps = [[0, 1], [0, 1]]
        B_deps = [[0], [1]]

        A = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_deps)
        B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_deps, B_factor_control_list=None)

        # Should not raise; Agent.__init__ calls self._validate() which calls validate_normalization
        _ = Agent(A, B, A_dependencies=A_deps, B_dependencies=B_deps, num_controls=num_controls)

    def test_agent_validate_normalization_raises_on_bad_A(self):
        """
        If A is not normalized along its outcome axis (axis=1 after broadcasting),
        Agent construction should raise a ValueError via validate_normalization
        """
        num_obs = [3, 4]
        num_states = [4, 5]
        num_controls = [1, 2]
        A_deps = [[0, 1], [0, 1]]
        B_deps = [[0], [1]]

        A_bad = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_deps)
        B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_deps, B_factor_control_list=None)

        # Corrupt A[0]: add to one categorical distribution so sums != 1 along axis=1
        A_bad[0][:,0,1] += 0.05  # preserves shape, breaks normalization on axis=1

        with self.assertRaises(ValueError):
            _ = Agent(A_bad, B, A_dependencies=A_deps, B_dependencies=B_deps, num_controls=num_controls)

    def test_agent_validate_normalization_raises_on_bad_B(self):
        """
        If B is zero-filled (or otherwise unnormalized) along its state axis (axis=1 after broadcasting),
        Agent construction should raise a ValueError via validate_normalization
        """
        num_obs = [3, 4]
        num_states = [4, 5]
        num_controls = [2, 2]
        A_deps = [[0, 1], [0, 1]]
        B_deps = [[0], [1]]

        A = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_deps)
        B_bad = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_deps, B_factor_control_list=None)

        # Corrupt B[0]: make it zero so sums == 0 along axis=1
        B_bad[0][:,0,1] *= 0.0  # preserves shape, breaks normalization on axis=1

        with self.assertRaises(ValueError):
            _ = Agent(A, B_bad, A_dependencies=A_deps, B_dependencies=B_deps, num_controls=num_controls)


if __name__ == "__main__":
    unittest.main()       








    
