#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Dimitrije Markovic, Conor Heins
"""

import unittest
from functools import partial

import numpy as np
from jax import numpy as jnp, random as jr
from jax import vmap, nn, jit, grad
import jax.tree_util as jtu
import math as pymath

from pymdp import utils
from pymdp import control
from pymdp.agent import Agent
from pymdp.maths import compute_log_likelihood_single_modality, log_stable
from equinox import Module, EquinoxRuntimeError

class TestAgentJax(unittest.TestCase):

    def test_no_desired_batch_no_batched_input_construction(self):
        """
        Tests for the case where the user wants no batch size, and they pass in tensors with no batch size
        """

        a_key, b_key = jr.split(jr.PRNGKey(1), 2)

        num_obs = [2, 3, 4]
        num_states = [4, 5, 2]
        num_controls = [2, 3, 1]

        A_dependencies = [[0], [0, 1], [0, 1, 2]]
        B_dependencies = [[0], [0, 1], [1, 2]]

        A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
        B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies)

        agent = Agent(
            A, B, 
            A_dependencies=A_dependencies, 
            B_dependencies=B_dependencies, 
            num_controls=num_controls,
        )

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_dependencies[m]])
            self.assertTrue(agent.A[m].shape == (1, num_obs[m]) + num_states_m)
            self.assertTrue(agent.C[m].shape == (1, num_obs[m]))

        for f in range(len(num_states)):
            num_states_f = tuple([num_states[f] for f in B_dependencies[f]])
            self.assertTrue(agent.B[f].shape == (1, num_states[f]) + num_states_f + (num_controls[f],))
            self.assertTrue(agent.D[f].shape == (1, num_states[f]))
        
        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)
        self.assertTrue(agent.num_controls == num_controls)
        
        # Test the version with complex action dependencies
        num_obs = [2, 3]
        num_states = [4, 5, 2]
        num_controls = [2, 3, 2]
        A_dependencies = [[0, 1], [1]]
        B_dependencies = [[0], [0, 1, 2], [2]]
        B_action_dependencies = [[], [0, 1], [0, 2]]

        A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
        B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies, B_action_dependencies=B_action_dependencies    )

        agent = Agent(
            A, B, 
            A_dependencies=A_dependencies, 
            B_dependencies=B_dependencies, 
            B_action_dependencies=B_action_dependencies,
            num_controls=num_controls,
            sampling_mode="full",
        )

        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_dependencies[m]])
            self.assertTrue(agent.A[m].shape == (1, num_obs[m]) + num_states_m)
            self.assertTrue(agent.C[m].shape == (1, num_obs[m]))

        num_controls_flattened = []
        for f,action_dependency in enumerate(B_action_dependencies):
            if action_dependency == []:
                num_controls_f_flattened = 1
            else:
                num_controls_f_flattened = pymath.prod([num_controls[d] for d in action_dependency])

            num_controls_flattened.append(num_controls_f_flattened)
            num_states_f = tuple([num_states[f] for f in B_dependencies[f]])
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

        a_key, b_key = jr.split(jr.PRNGKey(2), 2)
    
        A_dependencies = [[0], [0, 1], [0, 1, 2]]
        B_dependencies = [[0], [0, 1], [1, 2]]

        A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
        B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies, B_action_dependencies=None)

        agent = Agent(
            A, B, 
            A_dependencies=A_dependencies, 
            B_dependencies=B_dependencies, 
            num_controls=num_controls,
            batch_size=desired_batch_size,
        )

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_dependencies[m]])
            self.assertTrue(agent.A[m].shape == (desired_batch_size, num_obs[m]) + num_states_m)
            self.assertTrue(agent.C[m].shape == (desired_batch_size, num_obs[m]))

        for f in range(len(num_states)):
            num_states_f = tuple([num_states[f] for f in B_dependencies[f]])
            self.assertTrue(agent.B[f].shape == (desired_batch_size, num_states[f]) + num_states_f + (num_controls[f],))
            self.assertTrue(agent.D[f].shape == (desired_batch_size, num_states[f]))
        
        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)
        self.assertTrue(agent.num_controls == num_controls)

        # Test the version with complex action dependencies
        num_obs = [2, 3]
        num_states = [4, 5, 2]
        num_controls = [2, 3, 2]
        A_dependencies = [[0, 1], [1]]
        B_dependencies = [[0], [0, 1, 2], [2]]
        B_action_dependencies = [[], [0, 1], [0, 2]]
        desired_batch_size = 3

        
        A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
        B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies, B_action_dependencies=B_action_dependencies)

        agent = Agent(
            A, B, 
            A_dependencies=A_dependencies, 
            B_dependencies=B_dependencies, 
            B_action_dependencies=B_action_dependencies,
            num_controls=num_controls,
            sampling_mode="full",
            batch_size=desired_batch_size,
        )

        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_dependencies[m]])
            self.assertTrue(agent.A[m].shape == (desired_batch_size, num_obs[m]) + num_states_m)
            self.assertTrue(agent.C[m].shape == (desired_batch_size, num_obs[m]))

        num_controls_flattened = []
        for f,action_dependency in enumerate(B_action_dependencies):
            if action_dependency == []:
                num_controls_f_flattened = 1
            else:
                num_controls_f_flattened = pymath.prod([num_controls[d] for d in action_dependency])

            num_controls_flattened.append(num_controls_f_flattened)
            num_states_f = tuple([num_states[f] for f in B_dependencies[f]])
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

        a_key, b_key = jr.split(jr.PRNGKey(3), 2)

        A_dependencies = [[0], [0, 1], [0, 1, 2]]
        B_dependencies = [[0], [0, 1], [1, 2]]

        A_single = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
        B_single = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies, B_action_dependencies=None)
        A = [a[None,...] for a in A_single]
        B = [b[None,...] for b in B_single]

        agent = Agent(
            A, B, 
            A_dependencies=A_dependencies, 
            B_dependencies=B_dependencies, 
            num_controls=num_controls,
            batch_size=1,
        )

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_dependencies[m]])
            self.assertTrue(agent.A[m].shape == (1,) + (num_obs[m],) + num_states_m)
            self.assertTrue(agent.C[m].shape == (1,) + (num_obs[m],))

        for f in range(len(num_states)):
            num_states_f = tuple([num_states[f] for f in B_dependencies[f]])
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
            A_dependencies=A_dependencies, 
            B_dependencies=B_dependencies, 
            num_controls=num_controls,
            batch_size=desired_batch_size,
        )

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_dependencies[m]])
            self.assertTrue(agent.A[m].shape == (desired_batch_size,) + (num_obs[m],) + num_states_m)
            self.assertTrue(agent.C[m].shape == (desired_batch_size,) + (num_obs[m],))

        for f in range(len(num_states)):
            num_states_f = tuple([num_states[f] for f in B_dependencies[f]])
            self.assertTrue(agent.B[f].shape == (desired_batch_size,) + (num_states[f],) + num_states_f + (num_controls[f],))
            self.assertTrue(agent.D[f].shape == (desired_batch_size,) + (num_states[f],))
        
        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)
        self.assertTrue(agent.num_controls == num_controls)


        ### Now test the version with complex action dependencies
        num_obs = [2, 3]
        num_states = [4, 5, 2]
        num_controls = [2, 3, 2]
        A_dependencies = [[0, 1], [1]]
        B_dependencies = [[0], [0, 1, 2], [2]]
        B_action_dependencies = [[], [0, 1], [0, 2]]

        A_single = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
        B_single = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies, B_action_dependencies=B_action_dependencies)
        A = [a[None,...] for a in A_single]
        B = [b[None,...] for b in B_single]

        agent = Agent(
            A, B, 
            A_dependencies=A_dependencies, 
            B_dependencies=B_dependencies, 
            B_action_dependencies=B_action_dependencies,
            num_controls=num_controls,
            sampling_mode="full",
            batch_size=1,
        )

        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_dependencies[m]])
            self.assertTrue(agent.A[m].shape == (1, num_obs[m]) + num_states_m)
            self.assertTrue(agent.C[m].shape == (1, num_obs[m]))

        num_controls_flattened = []
        for f,action_dependency in enumerate(B_action_dependencies):
            if action_dependency == []:
                num_controls_f_flattened = 1
            else:
                num_controls_f_flattened = pymath.prod([num_controls[d] for d in action_dependency])

            num_controls_flattened.append(num_controls_f_flattened)
            num_states_f = tuple([num_states[f] for f in B_dependencies[f]])
            self.assertTrue(agent.B[f].shape == (1, num_states[f]) + num_states_f + (num_controls_f_flattened,))
            self.assertTrue(agent.D[f].shape == (1, num_states[f]))
        self.assertTrue(agent.num_controls == num_controls_flattened)

        desired_batch_size = 3
        A = [jnp.broadcast_to(a, (desired_batch_size,) + a.shape) for a in A_single]
        B = [jnp.broadcast_to(b, (desired_batch_size,) + b.shape) for b in B_single]

        agent = Agent(
            A, B, 
            A_dependencies=A_dependencies, 
            B_dependencies=B_dependencies, 
            B_action_dependencies=B_action_dependencies,
            num_controls=num_controls,
            sampling_mode="full",
            batch_size=desired_batch_size,
        )

        self.assertTrue(agent.num_obs == num_obs)
        self.assertTrue(agent.num_states == num_states)

        for m in range(len(num_obs)):
            num_states_m = tuple([num_states[f] for f in A_dependencies[m]])
            self.assertTrue(agent.A[m].shape == (desired_batch_size, num_obs[m]) + num_states_m)
            self.assertTrue(agent.C[m].shape == (desired_batch_size, num_obs[m]))

        num_controls_flattened = []
        for f,action_dependency in enumerate(B_action_dependencies):
            if action_dependency == []:
                num_controls_f_flattened = 1
            else:
                num_controls_f_flattened = pymath.prod([num_controls[d] for d in action_dependency])

            num_controls_flattened.append(num_controls_f_flattened)
            num_states_f = tuple([num_states[f] for f in B_dependencies[f]])
            self.assertTrue(agent.B[f].shape == (desired_batch_size, num_states[f]) + num_states_f + (num_controls_f_flattened,))
            self.assertTrue(agent.D[f].shape == (desired_batch_size, num_states[f]))
        self.assertTrue(agent.num_controls == num_controls_flattened)


    def test_vmappable_agent_methods(self):

        dim, N = 5, 10
        sampling_key = jr.PRNGKey(4)

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

        A_key, B_key, obs_key, test_key = jr.split(sampling_key, 4)

        all_A = vmap(utils.norm_dist)(jr.uniform(A_key, shape = (N, dim, dim)))
        all_B = vmap(utils.norm_dist)(jr.uniform(B_key, shape = (N, dim, dim)))
        all_obs = vmap(nn.one_hot, (0, None))(jr.choice(obs_key, dim, shape = (N,)), dim)

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
        num_obs = [5, 4, 4]
        num_states = [2, 3, 1]
        num_controls = [2, 3, 2]

        a_key, b_key = jr.split(jr.PRNGKey(5), 2)

        A_dependencies = [[0], [0, 1], [0, 1, 2]]
        B_dependencies = [[0], [0, 1], [1, 2]]
        B_action_dependencies = [[], [0, 1], [0, 2]]
        A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
        B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies, B_action_dependencies=B_action_dependencies)
        
        agent = Agent(
            A, B, 
            A_dependencies=A_dependencies, 
            B_dependencies=B_dependencies, 
            B_action_dependencies=B_action_dependencies,
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

        a_key, b_key = jr.split(jr.PRNGKey(6), 2)

        A_dependencies = [[0, 1], [0, 1]]
        B_deps = [[0], [1]]

        A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
        B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_deps, B_action_dependencies=None)

        # Should not raise; Agent.__init__ calls self._validate() which calls validate_normalization
        _ = Agent(A, B, A_dependencies=A_dependencies, B_dependencies=B_deps, num_controls=num_controls)

        # also in presence of Dirichlet priors
        pA = utils.list_array_scaled(jtu.tree_map(lambda x: x.shape, A), scale=10.0)  # uniform Dirichlet priors
        pB = [10.0 * b for b in B]
             
        # Should not raise; Agent.__init__ calls self._validate() which calls validate_normalization
        _ = Agent(A, B, A_dependencies=A_dependencies, B_dependencies=B_deps, num_controls=num_controls, pA=pA, pB=pB) 

    def test_agent_validate_normalization_raises_on_bad_A(self):
        """
        If A is not normalized along its outcome axis (axis=1 after broadcasting),
        Agent construction should raise a ValueError via validate_normalization
        """
        num_obs = [3, 4]
        num_states = [4, 5]
        num_controls = [1, 2]

        a_key, b_key = jr.split(jr.PRNGKey(7), 2)

        A_dependencies = [[0, 1], [0, 1]]
        B_dependencies = [[0], [1]]

        A_bad = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
        B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies, B_action_dependencies=None)
        # Corrupt A[0]: add to one categorical distribution so sums != 1 along axis=1
        A_bad[0] = A_bad[0].at[:,0,1].add(0.05)  # preserves shape, breaks normalization on axis=1

        with self.assertRaisesRegex((EquinoxRuntimeError, ValueError),
                             r"properly normalised and sum to 1"):
            _ = Agent(A_bad, B, A_dependencies=A_dependencies, B_dependencies=B_dependencies, num_controls=num_controls)
        
        # also raises in presence of Dirichlet priors
        pA = [(10.0 * a + 1.0) for a in A_bad]  # uniform Dirichlet priors
        pB = [(10.0 * b) for b in B]
        with self.assertRaisesRegex((EquinoxRuntimeError, ValueError),
                             r"properly normalised and sum to 1"):
            _ = Agent(A_bad, B, A_dependencies=A_dependencies, B_dependencies=B_dependencies, num_controls=num_controls, pA=pA, pB=pB)

    def test_agent_validate_normalization_raises_on_bad_B(self):
        """
        If B is zero-filled (or otherwise unnormalized) along its state axis (axis=1 after broadcasting),
        Agent construction should raise a ValueError via validate_normalization
        """
        num_obs = [3, 4]
        num_states = [4, 5]
        num_controls = [2, 2]

        a_key, b_key = jr.split(jr.PRNGKey(7), 2)

        A_dependencies = [[0, 1], [0, 1]]
        B_dependencies = [[0], [1]]

        A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
        B_bad = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies, B_action_dependencies=None)
        # Corrupt B[0]: make it zero so sums == 0 along axis=1
        B_bad[0] = B_bad[0].at[:,0,1].set(0.0) # preserves shape, breaks normalization on axis=1

        with self.assertRaisesRegex((EquinoxRuntimeError, ValueError),
                        r"sum to zero"):
            _ = Agent(A, B_bad, A_dependencies=A_dependencies, B_dependencies=B_dependencies, num_controls=num_controls)
        
        # also raises in presence of Dirichlet priors
        pA = [10.0 * a for a in A]  # uniform Dirichlet priors
        pB = [(10.0 * b + 1) for b in B_bad]
        with self.assertRaisesRegex((EquinoxRuntimeError, ValueError),
                        r"sum to zero"):
            _ = Agent(A, B_bad, A_dependencies=A_dependencies, B_dependencies=B_dependencies, num_controls=num_controls, pA=pA, pB=pB)

    def test_agent_with_A_learning_requires_pA(self):
        """Test that creating an agent with learn_A=True 
        but no pA raises an AssertionError."""
        
        num_obs = [1]
        num_states = [1] 
        num_controls = [1]

        a_key, b_key = jr.split(jr.PRNGKey(8), 2)
        A = utils.random_A_array(a_key, num_obs, num_states)
        B = utils.random_B_array(b_key, num_states, num_controls)

        with self.assertRaises(AssertionError):
            Agent(A=A, B=B, learn_A=True)

    def test_agent_construction_jittable(self):
        """
        Test that the constructor call of the Agent class is jittable
        """
        num_obs = [3, 4]
        num_states = [4, 5, 6]
        num_controls = [2, 3, 1]
        A_dependencies = [[0], [0, 1, 2]]
        B_dependencies = [[0], [1], [2]]
        batch_size = 2

        a_key, b_key, d_key = jr.split(jr.PRNGKey(123), 3)

        A = utils.random_A_array(
            a_key, num_obs, num_states, A_dependencies=A_dependencies
        )
        B = utils.random_B_array(
            b_key, num_states, num_controls, B_dependencies=B_dependencies
        )
        pA = utils.list_array_scaled(jtu.tree_map(lambda x: x.shape, A), scale=1.0)
        pB = utils.list_array_scaled(jtu.tree_map(lambda x: x.shape, B), scale=1.0)
        D = utils.random_factorized_categorical(d_key, num_states)

        def _broadcast(arr_list):
            return [
                jnp.broadcast_to(jnp.array(arr), (batch_size,) + arr.shape)
                for arr in arr_list
            ]

        A_batched = _broadcast(A)
        B_batched = _broadcast(B)
        pA_batched = _broadcast(pA)
        pB_batched = _broadcast(pB)
        D_batched = _broadcast(D)

        def construct_agent(A, B, pA, pB, D):
            """ Constructs simple wrapping function that just builds an Agent """

            agent = Agent(
                A,
                B,
                D=D,
                A_dependencies=A_dependencies,
                B_dependencies=B_dependencies,
                num_controls=num_controls,
                batch_size=batch_size,
                learn_A=True,
                learn_B=True,
                pA=pA,
                pB=pB,
            )

            return agent
        
        # write a unit test that the Agent constructor is jittable
        jitted_constructor = jit(construct_agent)
        agent_instance = jitted_constructor(
            A_batched, B_batched, pA_batched, pB_batched, D_batched
        )
        self.assertIsInstance(agent_instance, Agent)
    
    def test_b_learning_updates_inductive_matrix(self):
        """
        Ensure the inductive I matrices are recomputed after B-learning when inductive inference is enabled.
        """
        num_obs = [2]
        num_states = [2]
        num_controls = [2]
        A_dependencies = [[0]]
        B_dependencies = [[0]]
        batch_size = 1
        inductive_depth = 3

        a_key, b_key, d_key = jr.split(jr.PRNGKey(777), 3)
        A = utils.random_A_array(a_key, num_obs, num_states, A_dependencies=A_dependencies)
        B = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies)
        D = utils.random_factorized_categorical(d_key, num_states)
        H = [jnp.array([1.0, 0.0])]
        pB = [jnp.ones_like(b) for b in B]

        agent = Agent(
            A,
            B,
            D=D,
            H=H,
            pB=pB,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            num_controls=num_controls,
            batch_size=batch_size,
            learn_B=True,
            use_inductive=True,
            inductive_depth=inductive_depth,
            inductive_threshold=0.1,
        )

        beliefs = [jnp.array([[[0.7, 0.3], [0.2, 0.8]]], dtype=jnp.float32)]
        dummy_outcomes = [jnp.zeros((batch_size,), dtype=jnp.int32)]
        actions = jnp.array([[[1]]], dtype=jnp.int32)

        agent = agent.infer_parameters(
            beliefs,
            dummy_outcomes,
            actions,
        )

        expected_I = vmap(partial(control.generate_I_matrix, depth=agent.inductive_depth))(
            agent.H,
            agent.B,
            agent.inductive_threshold,
        )

        for f in range(agent.num_factors):
            self.assertTrue(jnp.array_equal(agent.I[f], expected_I[f]))

    def test_valid_gradients_one_step_ahead(self):
        """
        This unit test checks that gradients can be computed through a single time step of
        active inference, including construction of the agent, state inference, policy inference,
        and computation of multi-action log-probabilities.
        """

        num_obs = [3, 4]
        num_states = [4, 5, 6]
        num_controls = [2, 3, 1]
        A_dependencies = [[0], [0, 1, 2]]
        B_dependencies = [[0], [1], [2]]
        batch_size = 2

        a_key, b_key, d_key = jr.split(jr.PRNGKey(123), 3)

        A = utils.random_A_array(
            a_key, num_obs, num_states, A_dependencies=A_dependencies
        )
        B = utils.random_B_array(
            b_key, num_states, num_controls, B_dependencies=B_dependencies
        )
        pA = utils.list_array_scaled(jtu.tree_map(lambda x: x.shape, A), scale=1.0)
        pB = utils.list_array_scaled(jtu.tree_map(lambda x: x.shape, B), scale=1.0)
        D = utils.random_factorized_categorical(d_key, num_states)

        def _broadcast(arr_list):
            return [
                jnp.broadcast_to(jnp.array(arr), (batch_size,) + arr.shape)
                for arr in arr_list
            ]

        A_batched = _broadcast(A)
        B_batched = _broadcast(B)
        pA_batched = _broadcast(pA)
        pB_batched = _broadcast(pB)
        D_batched = _broadcast(D)

        def construct_agent(A, B, pA, pB, D, inference_algo="fpi"):
            """ Constructs simple wrapping function that just builds an Agent """

            agent = Agent(
                A,
                B,
                D=D,
                A_dependencies=A_dependencies,
                B_dependencies=B_dependencies,
                num_controls=num_controls,
                batch_size=batch_size,
                learn_A=True,
                learn_B=False,
                pA=pA,
                pB=pB,
                inference_algo=inference_algo,
            )

            return agent
        
        modality_keys = jr.split(jr.PRNGKey(0), len(num_obs))
        observations = [jr.randint(k, shape=(batch_size,), minval=0, maxval=d)[...,None] for d, k in zip(num_obs, modality_keys)]   

        def one_step_active_inference(A, B, pA, pB, D, inference_algo="fpi"):
            """
            Constructs an agent and runs a single step of active inference,
            returning the log-probabilities of multi-actions.
            """

            # construct agent
            agent = construct_agent(A, B, pA, pB, D, inference_algo=inference_algo)

            # infer states
            qs = agent.infer_states(observations, empirical_prior=D)

            # infer policies
            q_pi, G = agent.infer_policies(qs)

            # compute multi-action log-probabilities
            multiaction_probs = agent.multiaction_probabilities(q_pi)

            return log_stable(multiaction_probs).sum()
        
        gradients_wrt_params = grad(one_step_active_inference, argnums=(0,1,2,3,4))(
            A_batched, B_batched, pA_batched, pB_batched, D_batched, inference_algo="fpi"
        )
        self.assertTrue(all(jtu.tree_map(lambda x: ~jnp.any(jnp.isnan(x)), gradients_wrt_params)))

        gradients_wrt_params = grad(one_step_active_inference, argnums=(0,1,2,3,4))(
            A_batched, B_batched, pA_batched, pB_batched, D_batched, inference_algo="ovf"
        )
        self.assertTrue(all(jtu.tree_map(lambda x: ~jnp.any(jnp.isnan(x)), gradients_wrt_params)))

        gradients_wrt_params = grad(one_step_active_inference, argnums=(0,1,2,3,4))(
            A_batched, B_batched, pA_batched, pB_batched, D_batched, inference_algo="mmp"
        )
        self.assertTrue(all(jtu.tree_map(lambda x: ~jnp.any(jnp.isnan(x)), gradients_wrt_params)))

        gradients_wrt_params = grad(one_step_active_inference, argnums=(0,1,2,3,4))(
            A_batched, B_batched, pA_batched, pB_batched, D_batched, inference_algo="vmp"
        )
        self.assertTrue(all(jtu.tree_map(lambda x: ~jnp.any(jnp.isnan(x)), gradients_wrt_params)))

        def one_step_active_inference_with_learning(A, B, pA, pB, D, inference_algo="fpi"):
            """
            Constructs an agent and runs a single step of active inference, including parameter updating,
            returning the log-probabilities of multi-actions.
            """

            agent = construct_agent(A, B, pA, pB, D, inference_algo=inference_algo)

            # infer states
            qs = agent.infer_states(observations, empirical_prior=D)

            agent = agent.infer_parameters(
                qs,
                observations,
                actions=None,
            )

            # infer policies
            q_pi, G = agent.infer_policies(qs)

            # compute multi-action log-probabilities
            multiaction_probs = agent.multiaction_probabilities(q_pi)

            return log_stable(multiaction_probs).sum()
        
        gradients_wrt_params = grad(one_step_active_inference_with_learning, argnums=(0,1,2,3,4))(
            A_batched, B_batched, pA_batched, pB_batched, D_batched, inference_algo="fpi"
        )
        self.assertTrue(all(jtu.tree_map(lambda x: ~jnp.any(jnp.isnan(x)), gradients_wrt_params)))

        gradients_wrt_params = grad(one_step_active_inference_with_learning, argnums=(0,1,2,3,4))(
            A_batched, B_batched, pA_batched, pB_batched, D_batched, inference_algo="mmp"
        )
        self.assertTrue(all(jtu.tree_map(lambda x: ~jnp.any(jnp.isnan(x)), gradients_wrt_params)))

        gradients_wrt_params = grad(one_step_active_inference_with_learning, argnums=(0,1,2,3,4))(
            A_batched, B_batched, pA_batched, pB_batched, D_batched, inference_algo="vmp"
        )
        self.assertTrue(all(jtu.tree_map(lambda x: ~jnp.any(jnp.isnan(x)), gradients_wrt_params)))


        

if __name__ == "__main__":
    unittest.main()       








    
