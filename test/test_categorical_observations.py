#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import warnings
from jax import numpy as jnp, random as jr, nn
import jax.tree_util as jtu

from pymdp.agent import Agent
from pymdp import utils, inference, algos


class TestCategoricalObservationsCore(unittest.TestCase):
    """Core functionality tests for categorical observations"""

    def test_uncertain_observation_inference(self):
        """Test that uncertain categorical observations produce expected posteriors"""

        # A matrix where each state strongly predicts its corresponding observation
        A = [jnp.eye(3) * 0.9 + jnp.ones((3, 3)) * 0.033]
        prior = [jnp.ones(3) / 3]  # Uniform prior

        # Certain observation (one-hot)
        obs_certain = [nn.one_hot(0, 3)]
        qs_certain = algos.run_vanilla_fpi(A, obs_certain, prior, num_iter=16)

        # Should strongly believe in state 0
        self.assertTrue(jnp.allclose(jnp.array([0.933934, 0.03303304, 0.03303304]),qs_certain[0], atol=1e-3))

        # Uncertain observation (50/50 between outcomes 0 and 1)
        obs_uncertain = [jnp.array([0.5, 0.5, 0.0])]
        qs_uncertain = algos.run_vanilla_fpi(A, obs_uncertain, prior, num_iter=16)

        # Should spread belief between states 0 and 1
        self.assertTrue(jnp.allclose(jnp.array([0.4570241, 0.4570241, 0.08595184]),qs_uncertain[0], atol=1e-3))

        # Uniform uncertain observation
        obs_uniform = [jnp.ones(3) / 3]
        qs_uniform = algos.run_vanilla_fpi(A, obs_uniform, prior, num_iter=16)

        # Should remain close to uniform (observation provides no information)
        self.assertTrue(jnp.allclose(qs_uniform[0], jnp.ones(3) / 3, atol=0.1))

    def test_categorical_multimodality(self):
        """Test categorical observations with multiple modalities"""

        num_states = [3, 2]
        num_obs = [3, 2]

        A = utils.random_A_array(jr.PRNGKey(10), num_obs, num_states)
        prior = utils.random_factorized_categorical(jr.PRNGKey(11), num_states)

        # Categorical observations for both modalities
        obs = [
            jnp.array([0.7, 0.2, 0.1]),  # Modality 0: mostly outcome 0
            jnp.array([0.4, 0.6])        # Modality 1: mostly outcome 1
        ]

        qs = algos.run_vanilla_fpi(A, obs, prior, num_iter=16)

        # Check that inference runs without error and produces valid distributions
        for f in range(len(num_states)):
            self.assertTrue(jnp.allclose(qs[f].sum(), 1.0, atol=1e-5))
            self.assertTrue(jnp.all(qs[f] >= 0))

    def test_multi_factor_categorical(self):
        """Test categorical observations with multiple state factors"""

        num_states = [3, 4]
        num_obs = [5]

        A = utils.random_A_array(jr.PRNGKey(20), num_obs, num_states)
        prior = utils.random_factorized_categorical(jr.PRNGKey(21), num_states)

        # Uncertain categorical observation
        obs = [jnp.array([0.3, 0.3, 0.2, 0.1, 0.1])]

        qs = algos.run_vanilla_fpi(A, obs, prior, num_iter=16)

        # Verify valid posterior distributions
        self.assertTrue(jnp.allclose(qs[0].sum(), 1.0, atol=1e-5))
        self.assertTrue(jnp.allclose(qs[1].sum(), 1.0, atol=1e-5))


class TestCategoricalObservationsEdgeCases(unittest.TestCase):
    """Edge case and numerical stability tests"""

    def test_near_zero_probabilities(self):
        """Test categorical observations with very small probabilities"""

        num_states = [3]
        num_obs = [4]

        A = utils.random_A_array(jr.PRNGKey(30), num_obs, num_states)
        prior = utils.random_factorized_categorical(jr.PRNGKey(31), num_states)

        # Observation with very small probabilities
        obs = [jnp.array([0.98, 0.01, 0.005, 0.005])]

        qs = algos.run_vanilla_fpi(A, obs, prior, num_iter=16)

        # Should not produce NaN or infinite values
        self.assertFalse(jnp.any(jnp.isnan(qs[0])))
        self.assertFalse(jnp.any(jnp.isinf(qs[0])))
        self.assertTrue(jnp.allclose(qs[0].sum(), 1.0, atol=1e-5))

    def test_very_peaked_distribution(self):
        """Test with extremely peaked categorical observation"""

        num_states = [3]
        num_obs = [4]

        A = utils.random_A_array(jr.PRNGKey(40), num_obs, num_states)
        prior = utils.random_factorized_categorical(jr.PRNGKey(41), num_states)

        # Very peaked distribution (essentially one-hot but not exactly)
        obs = [jnp.array([0.9999, 0.0001/3, 0.0001/3, 0.0001/3])]

        qs = algos.run_vanilla_fpi(A, obs, prior, num_iter=16)

        # Should be numerically stable
        self.assertFalse(jnp.any(jnp.isnan(qs[0])))
        self.assertFalse(jnp.any(jnp.isinf(qs[0])))
        self.assertTrue(jnp.allclose(qs[0].sum(), 1.0, atol=1e-5))

    def test_uniform_categorical_observation(self):
        """Test with completely uniform categorical observation"""

        num_states = [3]
        num_obs = [4]

        A = utils.random_A_array(jr.PRNGKey(50), num_obs, num_states)
        prior = utils.random_factorized_categorical(jr.PRNGKey(51), num_states)

        # Uniform observation (complete uncertainty)
        obs = [jnp.ones(4) / 4]

        qs = algos.run_vanilla_fpi(A, obs, prior, num_iter=16)

        # Should converge to valid distribution
        self.assertTrue(jnp.allclose(qs[0].sum(), 1.0, atol=1e-5))
        self.assertTrue(jnp.all(qs[0] >= 0))


class TestCategoricalObservationsInferenceAlgorithms(unittest.TestCase):
    """Test categorical observations with different inference algorithms"""

    def setUp(self):
        """Common setup for algorithm tests"""
        self.num_states = [3, 2]
        self.num_obs = [3, 2]
        self.num_controls = [3, 1]

        # Define dependencies for factorized inference
        self.A_dependencies = [[0, 1], [1]]  # Modality 0 depends on both factors, modality 1 only on factor 1
        self.B_dependencies = [[0], [1]]

        self.A = utils.random_A_array(jr.PRNGKey(100), self.num_obs, self.num_states, A_dependencies=self.A_dependencies)
        self.B = utils.random_B_array(jr.PRNGKey(101), self.num_states, self.num_controls)
        self.C = utils.list_array_uniform(self.num_obs)

        self.obs_categorical = [
            jnp.array([0.6, 0.3, 0.1]),
            jnp.array([0.7, 0.3])
        ]
        self.prior = utils.random_factorized_categorical(jr.PRNGKey(102), self.num_states)

    def test_fpi_with_categorical_obs(self):
        """Test fixed-point iteration with categorical observations"""

        # Vanilla FPI needs A matrices without dependencies (full joint state space)
        A_full = utils.random_A_array(jr.PRNGKey(103), self.num_obs, self.num_states)

        qs = algos.run_vanilla_fpi(
            A_full,
            self.obs_categorical,
            self.prior,
            num_iter=16
        )

        # Verify valid posteriors
        for f in range(len(self.num_states)):
            self.assertTrue(jnp.allclose(qs[f].sum(), 1.0, atol=1e-5))
            self.assertTrue(jnp.all(qs[f] >= 0))

    def test_fpi_factorized_with_categorical_obs(self):
        """Test factorized FPI with categorical observations and dependencies"""

        qs = algos.run_factorized_fpi(
            self.A,
            self.obs_categorical,
            self.prior,
            self.A_dependencies,
            num_iter=16
        )

        # Verify valid posteriors
        for f in range(len(self.num_states)):
            self.assertTrue(jnp.allclose(qs[f].sum(), 1.0, atol=1e-5))
            self.assertTrue(jnp.all(qs[f] >= 0))

    def test_update_posterior_states_with_categorical(self):
        """Test high-level update_posterior_states with categorical observations"""

        # update_posterior_states expects observations with time dimension
        # Shape: (time=1, num_obs) for each modality
        obs_with_time = [
            jnp.array([[0.6, 0.3, 0.1]]),  # Shape (1, 3)
            jnp.array([[0.7, 0.3]])        # Shape (1, 2)
        ]

        qs_fpi = inference.update_posterior_states(
            self.A,
            self.B,
            obs_with_time,
            past_actions=None,
            prior=self.prior,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            num_iter=16,
            method="fpi",
            distr_obs=True,
        )

        for f in range(len(self.num_states)):
            self.assertTrue(jnp.allclose(qs_fpi[f].sum(), 1.0, atol=1e-5))


class TestCategoricalObservationsAgent(unittest.TestCase):
    """Integration tests with Agent class"""

    def test_agent_categorical_flag_false_discrete_obs(self):
        """Test agent with categorical_obs=False using discrete observations"""

        num_states = [3]
        num_obs = [4]
        num_controls = [3]

        A = utils.random_A_array(jr.PRNGKey(200), num_obs, num_states)
        B = utils.random_B_array(jr.PRNGKey(201), num_states, num_controls)

        agent = Agent(
            A=A, 
            B=B,
            policy_len=2,
            categorical_obs=False,
        )

        # Provide discrete observation with shape (batch=1, time=1)
        obs = [jnp.array([[2]])]
        qs = agent.infer_states(obs, agent.D)

        self.assertTrue(jnp.allclose(qs[0].sum(), 1.0, atol=1e-5))

    def test_agent_categorical_flag_true_categorical_obs(self):
        """Test agent with categorical_obs=True using categorical observations"""

        num_states = [3]
        num_obs = [4]
        num_controls = [3]

        A = utils.random_A_array(jr.PRNGKey(210), num_obs, num_states)
        B = utils.random_B_array(jr.PRNGKey(211), num_states, num_controls)

        agent = Agent(
            A=A, 
            B=B,
            policy_len=2,
            categorical_obs=True,
        )

        # Provide categorical observation with shape (batch=1, time=1, num_obs=4)
        obs = [jnp.array([[[0.5, 0.3, 0.15, 0.05]]])]
        qs = agent.infer_states(obs, agent.D)

        self.assertTrue(jnp.allclose(qs[0].sum(), 1.0, atol=1e-5))

    def test_agent_categorical_override(self):
        """Test that preprocess_fn parameter can override agent default"""

        num_states = [3]
        num_obs = [4]
        num_controls = [3]

        A = utils.random_A_array(jr.PRNGKey(220), num_obs, num_states)
        B = utils.random_B_array(jr.PRNGKey(221), num_states, num_controls)

        agent = Agent(
            A=A, 
            B=B,
            policy_len=2,
            categorical_obs=False,
        )

        # Override to use categorical observation for this call (shape: batch, time, num_obs)
        obs_categorical = [jnp.array([[[0.5, 0.3, 0.15, 0.05]]])]
        qs = agent.infer_states(obs_categorical, agent.D, preprocess_fn=lambda obs: obs)

        self.assertTrue(jnp.allclose(qs[0].sum(), 1.0, atol=1e-5))

    def test_agent_preprocess_fn_default_and_warning(self):
        """Test agent-level preprocess_fn default usage and warning when categorical_obs=False"""

        num_states = [3]
        num_obs = [4]
        num_controls = [3]

        A = utils.random_A_array(jr.PRNGKey(230), num_obs, num_states)
        B = utils.random_B_array(jr.PRNGKey(231), num_states, num_controls)

        called = {"count": 0}

        def preprocess_fn(obs):
            called["count"] += 1
            return obs

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent = Agent(
                A=A,
                B=B,
                policy_len=2,
                categorical_obs=False,
                preprocess_fn=preprocess_fn,
            )

        self.assertTrue(
            any(
                "preprocess_fn is set while categorical_obs=False" in str(warning.message)
                for warning in w
            )
        )

        obs_categorical = [jnp.array([[[0.5, 0.3, 0.15, 0.05]]])]
        qs = agent.infer_states(obs_categorical, agent.D)

        self.assertEqual(called["count"], 1)
        self.assertTrue(jnp.allclose(qs[0].sum(), 1.0, atol=1e-5))

    def test_agent_full_loop_categorical(self):
        """Test full agent perception-action loop with categorical observations"""

        A = [jnp.eye(3)]

        B = [jnp.eye(3).reshape(3, 3, 1).repeat(3, axis=2)]

        # Prefer observation 2
        C = [jnp.array([-1.0, -1.0, 2.0])]

        agent = Agent(
            A=A, 
            B=B, 
            C=C,
            categorical_obs=True,
            policy_len=1,
        )

        # Uncertain observation (shape: batch=1, time=1, num_obs=3)
        obs = [jnp.array([[[0.5, 0.4, 0.1]]])]

        qs = agent.infer_states(obs, agent.D)

        q_pi, _ = agent.infer_policies(qs)

        action = agent.sample_action(q_pi)

        self.assertTrue(jnp.allclose(qs[0].sum(), 1.0, atol=1e-5))
        self.assertTrue(jnp.allclose(q_pi.sum(), 1.0, atol=1e-5))
        self.assertIsInstance(action, (int, jnp.ndarray))


class TestCategoricalObservationsControl(unittest.TestCase):
    """Test categorical observations with control/planning features"""

    def test_policy_inference_with_categorical_obs(self):
        """Test that policy inference works with categorical observations"""

        A = [jnp.eye(3)]
        B = [jnp.eye(3).reshape(3, 3, 1).repeat(3, axis=2)]
        C = [jnp.array([-1.0, 0.0, 2.0])]

        agent = Agent(
            A=A, 
            B=B, 
            C=C,
            policy_len=2,
            categorical_obs=True,
        )

        obs = [jnp.array([[[0.6, 0.3, 0.1]]])]  # Shape (1, 1, 3)
        qs = agent.infer_states(obs, agent.D)
        q_pi, G = agent.infer_policies(qs)

        self.assertTrue(jnp.allclose(q_pi.sum(), 1.0, atol=1e-5))
        # G has shape (batch, num_policies), so check the last dimension
        self.assertEqual(G.shape[-1], len(agent.policies))

    def test_info_gain_with_categorical_obs(self):
        """Test epistemic value (info gain) computation with categorical observations"""

        num_states = [3, 2]
        num_obs = [3, 2]
        num_controls = [3, 1]

        A = utils.random_A_array(jr.PRNGKey(300), num_obs, num_states)
        B = utils.random_B_array(jr.PRNGKey(301), num_states, num_controls)

        agent = Agent(
            A=A, B=B,
            categorical_obs=True,
            policy_len=2,
        )

        obs = [
            jnp.array([[[0.5, 0.3, 0.2]]]),  # Shape (1, 1, 3)
            jnp.array([[[0.6, 0.4]]])        # Shape (1, 1, 2)
        ]

        qs = agent.infer_states(obs, agent.D)
        q_pi, G = agent.infer_policies(qs)

        self.assertTrue(jnp.allclose(q_pi.sum(), 1.0, atol=1e-5))

    def test_parameter_info_gain_with_categorical_obs(self):
        """Test parameter info gain with categorical observations"""

        num_states = [3]
        num_obs = [3]
        num_controls = [3]

        A = utils.random_A_array(jr.PRNGKey(310), num_obs, num_states)
        B = utils.random_B_array(jr.PRNGKey(311), num_states, num_controls)
        pA = utils.list_array_scaled(jtu.tree_map(lambda x: x.shape, A), scale=1.0)

        agent = Agent(
            A=A, B=B,
            pA=pA,
            categorical_obs=True,
            policy_len=2,
            use_states_info_gain=False,
            use_param_info_gain=True  # Enable parameter info gain
        )

        obs = [jnp.array([[[0.5, 0.3, 0.2]]])]  # Shape (1, 1, 3)
        qs = agent.infer_states(obs, agent.D)
        q_pi, G = agent.infer_policies(qs)

        self.assertTrue(jnp.allclose(q_pi.sum(), 1.0, atol=1e-5))


class TestCategoricalObservationsLearning(unittest.TestCase):
    """Test learning with categorical observations"""

    def test_learning_A_matrix_with_categorical(self):
        """Test that A matrix learning works with categorical observations"""

        num_states = [3]
        num_obs = [3]
        num_controls = [3]

        A = utils.random_A_array(jr.PRNGKey(400), num_obs, num_states)
        B = utils.random_B_array(jr.PRNGKey(401), num_states, num_controls)
        pA = utils.list_array_scaled(jtu.tree_map(lambda x: x.shape, A), scale=1.0)

        agent = Agent(
            A=A, 
            B=B,
            pA=pA,
            categorical_obs=True,
            policy_len=2,
            learn_A=True,
        )

        # Sequence of categorical observations (each with shape (1, 1, 3))
        obs_sequence = [
            [jnp.array([[[0.8, 0.15, 0.05]]])],
            [jnp.array([[[0.1, 0.7, 0.2]]])],
            [jnp.array([[[0.2, 0.2, 0.6]]])]
        ]

        initial_pA = agent.pA[0].copy()

        for obs in obs_sequence:
            qs = agent.infer_states(obs, agent.D)
            q_pi, _ = agent.infer_policies(qs)
            action = agent.sample_action(q_pi)
            # infer_parameters takes (beliefs, outcomes, actions) where actions shape is (batch, time, num_factors)
            actions = jnp.array([[[action]]])  # Shape (1, 1, 1) for batch=1, time=1, num_factors=1
            agent = agent.infer_parameters(qs, obs, actions)

        # pA should have been updated
        self.assertFalse(jnp.allclose(agent.pA[0], initial_pA, atol=1e-5))

    def test_learning_with_uncertain_observations(self):
        """Test that learning weights updates by observation certainty"""

        num_states = [2]
        num_controls = [2]

        A = [
            jnp.array(
                [
                    [0.9, 0.1], 
                    [0.1, 0.9]
                ]
            )
        ]
        B = utils.random_B_array(jr.PRNGKey(410), num_states, num_controls)
        pA = utils.list_array_scaled(jtu.tree_map(lambda x: x.shape, A), scale=1.0)

        # Agent 1: Certain observation
        agent1 = Agent(A=A, B=B, pA=pA, categorical_obs=True, learn_A=True, policy_len=1)

        # Agent 2: Uncertain observation
        agent2 = Agent(A=A, B=B, pA=pA, categorical_obs=True, learn_A=True, policy_len=1)

        # Certain observation (one-hot) with shape (1, 1, 2)
        obs_certain = [jnp.array([[[1.0, 0.0]]])]
        qs1 = agent1.infer_states(obs_certain, agent1.D)
        actions1 = jnp.array([[[0]]])  # Shape (1, 1, 1) for batch=1, time=1, num_factors=1
        agent1 = agent1.infer_parameters(qs1, obs_certain, actions1)

        # Uncertain observation (50/50) with shape (1, 1, 2)
        obs_uncertain = [jnp.array([[[0.5, 0.5]]])]
        qs2 = agent2.infer_states(obs_uncertain, agent2.D)
        actions2 = jnp.array([[[0]]])  # Shape (1, 1, 1)
        agent2 = agent2.infer_parameters(qs2, obs_uncertain, actions2)

        # Both should update, but certain observation should update more
        expected_pA_1 = jnp.array(
            [
                [
                    [1.9, 1.1],
                    [1.0, 1.0],
                ]
            ]
        )

        expected_pA_2 = jnp.array(
            [
                [
                    [1.25, 1.25],
                    [1.25, 1.25],
                ]
            ]
        )

        self.assertTrue(jnp.array_equal(agent1.pA[0], expected_pA_1))
        self.assertTrue(jnp.array_equal(agent2.pA[0], expected_pA_2))


class TestCategoricalObservationsBatched(unittest.TestCase):
    """Test categorical observations with batched agents"""

    def test_batched_categorical_observations(self):
        """Test categorical observations with batch_size > 1"""

        batch_size = 2
        num_states = [3]
        num_obs = [4]
        num_controls = [3]

        A = utils.random_A_array(jr.PRNGKey(500), num_obs, num_states)
        B = utils.random_B_array(jr.PRNGKey(501), num_states, num_controls)

        agent = Agent(
            A=A, 
            B=B,
            categorical_obs=True,
            policy_len=2,
            batch_size=batch_size,
        )

        # Batched categorical observations (shape: batch=2, time=1, num_obs=4)
        obs = [
                jnp.array(
                    [
                        [[0.7, 0.2, 0.05, 0.05]],   # Batch 0
                        [[0.3, 0.5, 0.15, 0.05]],   # Batch 1
                    ]
                )
            ]

        qs = agent.infer_states(obs, agent.D)

        # Verify correct batch dimension
        self.assertEqual(qs[0].shape[0], batch_size)

        # Verify each batch element is a valid distribution
        for b in range(batch_size):
            self.assertTrue(jnp.allclose(qs[0][b].sum(), 1.0, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
