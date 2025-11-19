#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for the jittable rollout loop."""

import unittest

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from pymdp.agent import Agent
from pymdp.envs.env import Env
from pymdp.envs.rollout import rollout, default_policy_search
from pymdp import utils

class TestRolloutFunction(unittest.TestCase):
    def setUp(self):
        self.num_obs = [3]
        self.num_states = [2]
        self.num_controls = [2]
        self.A_dependencies = [[0]]
        self.B_dependencies = [[0]]
        self.batch_size = 1

    def build_agent_env(
        self,
        *,
        learn_A=False,
        learn_B=False,
        learning_mode="online",
        inference_algo="fpi",
        policy_len=1,
        seed=0,
    ):
        
        A_key, B_key, D_key = jr.split(jr.PRNGKey(seed), 3)

        A = utils.random_A_array(A_key, self.num_obs, self.num_states, A_dependencies=self.A_dependencies)
        B = utils.random_B_array(B_key,
            self.num_states, self.num_controls, B_dependencies=self.B_dependencies
        )
        pA = utils.list_array_scaled([a.shape for a in A], scale=1.0)
        pB = utils.list_array_scaled([b.shape for b in B], scale=1.0)
        D = utils.random_factorized_categorical(D_key, self.num_states)

        def _broadcast(arr_list):
            return [
                jnp.broadcast_to(jnp.array(arr), (self.batch_size,) + arr.shape)
                for arr in arr_list
            ]

        A_batched = _broadcast(A)
        B_batched = _broadcast(B)
        pA_batched = _broadcast(pA)
        pB_batched = _broadcast(pB)
        D_batched = _broadcast(D)

        agent = Agent(
            A_batched,
            B_batched,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            num_controls=self.num_controls,
            batch_size=self.batch_size,
            learn_A=learn_A,
            learn_B=learn_B,
            pA=pA_batched if learn_A else None,
            pB=pB_batched if learn_B else None,
            learning_mode=learning_mode,
            inference_algo=inference_algo,
            policy_len=policy_len,
        )

        params = {"A": A_batched, "B": B_batched, "D": D_batched}
        dependencies = {"A": self.A_dependencies, "B": self.B_dependencies}
        env = Env(params, dependencies)

        initial = {
            "pA": [np.array(val) for val in (agent.pA if learn_A else [])],
            "pB": [np.array(val) for val in (agent.pB if learn_B else [])],
        }

        return agent, env, initial

    def test_rollout_collects_time_series(self):
        agent, env, _ = self.build_agent_env()
        key = jr.PRNGKey(0)
        num_steps = 3

        last, info, _ = rollout(agent, env, num_steps, key)

        self.assertIn("observation", info)
        self.assertIn("action", info)
        self.assertIn("qs", info)

        action = np.asarray(info["action"])
        self.assertEqual(action.shape[0], agent.batch_size)
        self.assertEqual(action.shape[1], num_steps + 1)

        obs_series = np.asarray(info["observation"][0])
        self.assertEqual(obs_series.shape[0], agent.batch_size)
        self.assertEqual(obs_series.shape[1], num_steps + 1)

        qs_series = np.asarray(info["qs"][0])
        self.assertEqual(qs_series.shape[0], agent.batch_size)
        self.assertEqual(qs_series.shape[1], num_steps + 1)

        self.assertEqual(last["action"].shape[0], agent.batch_size)

    def test_online_learning_updates_A_during_scan(self):
        agent, env, initial = self.build_agent_env(learn_A=True, learning_mode="online")
        key = jr.PRNGKey(1)
        num_steps = 3

        last, info, _ = rollout(agent, env, num_steps, key)

        pA_series = np.asarray(info["pA"][0])
        self.assertEqual(pA_series.shape[0], agent.batch_size)
        self.assertEqual(pA_series.shape[1], num_steps + 1)
        self.assertFalse(
            np.allclose(pA_series[:, 0, ...], pA_series[:, -1, ...])
        )

        final_pA = np.asarray(last["agent"].pA[0])
        self.assertFalse(np.allclose(final_pA, initial["pA"][0]))

    def test_offline_learning_defers_A_update(self):
        agent, env, initial = self.build_agent_env(learn_A=True, learning_mode="offline")
        key = jr.PRNGKey(2)
        num_steps = 3

        last, info, _ = rollout(agent, env, num_steps, key)

        pA_series = np.asarray(info["pA"][0])
        self.assertEqual(pA_series.shape[0], agent.batch_size)
        self.assertEqual(pA_series.shape[1], num_steps + 1)
        self.assertTrue(
            np.allclose(pA_series[:, 0, ...], pA_series[:, -1, ...])
        )

        final_pA = np.asarray(last["agent"].pA[0])
        self.assertFalse(np.allclose(final_pA, initial["pA"][0]))

    def test_online_learning_updates_B(self):
        agent, env, initial = self.build_agent_env(
            learn_B=True, learning_mode="online", policy_len=2
        )
        key = jr.PRNGKey(3)
        num_steps = 4

        last, info, _ = rollout(agent, env, num_steps, key)

        pB_series = np.asarray(info["pB"][0])
        self.assertEqual(pB_series.shape[0], agent.batch_size)
        self.assertEqual(pB_series.shape[1], num_steps + 1)
        self.assertFalse(
            np.allclose(pB_series[:, 0, ...], pB_series[:, -1, ...])
        )

        final_pB = np.asarray(last["agent"].pB[0])
        self.assertFalse(np.allclose(final_pB, initial["pB"][0]))

    def test_rollout_supports_multiple_inference_algorithms(self):
        for algo, seed in (("vmp", 4), ("ovf", 5)):
            policy_len = 2 if algo == "ovf" else 1
            agent, env, _ = self.build_agent_env(
                inference_algo=algo, policy_len=policy_len, seed=seed
            )
            key = jr.PRNGKey(seed)
            num_steps = 2

            _, info, _ = rollout(agent, env, num_steps, key)

            actions = np.asarray(info["action"])
            self.assertEqual(actions.shape[0], agent.batch_size)
            self.assertEqual(actions.shape[1], num_steps + 1)

            G = np.asarray(info["G"])
            self.assertEqual(G.shape[0], agent.batch_size)
            self.assertEqual(G.shape[1], num_steps + 1)

    def test_rollout_with_custom_policy_search_and_initial_carry(self):
        agent, env, _ = self.build_agent_env(learn_A=True, policy_len=2)
        init_key = jr.PRNGKey(6)
        keys = jr.split(init_key, agent.batch_size + 1)
        init_obs, seeded_env = env.reset(keys[1:])
        qs0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)

        initial_carry = {
            "qs": qs0,
            "action": -jnp.ones(
                (agent.batch_size, agent.policies.shape[-1]), dtype=jnp.int32
            ),
            "observation": init_obs,
            "env": seeded_env,
            "agent": agent,
            "rng_key": keys[0],
        }

        def custom_policy_search(agent_obj, beliefs, rng):
            qpi, extras = default_policy_search(agent_obj, beliefs, rng)
            extras["custom_flag"] = jnp.ones((agent_obj.batch_size,), dtype=jnp.float32)
            return qpi, extras

        num_steps = 2
        _, info, _ = rollout(
            agent,
            seeded_env,
            num_steps,
            keys[0],
            initial_carry=initial_carry,
            policy_search=custom_policy_search,
        )

        obs_series = np.asarray(info["observation"][0])
        np.testing.assert_array_equal(
            np.squeeze(obs_series[0, 0]),
            np.squeeze(np.asarray(init_obs[0])),
        )

        custom_flag = np.asarray(info["custom_flag"])
        self.assertEqual(custom_flag.shape[:2], (agent.batch_size, num_steps + 1))
        self.assertTrue(np.allclose(custom_flag, 1.0))

    def test_offline_B_learning_matches_outer_products(self):
        num_obs = [2]
        num_states = [2]
        num_controls = [1]
        A_dependencies = [[0]]
        B_dependencies = [[0]]

        identity_A = [np.eye(num_obs[0], dtype=np.float32)]
        toggle_B = [np.zeros((num_states[0], num_states[0], num_controls[0]), dtype=np.float32)]
        toggle_B[0][:, :, 0] = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        initial_state = [np.array([1.0, 0.0], dtype=np.float32)]
        prior_pB = [np.ones_like(toggle_B[0])]

        def _broadcast(arr_list):
            return [
                jnp.broadcast_to(jnp.array(arr), (self.batch_size,) + arr.shape)
                for arr in arr_list
            ]

        A_batched = _broadcast(identity_A)
        B_batched = _broadcast(toggle_B)
        D_batched = _broadcast(initial_state)
        pB_batched = _broadcast(prior_pB)

        agent = Agent(
            A_batched,
            B_batched,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            num_controls=num_controls,
            batch_size=self.batch_size,
            learn_B=True,
            learning_mode="offline",
            pB=pB_batched,
        )

        params = {"A": A_batched, "B": B_batched, "D": D_batched}
        dependencies = {"A": A_dependencies, "B": B_dependencies}
        env = Env(params, dependencies)

        num_steps = 4
        rng_key = jr.PRNGKey(8)

        last, info, _ = rollout(agent, env, num_steps, rng_key)

        qs = info["qs"][0]
        self.assertEqual(qs.shape[0], self.batch_size)
        self.assertEqual(qs.shape[1], num_steps + 1)

        counts = jnp.einsum("bti,btj->ij", qs[:, 1:, :], qs[:, :-1, :])
        expected_pB = prior_pB[0] + counts[..., None]
        expected_B = expected_pB / expected_pB.sum(axis=0, keepdims=True)

        learned_pB = last["agent"].pB[0][0]
        learned_B = last["agent"].B[0][0]

        self.assertTrue(jnp.allclose(learned_pB, expected_pB, atol=1e-5))
        self.assertTrue(jnp.allclose(learned_B, expected_B, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
