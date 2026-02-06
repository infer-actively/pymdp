#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for the jittable rollout loop."""

import unittest
import warnings

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import lax, vmap

from pymdp.agent import Agent
from pymdp.envs.env import PymdpEnv
from pymdp.envs.rollout import (
    rollout,
    default_policy_search,
    infer_and_plan,
    _append_to_window,
    MAX_WINDOWED_HISTORY_WITHOUT_HORIZON,
)
from pymdp import control
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
        inference_horizon=None,
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
            inference_horizon=inference_horizon,
            policy_len=policy_len,
        )

        env_params = {"A": A_batched, "B": B_batched, "D": D_batched}
        env = PymdpEnv(A_dependencies=self.A_dependencies, B_dependencies=self.B_dependencies)

        initial = {
            "pA": [np.array(val) for val in (agent.pA if learn_A else [])],
            "pB": [np.array(val) for val in (agent.pB if learn_B else [])],
        }

        return agent, env, env_params, initial

    def manual_windowed_rollout_reference(self, agent, env, num_steps, rng_key, env_params=None):
        batch_size = agent.batch_size
        is_sequence_method = agent.inference_algo in {"mmp", "vmp"}
        use_smoothing_online_windows = (
            (agent.inference_algo in {"ovf", "exact"})
            and (agent.learning_mode == "online")
            and (agent.learn_A or agent.learn_B)
        )
        self.assertTrue(is_sequence_method or use_smoothing_online_windows)

        history_len = (
            agent.inference_horizon
            if agent.inference_horizon is not None
            else min(num_steps + 1, MAX_WINDOWED_HISTORY_WITHOUT_HORIZON)
        )
        action_history_len = max(history_len - 1, 0)

        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        observation, env_state = vmap(env.reset)(keys[1:], env_params=env_params)
        action = -jnp.ones((batch_size, agent.policies.policy_arr.shape[-1]), dtype=jnp.int32)

        qs = jtu.tree_map(
            lambda x: jnp.broadcast_to(jnp.expand_dims(x, axis=1), (x.shape[0], history_len, x.shape[-1])),
            agent.D,
        )

        def _init_observation_history(obs):
            if obs.ndim == 1:
                obs = jnp.expand_dims(obs, axis=1)
            history_shape = (obs.shape[0], history_len) + obs.shape[2:]
            if agent.categorical_obs and obs.shape[-1] > 0:
                hist = jnp.zeros(history_shape, dtype=obs.dtype)
            else:
                hist = jnp.full(history_shape, -1, dtype=obs.dtype)
            start_idx = (0, history_len - 1) + (0,) * (hist.ndim - 2)
            return lax.dynamic_update_slice(hist, obs[:, :1, ...], start_idx)

        observation_hist = jtu.tree_map(_init_observation_history, observation)
        action_hist = -jnp.ones(
            (batch_size, action_history_len, agent.policies.policy_arr.shape[-1]),
            dtype=jnp.int32,
        )
        valid_steps = jnp.array(1, dtype=jnp.int32)
        empirical_prior = agent.D if is_sequence_method else None

        info_steps = []
        for _ in range(num_steps + 1):
            keys = jr.split(rng_key, batch_size + 2)
            rng_key = keys[0]

            k = int(valid_steps)
            obs_k = jtu.tree_map(lambda x: x[:, history_len - k :, ...], observation_hist)
            qs_prev_k = jtu.tree_map(lambda x: x[:, history_len - k :, ...], qs)
            actions_k = action_hist[:, history_len - k : history_len - 1, :]

            if is_sequence_method:
                past_actions_k = actions_k if k > 1 else None
                updated_agent, action_next, qs_k, xtra = infer_and_plan(
                    agent,
                    qs_prev_k,
                    obs_k,
                    action,
                    keys[1],
                    policy_search=default_policy_search,
                    past_actions=past_actions_k,
                    empirical_prior=empirical_prior,
                    learning_observations=obs_k,
                    learning_actions=past_actions_k,
                )
                qs = jtu.tree_map(
                    lambda prev, new: prev.at[:, history_len - k :, ...].set(new),
                    qs,
                    qs_k,
                )
                qs_latest = jtu.tree_map(lambda x: x[:, -1, ...], qs_k)
                if (k == history_len) and (history_len > 1):
                    qs_window_start = jtu.tree_map(lambda x: x[:, 0, ...], qs_k)
                    action_window_start = actions_k[:, 0, :]
                    propagate = lambda beliefs, B, act: control.compute_expected_state(
                        beliefs,
                        B,
                        act,
                        B_dependencies=updated_agent.B_dependencies,
                    )
                    empirical_prior = vmap(propagate)(
                        qs_window_start,
                        updated_agent.B,
                        action_window_start,
                    )
            else:
                updated_agent, action_next, qs_k, xtra = infer_and_plan(
                    agent,
                    qs_prev_k,
                    observation,
                    action,
                    keys[1],
                    policy_search=default_policy_search,
                    learning_observations=obs_k,
                    learning_actions=actions_k,
                    learning_beliefs=qs_prev_k,
                )
                qs_latest = jtu.tree_map(lambda x: x[:, -1, ...], qs_k)
                qs = jtu.tree_map(_append_to_window, qs, qs_k)

            observation_next, env_state_next = vmap(env.step)(
                keys[2:], env_state, action_next, env_params=env_params
            )
            observation_hist = jtu.tree_map(_append_to_window, observation_hist, observation_next)
            action_hist = _append_to_window(action_hist, action_next)
            valid_steps = jnp.minimum(valid_steps + 1, history_len)

            if updated_agent.learn_A:
                xtra["A"] = updated_agent.A
                xtra["pA"] = updated_agent.pA
            if updated_agent.learn_B:
                xtra["B"] = updated_agent.B
                xtra["pB"] = updated_agent.pB

            info_t = {
                "qs": qs_latest,
                "env_state": env_state,
                "observation": observation,
                "action": action_next,
            }
            info_t.update(xtra)
            info_steps.append(info_t)

            agent = updated_agent
            action = action_next
            observation = observation_next
            env_state = env_state_next

        info = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=1), *info_steps)
        last = {
            "action": action,
            "observation": observation,
            "observation_hist": observation_hist,
            "action_hist": action_hist,
            "valid_steps": valid_steps,
            "qs": qs,
            "env_state": env_state,
            "agent": agent,
            "rng_key": rng_key,
        }
        if is_sequence_method:
            last["empirical_prior"] = empirical_prior
        return last, info

    def test_rollout_collects_time_series(self):
        agent, env, env_params, _ = self.build_agent_env()
        key = jr.PRNGKey(0)
        num_steps = 3

        last, info = rollout(agent, env, num_steps, key, env_params=env_params)

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

    def test_rollout_env_state_matches_manual_steps(self):
        num_obs = [2]
        num_states = [2]
        num_controls = [2]
        A_dependencies = [[0]]
        B_dependencies = [[0]]
        batch_size = 2

        A = [jnp.eye(num_obs[0], dtype=jnp.float32)]
        b = jnp.zeros((num_states[0], num_states[0], num_controls[0]), dtype=jnp.float32)
        b = b.at[:, :, 0].set(jnp.eye(num_states[0], dtype=jnp.float32))
        b = b.at[:, :, 1].set(jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32))
        B = [b]
        D = [jnp.array([1.0, 0.0], dtype=jnp.float32)]

        def _broadcast(arr_list):
            return [
                jnp.broadcast_to(jnp.array(arr), (batch_size,) + arr.shape)
                for arr in arr_list
            ]

        A_batched = _broadcast(A)
        B_batched = _broadcast(B)
        D_batched = _broadcast(D)

        agent = Agent(
            A_batched,
            B_batched,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            num_controls=num_controls,
            batch_size=batch_size,
            D=D_batched,
        )

        env_params = {"A": A_batched, "B": B_batched, "D": D_batched}
        env = PymdpEnv(A_dependencies=A_dependencies, B_dependencies=B_dependencies)

        num_steps = 3
        key = jr.PRNGKey(9)

        last, info = rollout(agent, env, num_steps, key, env_params=env_params)

        env_state_series = info["env_state"][0]
        action_series = info["action"]

        current_state = [env_state_series[:, 0]]
        step_fn = lambda k, st, act, params: env.step(k, st, act, env_params=params)

        key = jr.PRNGKey(10)
        for t in range(num_steps + 1):
            key, step_key = jr.split(key)
            step_keys = jr.split(step_key, batch_size)
            _, next_state = vmap(step_fn, in_axes=(0, 0, 0, 0))(
                step_keys, current_state, action_series[:, t, :], env_params
            )

            if t < num_steps:
                np.testing.assert_array_equal(
                    np.asarray(next_state[0]), np.asarray(env_state_series[:, t + 1])
                )
            else:
                np.testing.assert_array_equal(
                    np.asarray(next_state[0]), np.asarray(last["env_state"][0])
                )

            current_state = next_state

    def test_online_learning_updates_A_during_scan(self):
        agent, env, env_params, initial = self.build_agent_env(learn_A=True, learning_mode="online")
        key = jr.PRNGKey(1)
        num_steps = 3

        last, info = rollout(agent, env, num_steps, key, env_params=env_params)

        pA_series = np.asarray(info["pA"][0])
        self.assertEqual(pA_series.shape[0], agent.batch_size)
        self.assertEqual(pA_series.shape[1], num_steps + 1)
        self.assertFalse(
            np.allclose(pA_series[:, 0, ...], pA_series[:, -1, ...])
        )

        final_pA = np.asarray(last["agent"].pA[0])
        self.assertFalse(np.allclose(final_pA, initial["pA"][0]))

    def test_offline_learning_defers_A_update(self):
        agent, env, env_params, initial = self.build_agent_env(learn_A=True, learning_mode="offline")
        key = jr.PRNGKey(2)
        num_steps = 3

        last, info = rollout(agent, env, num_steps, key, env_params=env_params)
        pA_series = np.asarray(info["pA"][0])
        self.assertEqual(pA_series.shape[0], agent.batch_size)
        self.assertEqual(pA_series.shape[1], num_steps + 1)
        self.assertTrue(
            np.allclose(pA_series[:, 0, ...], pA_series[:, -1, ...])
        )

        final_pA = np.asarray(last["agent"].pA[0])
        self.assertFalse(np.allclose(final_pA, initial["pA"][0]))

    def test_online_learning_updates_B(self):
        agent, env, env_params, initial = self.build_agent_env(
            learn_B=True, learning_mode="online", policy_len=2
        )
        key = jr.PRNGKey(3)
        num_steps = 4

        last, info = rollout(agent, env, num_steps, key, env_params=env_params)

        pB_series = np.asarray(info["pB"][0])
        self.assertEqual(pB_series.shape[0], agent.batch_size)
        self.assertEqual(pB_series.shape[1], num_steps + 1)
        self.assertFalse(
            np.allclose(pB_series[:, 0, ...], pB_series[:, -1, ...])
        )

        final_pB = np.asarray(last["agent"].pB[0])
        self.assertFalse(np.allclose(final_pB, initial["pB"][0]))

    def test_online_learning_updates_B_for_sequence_inference(self):
        num_steps = 4
        for algo, seed in (("mmp", 31), ("vmp", 32)):
            with self.subTest(inference_algo=algo):
                agent, env, env_params, initial = self.build_agent_env(
                    learn_B=True,
                    learning_mode="online",
                    inference_algo=algo,
                    inference_horizon=3,
                    policy_len=2,
                    seed=seed,
                )
                key = jr.PRNGKey(seed)

                last, info = rollout(agent, env, num_steps, key, env_params=env_params)

                self.assertIn("pB", info)
                pB_series = np.asarray(info["pB"][0])
                self.assertEqual(pB_series.shape[0], agent.batch_size)
                self.assertEqual(pB_series.shape[1], num_steps + 1)
                self.assertFalse(
                    np.allclose(pB_series[:, 0, ...], pB_series[:, -1, ...])
                )

                final_pB = np.asarray(last["agent"].pB[0])
                self.assertFalse(np.allclose(final_pB, initial["pB"][0]))

    def test_online_learning_updates_for_smoothing_inference_with_horizon(self):
        num_steps = 5
        for algo, seed in (("ovf", 33), ("exact", 34)):
            with self.subTest(inference_algo=algo, learn="B"):
                agent, env, env_params, initial = self.build_agent_env(
                    learn_B=True,
                    learning_mode="online",
                    inference_algo=algo,
                    inference_horizon=3,
                    policy_len=2,
                    seed=seed,
                )
                key = jr.PRNGKey(seed)

                last, info = rollout(agent, env, num_steps, key, env_params=env_params)

                self.assertIn("pB", info)
                pB_series = np.asarray(info["pB"][0])
                self.assertEqual(pB_series.shape[0], agent.batch_size)
                self.assertEqual(pB_series.shape[1], num_steps + 1)
                self.assertFalse(
                    np.allclose(pB_series[:, 0, ...], pB_series[:, -1, ...])
                )

                final_pB = np.asarray(last["agent"].pB[0])
                self.assertFalse(np.allclose(final_pB, initial["pB"][0]))

            with self.subTest(inference_algo=algo, learn="A"):
                agent, env, env_params, initial = self.build_agent_env(
                    learn_A=True,
                    learning_mode="online",
                    inference_algo=algo,
                    inference_horizon=3,
                    policy_len=2,
                    seed=seed + 100,
                )
                key = jr.PRNGKey(seed + 100)

                last, info = rollout(agent, env, num_steps, key, env_params=env_params)

                self.assertIn("pA", info)
                pA_series = np.asarray(info["pA"][0])
                self.assertEqual(pA_series.shape[0], agent.batch_size)
                self.assertEqual(pA_series.shape[1], num_steps + 1)
                self.assertFalse(
                    np.allclose(pA_series[:, 0, ...], pA_series[:, -1, ...])
                )

                final_pA = np.asarray(last["agent"].pA[0])
                self.assertFalse(np.allclose(final_pA, initial["pA"][0]))

    def test_online_learning_updates_A_only_for_sequence_inference(self):
        num_steps = 5
        for algo, seed in (("mmp", 111), ("vmp", 112)):
            with self.subTest(inference_algo=algo):
                agent, env, env_params, initial = self.build_agent_env(
                    learn_A=True,
                    learn_B=False,
                    learning_mode="online",
                    inference_algo=algo,
                    inference_horizon=4,
                    policy_len=2,
                    seed=seed,
                )
                key = jr.PRNGKey(seed)

                last, info = rollout(agent, env, num_steps, key, env_params=env_params)

                self.assertIn("pA", info)
                self.assertNotIn("pB", info)
                pA_series = np.asarray(info["pA"][0])
                self.assertEqual(pA_series.shape[0], agent.batch_size)
                self.assertEqual(pA_series.shape[1], num_steps + 1)
                self.assertFalse(
                    np.allclose(pA_series[:, 0, ...], pA_series[:, -1, ...])
                )

                final_pA = np.asarray(last["agent"].pA[0])
                self.assertFalse(np.allclose(final_pA, initial["pA"][0]))

    def test_sequence_rollout_updates_empirical_prior_with_finite_horizon(self):
        num_steps = 4
        for algo, seed in (("mmp", 41), ("vmp", 42)):
            with self.subTest(inference_algo=algo):
                agent, env, env_params, _ = self.build_agent_env(
                    inference_algo=algo,
                    inference_horizon=3,
                    policy_len=2,
                    seed=seed,
                )
                key = jr.PRNGKey(seed)

                _, info = rollout(agent, env, num_steps, key, env_params=env_params)

                prior_series = np.asarray(info["empirical_prior"][0])
                self.assertEqual(prior_series.shape[0], agent.batch_size)
                self.assertEqual(prior_series.shape[1], num_steps + 1)
                self.assertFalse(
                    np.allclose(prior_series[:, 0, ...], prior_series[:, -1, ...])
                )

    def test_sequence_rollout_keeps_empirical_prior_fixed_during_warmup(self):
        horizon = 3
        num_steps = 5
        for algo, seed in (("mmp", 51), ("vmp", 52)):
            with self.subTest(inference_algo=algo):
                agent, env, env_params, _ = self.build_agent_env(
                    inference_algo=algo,
                    inference_horizon=horizon,
                    policy_len=2,
                    seed=seed,
                )
                key = jr.PRNGKey(seed)

                _, info = rollout(agent, env, num_steps, key, env_params=env_params)
                prior_series = np.asarray(info["empirical_prior"][0])[0]  # (T, Ns)
                initial_prior = np.asarray(agent.D[0])[0]

                self.assertTrue(
                    np.allclose(
                        prior_series[:horizon],
                        np.broadcast_to(initial_prior, (horizon, initial_prior.shape[-1])),
                    )
                )

    def test_rollout_caps_history_without_inference_horizon(self):
        num_steps = MAX_WINDOWED_HISTORY_WITHOUT_HORIZON + 2
        agent, env, env_params, _ = self.build_agent_env(
            inference_algo="mmp",
            inference_horizon=None,
            policy_len=2,
            seed=61,
        )
        key = jr.PRNGKey(61)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            last, _ = rollout(agent, env, num_steps, key, env_params=env_params)

        self.assertTrue(
            any("capping rollout history" in str(w.message).lower() for w in caught)
        )
        self.assertEqual(last["qs"][0].shape[1], MAX_WINDOWED_HISTORY_WITHOUT_HORIZON)

    def test_rollout_supports_multiple_inference_algorithms(self):
        for algo, seed in (("mmp", 3), ("vmp", 4), ("ovf", 5), ("exact", 6)):
            agent, env, env_params, _ = self.build_agent_env(
                inference_algo=algo,
                inference_horizon=2 if algo in ("mmp", "vmp") else None,
                policy_len=2,
                seed=seed,
            )
            key = jr.PRNGKey(seed)
            num_steps = 2

            _, info = rollout(agent, env, num_steps, key, env_params=env_params)

            actions = np.asarray(info["action"])
            self.assertEqual(actions.shape[0], agent.batch_size)
            self.assertEqual(actions.shape[1], num_steps + 1)

            G = np.asarray(info["G"])
            self.assertEqual(G.shape[0], agent.batch_size)
            self.assertEqual(G.shape[1], num_steps + 1)

    def test_rollout_modes_for_ovf_and_exact(self):
        modes = (
            {
                "name": "planning_only",
                "learn_A": False,
                "learn_B": False,
                "learning_mode": "online",
            },
            {
                "name": "online_learning",
                "learn_A": False,
                "learn_B": True,
                "learning_mode": "online",
            },
            {
                "name": "offline_learning",
                "learn_A": False,
                "learn_B": True,
                "learning_mode": "offline",
            },
        )

        num_steps = 4
        for algo, seed_base in (("ovf", 40), ("exact", 80)):
            for mode_idx, mode in enumerate(modes):
                with self.subTest(algo=algo, mode=mode["name"]):
                    agent, env, env_params, initial = self.build_agent_env(
                        learn_A=mode["learn_A"],
                        learn_B=mode["learn_B"],
                        learning_mode=mode["learning_mode"],
                        inference_algo=algo,
                        policy_len=2,
                        seed=seed_base + mode_idx,
                    )
                    key = jr.PRNGKey(seed_base + 100 + mode_idx)
                    last, info = rollout(agent, env, num_steps, key, env_params=env_params)

                    if mode["name"] == "planning_only":
                        self.assertNotIn("pA", info)
                        self.assertNotIn("pB", info)
                        continue

                    self.assertNotIn("pA", info)
                    self.assertIn("pB", info)
                    pB_series = np.asarray(info["pB"][0])

                    if mode["name"] == "online_learning":
                        self.assertFalse(
                            np.allclose(pB_series[:, 0, ...], pB_series[:, -1, ...])
                        )
                    else:
                        self.assertTrue(
                            np.allclose(pB_series[:, 0, ...], pB_series[:, -1, ...])
                        )

                    final_pB = np.asarray(last["agent"].pB[0])
                    self.assertFalse(np.allclose(final_pB, initial["pB"][0]))

    def _assert_rollout_matches_manual_reference(self, algo, seed, include_empirical_prior=False):
        num_steps = 5
        kwargs = dict(
            learn_A=True,
            learn_B=True,
            learning_mode="online",
            inference_algo=algo,
            inference_horizon=4,
            policy_len=2,
            seed=seed,
        )
        agent, env, env_params, _ = self.build_agent_env(**kwargs)
        ref_agent, ref_env, ref_env_params, _ = self.build_agent_env(**kwargs)
        key = jr.PRNGKey(seed + 10_000)

        last, info = rollout(agent, env, num_steps, key, env_params=env_params)
        ref_last, ref_info = self.manual_windowed_rollout_reference(
            ref_agent, ref_env, num_steps, key, env_params=ref_env_params
        )

        np.testing.assert_allclose(np.asarray(info["action"]), np.asarray(ref_info["action"]), atol=1e-5)
        np.testing.assert_allclose(np.asarray(info["qs"][0]), np.asarray(ref_info["qs"][0]), atol=1e-5)
        if include_empirical_prior:
            np.testing.assert_allclose(
                np.asarray(info["empirical_prior"][0]),
                np.asarray(ref_info["empirical_prior"][0]),
                atol=1e-5,
            )
        np.testing.assert_allclose(np.asarray(info["pA"][0]), np.asarray(ref_info["pA"][0]), atol=1e-5)
        np.testing.assert_allclose(np.asarray(info["pB"][0]), np.asarray(ref_info["pB"][0]), atol=1e-5)
        np.testing.assert_allclose(np.asarray(last["agent"].pA[0]), np.asarray(ref_last["agent"].pA[0]), atol=1e-5)
        np.testing.assert_allclose(np.asarray(last["agent"].pB[0]), np.asarray(ref_last["agent"].pB[0]), atol=1e-5)

    def test_sequence_rollout_matches_manual_window_branch_reference(self):
        for algo, seed in (("mmp", 501), ("vmp", 502)):
            with self.subTest(inference_algo=algo):
                self._assert_rollout_matches_manual_reference(algo, seed, include_empirical_prior=True)

    def test_smoothing_rollout_matches_manual_window_branch_reference(self):
        for algo, seed in (("ovf", 601), ("exact", 602)):
            with self.subTest(inference_algo=algo):
                self._assert_rollout_matches_manual_reference(algo, seed)


    def test_rollout_with_custom_policy_search_and_initial_carry(self):
        agent, env, env_params, _ = self.build_agent_env(learn_A=True, policy_len=2)
        init_key = jr.PRNGKey(6)
        keys = jr.split(init_key, agent.batch_size + 1)
        init_obs, initial_state = vmap(env.reset)(keys[1:], env_params=env_params)
        qs0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)

        initial_carry = {
            "qs": qs0,
            "action": -jnp.ones(
                (agent.batch_size, agent.policies.policy_arr.shape[-1]), dtype=jnp.int32
            ),
            "observation": init_obs,
            "env_state": initial_state,
            "agent": agent,
            "rng_key": keys[0],
        }

        def custom_policy_search(agent_obj, beliefs, rng):
            qpi, extras = default_policy_search(agent_obj, beliefs, rng)
            extras["custom_flag"] = jnp.ones((agent_obj.batch_size,), dtype=jnp.float32)
            return qpi, extras

        num_steps = 2
        _, info = rollout(
            agent,
            env,
            num_steps,
            keys[0],
            initial_carry=initial_carry,
            policy_search=custom_policy_search,
            env_params=env_params,
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

        env_params = {"A": A_batched, "B": B_batched, "D": D_batched}
        env = PymdpEnv(A_dependencies=A_dependencies, B_dependencies=B_dependencies)

        num_steps = 4
        rng_key = jr.PRNGKey(8)

        last, info = rollout(agent, env, num_steps, rng_key, env_params=env_params)

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
