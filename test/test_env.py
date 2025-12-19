#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for the PymdpEnv."""

import unittest

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import vmap

from pymdp.envs.env import PymdpEnv


def _stack_params(params_list):
    return jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *params_list)


def _make_deterministic_params(toggle_action_one=True):
    A = [jnp.eye(2, dtype=jnp.float32)]

    b = jnp.zeros((2, 2, 2), dtype=jnp.float32)
    b = b.at[:, :, 0].set(jnp.eye(2, dtype=jnp.float32))
    if toggle_action_one:
        b = b.at[:, :, 1].set(
            jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
        )
    else:
        b = b.at[:, :, 1].set(jnp.eye(2, dtype=jnp.float32))

    B = [b]
    D = [jnp.array([1.0, 0.0], dtype=jnp.float32)]
    return {"A": A, "B": B, "D": D}


def _make_stochastic_params():
    A = [jnp.array([[0.9, 0.2], [0.1, 0.8]], dtype=jnp.float32)]
    B = [
        jnp.array(
            [
                [[0.25, 0.7], [0.6, 0.1]],
                [[0.75, 0.3], [0.4, 0.9]],
            ],
            dtype=jnp.float32,
        )
    ]
    D = [jnp.array([0.5, 0.5], dtype=jnp.float32)]
    return {"A": A, "B": B, "D": D}


class TestPymdpEnv(unittest.TestCase):
    def setUp(self):
        self.A_dependencies = [[0]]
        self.B_dependencies = [[0]]

    def test_reset_respects_state_override(self):
        params = _make_deterministic_params()
        env = PymdpEnv(
            A=params["A"],
            B=params["B"],
            D=params["D"],
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
        )

        key = jr.PRNGKey(0)
        state = [jnp.array(1.0, dtype=jnp.float32)]

        obs, state_out = env.reset(key, state=state)

        self.assertTrue(bool(jnp.array_equal(state_out[0], state[0])))
        self.assertTrue(
            bool(
                jnp.array_equal(
                    jnp.squeeze(obs[0]), jnp.array(1.0, dtype=obs[0].dtype)
                )
            )
        )

    def test_step_action_none_keeps_state(self):
        params = _make_deterministic_params()
        env = PymdpEnv(
            A=params["A"],
            B=params["B"],
            D=params["D"],
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
        )

        key = jr.PRNGKey(1)
        state = [jnp.array(0.0, dtype=jnp.float32)]

        obs, state_out = env.step(key, state, action=None)

        self.assertTrue(bool(jnp.array_equal(state_out[0], state[0])))
        self.assertTrue(
            bool(
                jnp.array_equal(
                    jnp.squeeze(obs[0]), jnp.array(0.0, dtype=obs[0].dtype)
                )
            )
        )

    def test_env_params_override_defaults(self):
        default_params = _make_deterministic_params(toggle_action_one=False)
        env = PymdpEnv(
            A=default_params["A"],
            B=default_params["B"],
            D=default_params["D"],
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
        )

        override_params = _make_deterministic_params(toggle_action_one=False)
        override_params["D"] = [jnp.array([0.0, 1.0], dtype=jnp.float32)]

        key = jr.PRNGKey(2)
        obs, state = env.reset(key, env_params=override_params)

        self.assertTrue(
            bool(
                jnp.array_equal(state[0], jnp.array(1.0, dtype=state[0].dtype))
            )
        )
        self.assertTrue(
            bool(
                jnp.array_equal(
                    jnp.squeeze(obs[0]), jnp.array(1.0, dtype=obs[0].dtype)
                )
            )
        )

    def test_generate_env_params_batches(self):
        params = _make_deterministic_params()
        env = PymdpEnv(
            A=params["A"],
            B=params["B"],
            D=params["D"],
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
        )

        env_params = env.generate_env_params()
        self.assertEqual(env_params["A"][0].shape, params["A"][0].shape)

        batch_size = 3
        env_params_batched = env.generate_env_params(batch_size=batch_size)
        self.assertEqual(env_params_batched["A"][0].shape[0], batch_size)
        self.assertEqual(env_params_batched["B"][0].shape[0], batch_size)
        self.assertEqual(env_params_batched["D"][0].shape[0], batch_size)

    def test_vmap_over_keys_matches_manual(self):
        params = _make_stochastic_params()
        env = PymdpEnv(
            A=params["A"],
            B=params["B"],
            D=params["D"],
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
        )

        state = [jnp.array(0.0, dtype=jnp.float32)]
        action = jnp.array([0], dtype=jnp.int32)
        keys = jr.split(jr.PRNGKey(3), 6)

        step_fn = vmap(lambda k: env.step(k, state, action))
        obs_vmap, state_vmap = step_fn(keys)

        manual_obs = []
        manual_state = []
        for key in keys:
            obs, st = env.step(key, state, action)
            manual_obs.append(obs)
            manual_state.append(st)

        stacked_obs = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *manual_obs)
        stacked_state = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *manual_state)

        self.assertTrue(bool(jnp.array_equal(obs_vmap[0], stacked_obs[0])))
        self.assertTrue(bool(jnp.array_equal(state_vmap[0], stacked_state[0])))

        self.assertGreater(int(jnp.unique(state_vmap[0]).size), 1)

    def test_vmap_over_env_params_matches_manual(self):
        params_stay = _make_deterministic_params(toggle_action_one=False)
        params_toggle = _make_deterministic_params(toggle_action_one=True)

        env = PymdpEnv(
            A_dependencies=self.A_dependencies, B_dependencies=self.B_dependencies
        )
        state = [jnp.array(0.0, dtype=jnp.float32)]
        action = jnp.array([1], dtype=jnp.int32)
        key = jr.PRNGKey(4)

        param_list = [params_stay, params_toggle]
        batched_params = _stack_params(param_list)

        step_fn = vmap(lambda params: env.step(key, state, action, env_params=params))
        obs_vmap, state_vmap = step_fn(batched_params)

        manual_obs = []
        manual_state = []
        for params in param_list:
            obs, st = env.step(key, state, action, env_params=params)
            manual_obs.append(obs)
            manual_state.append(st)

        stacked_obs = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *manual_obs)
        stacked_state = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *manual_state)

        self.assertTrue(bool(jnp.array_equal(obs_vmap[0], stacked_obs[0])))
        self.assertTrue(bool(jnp.array_equal(state_vmap[0], stacked_state[0])))

        self.assertTrue(bool(jnp.not_equal(state_vmap[0][0], state_vmap[0][1])))

    def test_vmap_over_state_action_and_keys(self):
        params = _make_deterministic_params(toggle_action_one=True)
        env = PymdpEnv(
            A_dependencies=self.A_dependencies, B_dependencies=self.B_dependencies
        )

        state = [jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)]
        actions = jnp.array([[0], [1], [1]], dtype=jnp.int32)
        keys = jr.split(jr.PRNGKey(5), actions.shape[0])

        step_fn = vmap(
            lambda k, st, act: env.step(k, st, act, env_params=params),
            in_axes=(0, 0, 0),
        )
        obs_vmap, state_vmap = step_fn(keys, state, actions)

        manual_obs = []
        manual_state = []
        for idx in range(actions.shape[0]):
            obs, st = env.step(keys[idx], [state[0][idx]], actions[idx], env_params=params)
            manual_obs.append(obs)
            manual_state.append(st)

        stacked_obs = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *manual_obs)
        stacked_state = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *manual_state)

        self.assertTrue(bool(jnp.array_equal(obs_vmap[0], stacked_obs[0])))
        self.assertTrue(bool(jnp.array_equal(state_vmap[0], stacked_state[0])))


if __name__ == "__main__":
    unittest.main()
