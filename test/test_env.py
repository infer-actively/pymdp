#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for the PymdpEnv."""

import unittest

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import vmap

from pymdp.agent import Agent
from pymdp.distribution import compile_model
from pymdp.envs.env import PymdpEnv, make
from pymdp.envs.rollout import rollout


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


def _compile_multi_control_model():
    """Model with a factor moved by two controls and an uncontrolled factor.

    `pos` moves 0, 1 or 2 steps around a ring of 3 locations depending on
    `move` and `boost`; `clock` flips every step with no control.
    """
    model = compile_model(
        {
            "observations": {"o": {"elements": ["a", "b", "c"], "depends_on": ["pos"]}},
            "controls": {
                "move": {"elements": ["stay", "fwd"]},
                "boost": {"elements": ["off", "on"]},
            },
            "states": {
                "pos": {
                    "elements": ["a", "b", "c"],
                    "depends_on": ["pos"],
                    "controlled_by": ["move", "boost"],
                },
                "clock": {"elements": ["t0", "t1"], "depends_on": ["clock"], "controlled_by": []},
            },
        }
    )
    model.A["o"].data = np.eye(3)
    for s in range(3):
        model.B["pos"][s, s, "stay", :] = 1.0
        model.B["pos"][(s + 1) % 3, s, "fwd", "off"] = 1.0
        model.B["pos"][(s + 2) % 3, s, "fwd", "on"] = 1.0
    model.B["clock"].data = np.array([[0.0, 1.0], [1.0, 0.0]])
    model.D["pos"].data = np.array([1.0, 0.0, 0.0])
    model.D["clock"].data = np.array([1.0, 0.0])
    return model


class TestPymdpEnv(unittest.TestCase):
    def setUp(self):
        self.A_dependencies = [[0]]
        self.B_dependencies = [[0]]

    def test_reset_respects_state_override(self):
        """Reset should honor a provided state override and emit matching obs."""
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
        """Stepping with action=None should keep the state unchanged."""
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
        """Explicit env_params should override the env's internal A/B/D."""
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
        """generate_env_params should broadcast to a requested batch size."""
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
        """Vmap over keys should match manual per-key stepping."""
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
        """Vmap over env_params should match manual per-param stepping."""
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
        """Vmap over keys/state/action should match manual stepping."""
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

    def test_make_env_params_respects_input_batch(self):
        """make() should preserve input shapes and env_params batching."""
        params = _make_deterministic_params()
        env, env_params = make(
            params["A"],
            params["B"],
            params["D"],
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            make_env_params=False,
        )
        self.assertIsNone(env_params)
        self.assertEqual(env.A[0].shape, params["A"][0].shape)
        self.assertEqual(env.B[0].shape, params["B"][0].shape)
        self.assertEqual(env.D[0].shape, params["D"][0].shape)

        env, env_params = make(
            params["A"],
            params["B"],
            params["D"],
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            make_env_params=True,
        )
        self.assertIsNotNone(env_params)
        self.assertEqual(env_params["A"][0].shape, params["A"][0].shape)
        self.assertEqual(env_params["B"][0].shape, params["B"][0].shape)
        self.assertEqual(env_params["D"][0].shape, params["D"][0].shape)
        self.assertTrue(bool(jnp.array_equal(env_params["A"][0], params["A"][0])))
        self.assertTrue(bool(jnp.array_equal(env_params["B"][0], params["B"][0])))
        self.assertTrue(bool(jnp.array_equal(env_params["D"][0], params["D"][0])))

        batch_size = 4
        params_batched = jtu.tree_map(
            lambda x: jnp.broadcast_to(jnp.asarray(x), (batch_size,) + x.shape),
            params,
        )
        env, env_params = make(
            params_batched["A"],
            params_batched["B"],
            params_batched["D"],
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            make_env_params=True,
        )
        self.assertEqual(env_params["A"][0].shape, params_batched["A"][0].shape)
        self.assertEqual(env_params["B"][0].shape, params_batched["B"][0].shape)
        self.assertEqual(env_params["D"][0].shape, params_batched["D"][0].shape)
        self.assertEqual(env_params["A"][0].shape[0], batch_size)
        self.assertEqual(env.A[0].shape[0], batch_size)
        self.assertEqual(env.B[0].shape[0], batch_size)
        self.assertEqual(env.D[0].shape[0], batch_size)
        self.assertTrue(
            bool(jnp.array_equal(env_params["A"][0], params_batched["A"][0]))
        )
        self.assertTrue(
            bool(jnp.array_equal(env_params["B"][0], params_batched["B"][0]))
        )
        self.assertTrue(
            bool(jnp.array_equal(env_params["D"][0], params_batched["D"][0]))
        )

        env, env_params = make(
            params_batched["A"],
            params_batched["B"],
            params_batched["D"],
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            make_env_params=False,
        )
        self.assertIsNone(env_params)
        self.assertEqual(env.A[0].shape[0], batch_size)
        self.assertEqual(env.B[0].shape[0], batch_size)
        self.assertEqual(env.D[0].shape[0], batch_size)


    def test_B_action_dependencies_flattened_like_agent(self):
        """Control axes in B should be flattened exactly as the Agent does."""
        model = _compile_multi_control_model()
        env = PymdpEnv(**model)
        agent = Agent(**model)

        self.assertEqual([b.shape for b in env.B], [(3, 3, 4), (2, 2, 1)])
        for env_b, agent_b in zip(env.B, agent.B):
            self.assertTrue(bool(jnp.array_equal(env_b, agent_b[0])))

        # Flat action 3 is (move="fwd", boost="on"): two steps from "a" to "c".
        state = [jnp.array(0.0), jnp.array(0.0)]
        _, next_state = env.step(jr.PRNGKey(1), state, jnp.array([3, 0]))
        self.assertEqual(next_state[0].shape, ())
        self.assertEqual(float(next_state[0]), 2.0)
        self.assertEqual(float(next_state[1]), 1.0)

    def test_make_env_params_use_flattened_B(self):
        """make() should return env_params["B"] in the same layout as env.B."""
        model = _compile_multi_control_model()
        env, env_params = make(**model, make_env_params=True)

        self.assertEqual([b.shape for b in env_params["B"]], [(3, 3, 4), (2, 2, 1)])
        for env_b, param_b in zip(env.B, env_params["B"]):
            self.assertTrue(bool(jnp.array_equal(env_b, param_b)))

    def test_rollout_with_B_action_dependencies(self):
        """rollout() should run on a compiled model with multi-control and uncontrolled factors."""
        model = _compile_multi_control_model()
        env = PymdpEnv(**model)
        agent = Agent(**model, action_selection="stochastic")

        num_timesteps = 4
        _, info = rollout(agent, env, num_timesteps=num_timesteps, rng_key=jr.PRNGKey(0))

        pos = np.asarray(info["env_state"][0][0])
        clock = np.asarray(info["env_state"][1][0])
        actions = np.asarray(info["action"][0, :, 0])
        steps_per_action = np.array([0, 0, 1, 2])  # flat index of (move, boost) in C order
        for t in range(num_timesteps):
            self.assertEqual(pos[t + 1], (pos[t] + steps_per_action[actions[t]]) % 3)
            self.assertEqual(clock[t + 1], 1 - clock[t])


if __name__ == "__main__":
    unittest.main()
