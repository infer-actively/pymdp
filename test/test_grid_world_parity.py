#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests comparing GridWorld implementations."""

import unittest

import numpy as np
import jax.numpy as jnp
from jax import random as jr, vmap
from jax import tree_util as jtu

from pymdp.envs.grid_world import GridWorld
from pymdp.legacy.envs.grid_worlds import GridWorldEnv


class TestGridWorldParity(unittest.TestCase):
    """Ensure the JAX GridWorld matches the legacy NumPy environment."""

    def test_jax_matches_legacy_transition_and_observation(self):
        """JAX GridWorld should reproduce legacy A, B, and one-hot D."""
        shape = (3, 4)
        initial_state = 5  # linear index within the grid

        legacy_env = GridWorldEnv(shape=list(shape), init_state=initial_state)
        jax_env = GridWorld(
            shape=shape,
            include_stay=True,
            success_prob=1.0,
            initial_position=initial_state,
        )

        legacy_A = legacy_env.get_likelihood_dist()
        legacy_B = legacy_env.get_transition_dist()

        jax_A = np.asarray(jax_env.A[0])
        jax_B = np.asarray(jax_env.B[0])
        jax_D = np.asarray(jax_env.D[0])

        # Transition dynamics and observation likelihood should match exactly
        np.testing.assert_array_equal(jax_A, legacy_A)
        np.testing.assert_array_equal(jax_B, legacy_B)

        # Initial distribution should remain a one-hot at the specified start state
        expected_D = np.zeros(np.prod(shape))
        expected_D[initial_state] = 1.0
        np.testing.assert_array_equal(jax_D, expected_D)

        # Sanity checks on shared metadata
        self.assertEqual(legacy_env.n_states, legacy_A.shape[0])
        self.assertEqual(legacy_env.n_states, jax_A.shape[0])
        self.assertEqual(legacy_env.n_control, jax_B.shape[-1])

    def test_step_outputs_match(self):
        """Stepping both envs should yield identical states and observations."""
        shape = (3, 3)
        num_states = shape[0] * shape[1]
        actions = [
            GridWorld.UP,
            GridWorld.RIGHT,
            GridWorld.DOWN,
            GridWorld.LEFT,
            GridWorld.STAY,
            GridWorld.DOWN,
            GridWorld.RIGHT,
        ]

        key = jr.PRNGKey(0)

        for initial_state in range(num_states):
            legacy_env = GridWorldEnv(shape=list(shape), init_state=initial_state)
            jax_env = GridWorld(
                shape=shape,
                include_stay=True,
                success_prob=1.0,
                initial_position=initial_state,
            )

            # Reset both envs deterministically to the same starting state
            legacy_env.reset(init_state=initial_state)
            key, reset_key = jr.split(key)

            jax_obs, jax_state = jax_env.reset(
                reset_key, state=[jnp.array(initial_state, dtype=jnp.int32)]
            )

            # Verify initial observation alignment
            jax_initial_obs = int(np.squeeze(np.asarray(jax_obs[0])))
            self.assertEqual(jax_initial_obs, legacy_env.state)
            self.assertEqual(int(np.asarray(jax_state[0])), legacy_env.state)

            # Step through the shared action sequence
            for action in actions:
                expected_state = legacy_env.step(action)

                key, step_key = jr.split(key)
                action_array = [jnp.array(action, dtype=jnp.int32)]
                jax_obs, jax_state = jax_env.step(step_key, jax_state, action_array)

                jax_state_int = int(np.asarray(jax_state[0]))
                jax_observation = int(np.squeeze(np.asarray(jax_obs[0])))

                self.assertEqual(jax_state_int, expected_state)
                self.assertEqual(jax_observation, expected_state)

    def test_batched_jax_env_matches_series_of_numpy_envs(self):
        """
        Stepping a series of individual NumPy GridWorldEnvs should match a batch of JAX GridWorlds, using the `env_params` argument
        to represent the different environments.
        """
        shape = (3, 3)
        batch_size = 4
        initial_states = [0, 2, 4, 7]
        actions = [
            GridWorld.UP,
            GridWorld.RIGHT,
            GridWorld.DOWN,
            GridWorld.LEFT,
            GridWorld.STAY,
        ]

        # Build a set of legacy envs with distinct initial states
        legacy_envs = []
        for init_state in initial_states:
            env = GridWorldEnv(shape=list(shape), init_state=init_state)
            env.reset(init_state=init_state)
            legacy_envs.append(env)

        # Build per-env parameters for the JAX GridWorld and batch them
        per_env_jax = [
            GridWorld(
                shape=shape,
                include_stay=True,
                success_prob=1.0,
                initial_position=init_state,
            )
            for init_state in initial_states
        ]

        env_param_list = [env.generate_env_params() for env in per_env_jax]
        batched_env_params = jtu.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *env_param_list
        )

        # Use a single JAX env instance and feed batched env_params into vmapped reset/step
        jax_env = GridWorld(shape=shape, include_stay=True, success_prob=1.0)

        key = jr.PRNGKey(0)
        key, reset_key = jr.split(key)
        reset_keys = jr.split(reset_key, batch_size)

        reset_fn = vmap(
            lambda k, params: jax_env.reset(k, env_params=params), in_axes=(0, 0)
        )
        jax_obs, jax_state = reset_fn(reset_keys, batched_env_params)

        # Initial alignment
        legacy_states = np.array([env.state for env in legacy_envs])
        np.testing.assert_array_equal(
            np.squeeze(np.asarray(jax_obs[0])), legacy_states
        )
        np.testing.assert_array_equal(np.asarray(jax_state[0]), legacy_states)

        step_fn = vmap(
            lambda k, st, params, act: jax_env.step(k, st, [act], env_params=params),
            in_axes=(0, 0, 0, None),
        )

        # Step through the action sequence and compare after each step
        for action in actions:
            legacy_states = np.array([env.step(action) for env in legacy_envs])

            key, step_key = jr.split(key)
            step_keys = jr.split(step_key, batch_size)
            jax_obs, jax_state = step_fn(step_keys, jax_state, batched_env_params, action)

            jax_state_flat = np.asarray(jax_state[0])
            jax_obs_flat = np.squeeze(np.asarray(jax_obs[0]))

            np.testing.assert_array_equal(jax_state_flat, legacy_states)
            np.testing.assert_array_equal(jax_obs_flat, legacy_states)


if __name__ == "__main__":
    unittest.main()
