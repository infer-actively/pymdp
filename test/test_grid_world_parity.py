#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests comparing GridWorld implementations."""

import unittest

import numpy as np
import jax.numpy as jnp
from jax import random as jr

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
            batch_size=1,
            initial_position=initial_state,
        )

        legacy_A = legacy_env.get_likelihood_dist()
        legacy_B = legacy_env.get_transition_dist()

        jax_A = np.asarray(jax_env.params["A"][0][0])
        jax_B = np.asarray(jax_env.params["B"][0][0])
        jax_D = np.asarray(jax_env.params["D"][0][0])

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
                batch_size=1,
                initial_position=initial_state,
            )
            batch_size = jax_env.params["D"][0].shape[0]

            # Reset both envs deterministically to the same starting state
            legacy_env.reset(init_state=initial_state)
            key, reset_key = jr.split(key)
            reset_keys = jr.split(reset_key, batch_size + 1)
            jax_obs, jax_env_state = jax_env.reset(
                reset_keys[1:], state=[jnp.array([initial_state], dtype=jnp.int32)]
            )

            # Verify initial observation alignment
            jax_initial_obs = int(np.squeeze(np.asarray(jax_obs[0])))
            self.assertEqual(jax_initial_obs, legacy_env.state)
            self.assertEqual(int(np.asarray(jax_env_state.state[0])[0]), legacy_env.state)

            # Step through the shared action sequence
            for action in actions:
                expected_state = legacy_env.step(action)

                key, step_key = jr.split(key)
                step_keys = jr.split(step_key, batch_size + 1)
                action_array = jnp.array([[action]], dtype=jnp.int32)
                jax_obs, jax_env_state = jax_env_state.step(
                    step_keys[1:], actions=action_array
                )

                jax_state = int(np.asarray(jax_env_state.state[0])[0])
                jax_observation = int(np.squeeze(np.asarray(jax_obs[0])))

                self.assertEqual(jax_state, expected_state)
                self.assertEqual(jax_observation, expected_state)


if __name__ == "__main__":
    unittest.main()
