import unittest

import jax.numpy as jnp

from pymdp.envs.tmaze import TMaze, SimplifiedTMaze


class TestTMazeVariants(unittest.TestCase):
    def test_classic_shapes(self):
        env = TMaze()

        self.assertEqual(len(env.A), 3)
        self.assertEqual(env.A[0].shape, (5, 5))
        self.assertEqual(env.A[1].shape, (3, 5, 2))
        self.assertEqual(env.A[2].shape, (3, 5, 2))
        self.assertEqual(env.A_dependencies, [[0], [0, 1], [0, 1]])

        self.assertEqual(env.B[0].shape, (5, 5, 5))
        self.assertEqual(env.B[1].shape, (2, 2, 1))

        self.assertEqual(env.D[0].shape, (5,))
        self.assertEqual(env.D[1].shape, (2,))

    def test_simplified_shapes(self):
        env = SimplifiedTMaze()

        self.assertEqual(len(env.A), 2)
        self.assertEqual(env.A[0].shape, (5, 4, 2))
        self.assertEqual(env.A[1].shape, (3, 4, 2))
        self.assertEqual(env.A_dependencies, [[0, 1], [0, 1]])

        self.assertEqual(env.B[0].shape, (4, 4, 4))
        self.assertEqual(env.B[1].shape, (2, 2, 1))

        self.assertEqual(env.D[0].shape, (4,))
        self.assertEqual(env.D[1].shape, (2,))

    def test_classic_cue_validity(self):
        cue_validity = 0.7
        env = TMaze(cue_validity=cue_validity)
        cue_loc = 3

        self.assertTrue(bool(jnp.isclose(env.A[2][1, cue_loc, 0], cue_validity)))
        self.assertTrue(bool(jnp.isclose(env.A[2][2, cue_loc, 0], 1 - cue_validity)))
        self.assertTrue(bool(jnp.isclose(env.A[2][2, cue_loc, 1], cue_validity)))
        self.assertTrue(bool(jnp.isclose(env.A[2][1, cue_loc, 1], 1 - cue_validity)))

    def test_simplified_cue_validity(self):
        cue_validity = 0.6
        env = SimplifiedTMaze(cue_validity=cue_validity)
        cue_loc = 3

        self.assertTrue(bool(jnp.isclose(env.A[0][3, cue_loc, 0], cue_validity)))
        self.assertTrue(bool(jnp.isclose(env.A[0][4, cue_loc, 0], 1 - cue_validity)))
        self.assertTrue(bool(jnp.isclose(env.A[0][4, cue_loc, 1], cue_validity)))
        self.assertTrue(bool(jnp.isclose(env.A[0][3, cue_loc, 1], 1 - cue_validity)))

    def test_reward_outcomes_independent(self):
        env = TMaze(reward_probability=0.8, punishment_probability=0.3, dependent_outcomes=False)
        reward_condition = 0
        correct_loc = 1
        wrong_loc = 2

        self.assertTrue(bool(jnp.isclose(env.A[1][1, correct_loc, reward_condition], 0.8)))
        self.assertTrue(bool(jnp.isclose(env.A[1][0, correct_loc, reward_condition], 0.2)))
        self.assertTrue(bool(jnp.isclose(env.A[1][2, correct_loc, reward_condition], 0.0)))

        self.assertTrue(bool(jnp.isclose(env.A[1][2, wrong_loc, reward_condition], 0.3)))
        self.assertTrue(bool(jnp.isclose(env.A[1][0, wrong_loc, reward_condition], 0.7)))
        self.assertTrue(bool(jnp.isclose(env.A[1][1, wrong_loc, reward_condition], 0.0)))

    def test_reward_outcomes_dependent(self):
        env = TMaze(reward_probability=0.8, dependent_outcomes=True)
        reward_condition = 0
        correct_loc = 1
        wrong_loc = 2

        self.assertTrue(bool(jnp.isclose(env.A[1][1, correct_loc, reward_condition], 0.8)))
        self.assertTrue(bool(jnp.isclose(env.A[1][2, correct_loc, reward_condition], 0.2)))
        self.assertTrue(bool(jnp.isclose(env.A[1][0, correct_loc, reward_condition], 0.0)))

        self.assertTrue(bool(jnp.isclose(env.A[1][2, wrong_loc, reward_condition], 0.8)))
        self.assertTrue(bool(jnp.isclose(env.A[1][1, wrong_loc, reward_condition], 0.2)))
        self.assertTrue(bool(jnp.isclose(env.A[1][0, wrong_loc, reward_condition], 0.0)))

    def test_simplified_reward_and_punishment_probabilities(self):
        env = SimplifiedTMaze(reward_probability=0.9, punishment_probability=0.2, dependent_outcomes=False)
        reward_condition = 1
        correct_loc = 2
        wrong_loc = 1

        self.assertTrue(bool(jnp.isclose(env.A[1][1, correct_loc, reward_condition], 0.9)))
        self.assertTrue(bool(jnp.isclose(env.A[1][0, correct_loc, reward_condition], 0.1)))
        self.assertTrue(bool(jnp.isclose(env.A[1][2, correct_loc, reward_condition], 0.0)))

        self.assertTrue(bool(jnp.isclose(env.A[1][2, wrong_loc, reward_condition], 0.2)))
        self.assertTrue(bool(jnp.isclose(env.A[1][0, wrong_loc, reward_condition], 0.8)))
        self.assertTrue(bool(jnp.isclose(env.A[1][1, wrong_loc, reward_condition], 0.0)))

    def test_classic_transition_connectivity(self):
        env = TMaze()
        B_loc = env.B[0]

        centre = 0
        left = 1
        right = 2
        cue = 3
        middle = 4

        self.assertTrue(bool(jnp.isclose(B_loc[cue, centre, cue], 1.0)))
        self.assertTrue(bool(jnp.isclose(B_loc[centre, centre, left], 1.0)))
        self.assertTrue(bool(jnp.isclose(B_loc[left, centre, left], 0.0)))

        self.assertTrue(bool(jnp.isclose(B_loc[middle, left, middle], 1.0)))
        self.assertTrue(bool(jnp.isclose(B_loc[left, left, right], 1.0)))
        self.assertTrue(bool(jnp.isclose(B_loc[right, left, right], 0.0)))

        self.assertTrue(bool(jnp.isclose(B_loc[centre, cue, centre], 1.0)))
        self.assertTrue(bool(jnp.isclose(B_loc[cue, cue, left], 1.0)))
        self.assertTrue(bool(jnp.isclose(B_loc[left, cue, left], 0.0)))

        self.assertTrue(bool(jnp.allclose(jnp.sum(B_loc, axis=0), 1.0)))

    def test_simplified_transition_connectivity(self):
        env = SimplifiedTMaze()
        B_loc = env.B[0]

        self.assertTrue(bool(jnp.isclose(B_loc[1, 0, 1], 1.0)))
        self.assertTrue(bool(jnp.isclose(B_loc[0, 0, 1], 0.0)))
        self.assertTrue(bool(jnp.isclose(B_loc[2, 3, 2], 1.0)))
        self.assertTrue(bool(jnp.isclose(B_loc[3, 3, 3], 1.0)))

        self.assertTrue(bool(jnp.allclose(jnp.sum(B_loc, axis=0), 1.0)))

    def test_render_accepts_singleton_discrete_and_categorical_observations(self):
        env = TMaze()

        discrete_obs = [
            jnp.array([[[0]]], dtype=jnp.int32),
            jnp.array([[[0]]], dtype=jnp.int32),
            jnp.array([[[0]]], dtype=jnp.int32),
        ]
        categorical_obs = [
            jnp.array([[[1.0, 0.0, 0.0, 0.0, 0.0]]]),
            jnp.array([[[1.0, 0.0, 0.0]]]),
            jnp.array([[[1.0, 0.0, 0.0]]]),
        ]

        discrete_img = env.render(discrete_obs, mode="rgb_array")
        categorical_img = env.render(categorical_obs, mode="rgb_array")

        self.assertIsNotNone(discrete_img)
        self.assertIsNotNone(categorical_img)
