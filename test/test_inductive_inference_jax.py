#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for inductive inference."""

import unittest

import numpy as np
from jax.nn import one_hot
import jax.numpy as jnp

import pymdp.control as ctl_jax


def _chain_transition(num_states: int) -> jnp.ndarray:
    """Single-action chain with a self-loop on the goal state."""
    B = jnp.zeros((num_states, num_states, 1), dtype=jnp.float32)
    for prev in range(num_states - 1):
        B = B.at[prev + 1, prev, 0].set(1.0)
    B = B.at[num_states - 1, num_states - 1, 0].set(1.0)
    return B


def _advance_or_stay_transition() -> jnp.ndarray:
    """Two-action transition: advance along a chain or stay put."""
    B = jnp.zeros((3, 3, 2), dtype=jnp.float32)
    B = B.at[:, :, 0].set(
        jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=jnp.float32,
        )
    )
    B = B.at[:, :, 1].set(jnp.eye(3, dtype=jnp.float32))
    return B


def _manual_chain_I() -> list[jnp.ndarray]:
    return [
        jnp.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=jnp.float32,
        )
    ]

class TestInductiveInferenceJax(unittest.TestCase):

    def test_generate_I_matrix_matches_chain_reachability(self):
        H = [jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)]
        B = [_chain_transition(3)]

        I = ctl_jax.generate_I_matrix(H, B, threshold=0.5, depth=4)

        expected = jnp.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=jnp.float32,
        )
        self.assertTrue(np.array_equal(np.array(I[0]), np.array(expected)))

    def test_generate_I_matrix_respects_threshold_pruning(self):
        H = [jnp.array([0.0, 1.0], dtype=jnp.float32)]

        reachable = jnp.zeros((2, 2, 2), dtype=jnp.float32)
        reachable = reachable.at[:, :, 0].set(
            jnp.array([[0.49, 0.0], [0.51, 1.0]], dtype=jnp.float32)
        )
        reachable = reachable.at[:, :, 1].set(
            jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
        )

        not_reachable = jnp.zeros((2, 2, 2), dtype=jnp.float32)
        not_reachable = not_reachable.at[:, :, 0].set(
            jnp.array([[0.51, 0.0], [0.49, 1.0]], dtype=jnp.float32)
        )
        not_reachable = not_reachable.at[:, :, 1].set(
            jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
        )

        I_reachable = ctl_jax.generate_I_matrix(H, [reachable], threshold=0.5, depth=2)
        I_not_reachable = ctl_jax.generate_I_matrix(H, [not_reachable], threshold=0.5, depth=2)

        self.assertEqual(float(I_reachable[0][1, 0]), 1.0)
        self.assertEqual(float(I_not_reachable[0][1, 0]), 0.0)

    def test_generate_I_matrix_respects_depth_truncation(self):
        H = [jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)]
        B = [_chain_transition(4)]

        I_depth_3 = ctl_jax.generate_I_matrix(H, B, threshold=0.5, depth=3)
        I_depth_4 = ctl_jax.generate_I_matrix(H, B, threshold=0.5, depth=4)

        self.assertEqual(float(I_depth_3[0][-1, 0]), 0.0)
        self.assertEqual(float(I_depth_4[0][-1, 0]), 1.0)

    def test_calc_inductive_value_on_path_zero_and_off_path_logeps(self):
        I = _manual_chain_I()
        qs = [one_hot(0, 3)]
        log_eps = float(jnp.log(1e-3))

        on_path = ctl_jax.calc_inductive_value_t(qs, [one_hot(1, 3)], I, epsilon=1e-3)
        off_path = ctl_jax.calc_inductive_value_t(qs, [one_hot(0, 3)], I, epsilon=1e-3)

        self.assertAlmostEqual(float(on_path), 0.0, places=6)
        self.assertAlmostEqual(float(off_path), log_eps, places=6)

    def test_calc_inductive_value_scales_with_off_path_mass(self):
        I = _manual_chain_I()
        qs = [one_hot(0, 3)]
        qs_next = [jnp.array([0.25, 0.75, 0.0], dtype=jnp.float32)]

        value = ctl_jax.calc_inductive_value_t(qs, qs_next, I, epsilon=1e-3)
        expected = 0.25 * float(jnp.log(1e-3))

        self.assertAlmostEqual(float(value), expected, places=6)

    def test_calc_inductive_value_is_zero_when_goal_unreachable(self):
        I_unreachable = [
            jnp.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=jnp.float32,
            )
        ]
        qs = [one_hot(0, 3)]
        qs_next = [jnp.array([0.4, 0.6, 0.0], dtype=jnp.float32)]

        value = ctl_jax.calc_inductive_value_t(qs, qs_next, I_unreachable, epsilon=1e-3)

        self.assertAlmostEqual(float(value), 0.0, places=6)

    def test_calc_inductive_value_depends_on_map_current_state(self):
        I = _manual_chain_I()
        qs_next = [jnp.array([0.25, 0.75, 0.0], dtype=jnp.float32)]

        qs_a = [jnp.array([0.51, 0.49, 0.0], dtype=jnp.float32)]
        qs_b = [jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)]
        qs_c = [jnp.array([0.49, 0.51, 0.0], dtype=jnp.float32)]

        value_a = ctl_jax.calc_inductive_value_t(qs_a, qs_next, I, epsilon=1e-3)
        value_b = ctl_jax.calc_inductive_value_t(qs_b, qs_next, I, epsilon=1e-3)
        value_c = ctl_jax.calc_inductive_value_t(qs_c, qs_next, I, epsilon=1e-3)

        self.assertAlmostEqual(float(value_a), float(value_b), places=6)
        self.assertAlmostEqual(float(value_c), float(jnp.log(1e-3)), places=6)

    def test_one_step_policy_ranking_prefers_goal_directed_action(self):
        A = [jnp.eye(3, dtype=jnp.float32)]
        B = [_advance_or_stay_transition()]
        C = [jnp.zeros((3,), dtype=jnp.float32)]
        E = jnp.array([0.5, 0.5], dtype=jnp.float32)
        qs_init = [one_hot(0, 3)]
        I = _manual_chain_I()
        policy_matrix = jnp.array([[[0]], [[1]]], dtype=jnp.int32)

        q_pi, neg_efe = ctl_jax.update_posterior_policies_inductive(
            policy_matrix,
            qs_init,
            A,
            B,
            C,
            E,
            None,
            None,
            [[0]],
            [[0]],
            I,
            gamma=1.0,
            inductive_epsilon=1e-3,
            use_utility=False,
            use_states_info_gain=False,
            use_param_info_gain=False,
            use_inductive=True,
        )

        expected_neg_efe = np.array([0.0, jnp.log(1e-3)], dtype=np.float32)
        expected_q_pi = np.array([1.0 / (1.0 + 1e-3), 1e-3 / (1.0 + 1e-3)], dtype=np.float32)

        self.assertTrue(np.allclose(np.array(neg_efe), expected_neg_efe, atol=1e-6))
        self.assertTrue(np.allclose(np.array(q_pi), expected_q_pi, atol=1e-6))

    def test_compute_neg_efe_policy_inductive_matches_non_inductive_when_disabled(self):
        A = [jnp.eye(3, dtype=jnp.float32)]
        B = [_advance_or_stay_transition()]
        C = [jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)]
        qs_init = [one_hot(0, 3)]
        policy = jnp.array([[0]], dtype=jnp.int32)
        I = _manual_chain_I()

        neg_efe = ctl_jax.compute_neg_efe_policy(
            qs_init,
            A,
            B,
            C,
            None,
            None,
            [[0]],
            [[0]],
            policy,
            use_utility=True,
            use_states_info_gain=False,
            use_param_info_gain=False,
        )
        neg_efe_inductive_disabled = ctl_jax.compute_neg_efe_policy_inductive(
            qs_init,
            A,
            B,
            C,
            None,
            None,
            [[0]],
            [[0]],
            I,
            policy,
            use_utility=True,
            use_states_info_gain=False,
            use_param_info_gain=False,
            use_inductive=False,
        )

        self.assertAlmostEqual(float(neg_efe), float(neg_efe_inductive_disabled), places=6)

    def test_multistep_inductive_scoring_stays_anchored_to_qs_init(self):
        A = [jnp.eye(3, dtype=jnp.float32)]
        B = [_advance_or_stay_transition()]
        C = [jnp.zeros((3,), dtype=jnp.float32)]
        qs_init = [one_hot(0, 3)]
        I = _manual_chain_I()

        advance_then_stay = jnp.array([[0], [1]], dtype=jnp.int32)
        advance_twice = jnp.array([[0], [0]], dtype=jnp.int32)

        score_a = ctl_jax.compute_neg_efe_policy_inductive(
            qs_init,
            A,
            B,
            C,
            None,
            None,
            [[0]],
            [[0]],
            I,
            advance_then_stay,
            use_utility=False,
            use_states_info_gain=False,
            use_param_info_gain=False,
            use_inductive=True,
        )
        score_b = ctl_jax.compute_neg_efe_policy_inductive(
            qs_init,
            A,
            B,
            C,
            None,
            None,
            [[0]],
            [[0]],
            I,
            advance_twice,
            use_utility=False,
            use_states_info_gain=False,
            use_param_info_gain=False,
            use_inductive=True,
        )

        self.assertAlmostEqual(float(score_a), float(score_b), places=6)
        self.assertAlmostEqual(float(score_a), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
