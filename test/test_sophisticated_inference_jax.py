#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests for sophisticated inference planning (JAX)."""

import unittest

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from pymdp import control, utils
from pymdp.agent import Agent
from pymdp.planning.si import si_policy_search


def _build_single_cue_model():
    num_locations = 4  # 0: start, 1: left, 2: right, 3: cue
    num_reward_states = 2  # 0: reward on left, 1: reward on right

    cue_obs = jnp.zeros((5, num_locations, num_reward_states), dtype=jnp.float32)
    reward_obs = jnp.zeros((3, num_locations, num_reward_states), dtype=jnp.float32)

    for loc in range(num_locations):
        for reward_state in range(num_reward_states):
            if loc == 0:
                cue_obs = cue_obs.at[0, loc, reward_state].set(1.0)
                reward_obs = reward_obs.at[0, loc, reward_state].set(1.0)
            elif loc == 3:
                cue_idx = 3 if reward_state == 0 else 4
                cue_obs = cue_obs.at[cue_idx, loc, reward_state].set(1.0)
                reward_obs = reward_obs.at[0, loc, reward_state].set(1.0)
            elif loc == 1:
                cue_obs = cue_obs.at[1, loc, reward_state].set(1.0)
                observation_idx = 1 if reward_state == 0 else 2
                reward_obs = reward_obs.at[observation_idx, loc, reward_state].set(1.0)
            elif loc == 2:
                cue_obs = cue_obs.at[2, loc, reward_state].set(1.0)
                observation_idx = 1 if reward_state == 1 else 2
                reward_obs = reward_obs.at[observation_idx, loc, reward_state].set(1.0)

    A = [cue_obs, reward_obs]
    A_dependencies = [[0, 1], [0, 1]]

    B_loc = utils.create_controllable_B(num_locations, num_locations)[0]
    B_reward = jnp.eye(num_reward_states, dtype=jnp.float32).reshape(
        num_reward_states, num_reward_states, 1
    )
    B = [B_loc, B_reward]
    B_dependencies = [[0], [1]]

    D = [
        jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([0.5, 0.5], dtype=jnp.float32),
    ]

    return A, B, A_dependencies, B_dependencies, D


def _build_dual_cue_model():
    num_locations = 5  # 0: start, 1: cue A, 2: cue B, 3: reward L, 4: reward R
    num_reward_states = 2
    num_distractor_states = 2

    loc_obs = jnp.eye(num_locations, dtype=jnp.float32)
    cue_a_obs = jnp.zeros((3, num_locations, num_reward_states), dtype=jnp.float32)
    cue_b_obs = jnp.zeros((3, num_locations, num_distractor_states), dtype=jnp.float32)
    reward_obs = jnp.zeros((3, num_locations, num_reward_states), dtype=jnp.float32)

    for loc in range(num_locations):
        for reward_state in range(num_reward_states):
            if loc == 1:
                cue_idx = 1 if reward_state == 0 else 2
                cue_a_obs = cue_a_obs.at[cue_idx, loc, reward_state].set(1.0)
            else:
                cue_a_obs = cue_a_obs.at[0, loc, reward_state].set(1.0)

            if loc == 3:
                observation_idx = 1 if reward_state == 0 else 2
                reward_obs = reward_obs.at[observation_idx, loc, reward_state].set(1.0)
            elif loc == 4:
                observation_idx = 1 if reward_state == 1 else 2
                reward_obs = reward_obs.at[observation_idx, loc, reward_state].set(1.0)
            else:
                reward_obs = reward_obs.at[0, loc, reward_state].set(1.0)

        for distractor_state in range(num_distractor_states):
            if loc == 2:
                cue_idx = 1 if distractor_state == 0 else 2
                cue_b_obs = cue_b_obs.at[cue_idx, loc, distractor_state].set(1.0)
            else:
                cue_b_obs = cue_b_obs.at[0, loc, distractor_state].set(1.0)

    A = [loc_obs, cue_a_obs, cue_b_obs, reward_obs]
    A_dependencies = [[0], [0, 1], [0, 2], [0, 1]]

    B_loc = utils.create_controllable_B(num_locations, num_locations)[0]
    B_reward = jnp.eye(num_reward_states, dtype=jnp.float32).reshape(
        num_reward_states, num_reward_states, 1
    )
    B_distractor = jnp.eye(num_distractor_states, dtype=jnp.float32).reshape(
        num_distractor_states, num_distractor_states, 1
    )
    B = [B_loc, B_reward, B_distractor]
    B_dependencies = [[0], [1], [2]]

    D = [
        jnp.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([0.5, 0.5], dtype=jnp.float32),
        jnp.array([0.5, 0.5], dtype=jnp.float32),
    ]

    return A, B, A_dependencies, B_dependencies, D


class TestSophisticatedInferenceJax(unittest.TestCase):
    def _run_si_search(self, agent, horizon):
        qs = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)
        search_fn = si_policy_search(
            horizon=horizon,
            max_nodes=512,
            max_branching=32,
            policy_prune_threshold=0.0,
            observation_prune_threshold=0.0,
            entropy_stop_threshold=-1.0,
            gamma=4.0,
            topk_obsspace=2,
        )
        q_pi, _ = search_fn(agent, qs=qs, rng_key=jr.PRNGKey(0))
        return q_pi, qs

    def _run_vanilla_search(self, agent):
        qs = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)
        q_pi, _ = agent.infer_policies(qs)
        return q_pi

    def test_si_accepts_costly_informative_cue(self):
        A, B, A_dependencies, B_dependencies, D = _build_single_cue_model()
        cue_cost = -2.0
        reward_value = 6.0
        punishment_value = -12.0

        C = [
            jnp.array([0.0, 0.0, 0.0, cue_cost, cue_cost], dtype=jnp.float32),
            jnp.array([0.0, reward_value, punishment_value], dtype=jnp.float32),
        ]

        agent = Agent(
            A,
            B,
            C=C,
            D=D,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            num_controls=[4, 1],
            policy_len=1,
        )

        q_pi, qs = self._run_si_search(agent, horizon=2)
        action_marginals = control.get_marginals(q_pi[0], agent.policies, agent.num_controls)

        cue_action = int(jnp.argmax(action_marginals[0]))
        self.assertEqual(cue_action, 3)

        cue_prob = float(action_marginals[0][3])
        left_prob = float(action_marginals[0][1])
        right_prob = float(action_marginals[0][2])
        self.assertGreater(cue_prob, left_prob)
        self.assertGreater(cue_prob, right_prob)

        qs_current = D
        cue_policy = [3, 0]
        qs_next = control.compute_expected_state(qs_current, B, cue_policy, B_dependencies)
        qo = control.compute_expected_obs(qs_next, A, A_dependencies)
        immediate_u = control.compute_expected_utility(qo, C)
        self.assertLess(float(immediate_u), 0.0)

        vanilla_agent = Agent(
            A,
            B,
            C=C,
            D=D,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            num_controls=[4, 1],
            policy_len=2,
            gamma=4.0,
            use_utility=True,
            use_states_info_gain=True,
        )
        q_pi_vanilla = self._run_vanilla_search(vanilla_agent)
        action_marginals_vanilla = control.get_marginals(
            q_pi_vanilla[0], vanilla_agent.policies, vanilla_agent.num_controls
        )
        stay_prob = float(action_marginals_vanilla[0][0])
        cue_prob = float(action_marginals_vanilla[0][3])
        self.assertLess(cue_prob, stay_prob)
        self.assertNotEqual(int(jnp.argmax(action_marginals_vanilla[0])), 3)

    def test_si_ignores_irrelevant_distractor_cue(self):
        A, B, A_dependencies, B_dependencies, D = _build_dual_cue_model()
        cue_cost = -2.0
        reward_value = 6.0
        punishment_value = -12.0

        C = [
            jnp.zeros(5, dtype=jnp.float32),
            jnp.array([0.0, cue_cost, cue_cost], dtype=jnp.float32),
            jnp.array([0.0, cue_cost, cue_cost], dtype=jnp.float32),
            jnp.array([0.0, reward_value, punishment_value], dtype=jnp.float32),
        ]

        agent = Agent(
            A,
            B,
            C=C,
            D=D,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            num_controls=[5, 1, 1],
            policy_len=1,
        )

        q_pi, _ = self._run_si_search(agent, horizon=2)
        action_marginals = control.get_marginals(q_pi[0], agent.policies, agent.num_controls)

        cue_a_prob = float(action_marginals[0][1])
        cue_b_prob = float(action_marginals[0][2])
        self.assertGreater(cue_a_prob, cue_b_prob)
        self.assertEqual(int(jnp.argmax(action_marginals[0])), 1)

        vanilla_agent = Agent(
            A,
            B,
            C=C,
            D=D,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            num_controls=[5, 1, 1],
            policy_len=2,
            gamma=4.0,
            use_utility=True,
            use_states_info_gain=True,
        )
        q_pi_vanilla = self._run_vanilla_search(vanilla_agent)
        action_marginals_vanilla = control.get_marginals(
            q_pi_vanilla[0], vanilla_agent.policies, vanilla_agent.num_controls
        )
        cue_a_prob = float(action_marginals_vanilla[0][1])
        cue_b_prob = float(action_marginals_vanilla[0][2])
        self.assertAlmostEqual(cue_a_prob, cue_b_prob, places=6)


if __name__ == "__main__":
    unittest.main()
