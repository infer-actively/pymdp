"""MCTS policy search through the canonical rollout loop.

`pymdp.planning.mcts` used to ship its own `rollout`, written against an older,
stateful `Env` API. It could not run (issue #427) and has been removed: MCTS is
run through `pymdp.envs.rollout.rollout` with
`policy_search=mcts_policy_search(...)`, as
`examples/experimental/sophisticated_inference/mcts_generalized_tmaze.ipynb`
does. Until this test, nothing in the suite imported `pymdp.planning.mcts`.
"""

import unittest

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np

from pymdp import utils
from pymdp.agent import Agent
from pymdp.envs.env import PymdpEnv
from pymdp.envs.rollout import rollout
from pymdp.planning.mcts import mcts_policy_search


class TestMCTSPolicySearchRollout(unittest.TestCase):
    def setUp(self):
        self.num_obs = [3]
        self.num_states = [2]
        self.num_controls = [2]
        self.A_dependencies = [[0]]
        self.B_dependencies = [[0]]
        self.batch_size = 2

    def build_agent_env(self, seed=0):
        A_key, B_key, D_key = jr.split(jr.PRNGKey(seed), 3)
        A = utils.random_A_array(
            A_key, self.num_obs, self.num_states, A_dependencies=self.A_dependencies
        )
        B = utils.random_B_array(
            B_key,
            self.num_states,
            self.num_controls,
            B_dependencies=self.B_dependencies,
        )
        D = utils.random_factorized_categorical(D_key, self.num_states)

        def _broadcast(arr_list):
            return [
                jnp.broadcast_to(jnp.array(arr), (self.batch_size,) + arr.shape)
                for arr in arr_list
            ]

        A_batched = _broadcast(A)
        B_batched = _broadcast(B)
        D_batched = _broadcast(D)

        agent = Agent(
            A_batched,
            B_batched,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            num_controls=self.num_controls,
            batch_size=self.batch_size,
        )
        env = PymdpEnv(
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
        )
        env_params = {"A": A_batched, "B": B_batched, "D": D_batched}
        return agent, env, env_params

    def test_rollout_with_mcts_policy_search(self):
        agent, env, env_params = self.build_agent_env()
        num_steps = 3

        _, info = rollout(
            agent,
            env,
            num_steps,
            jr.PRNGKey(1),
            policy_search=mcts_policy_search(max_depth=2, num_simulations=8),
            env_params=env_params,
        )

        # one policy posterior per batch element and timestep (the initial
        # step plus `num_steps`), over the agent's policies
        qpi = np.asarray(info["qpi"])
        self.assertEqual(qpi.shape[:2], (self.batch_size, num_steps + 1))
        self.assertEqual(qpi.shape[-1], agent.E.shape[-1])
        self.assertTrue(np.all(np.isfinite(qpi)))
        np.testing.assert_allclose(qpi.sum(axis=-1), 1.0, atol=1e-5)


class TestMCTSSequenceBeliefs(unittest.TestCase):
    def test_mcts_plans_from_latest_belief_in_window(self):
        # Ring of 3 locations, fully observed; actions 0=left, 1=stay, 2=right.
        # The agent walked 0 -> 1 -> 2 and prefers location 2, so it should stay.
        num_states = 3
        eye = jnp.eye(num_states)
        B = jnp.stack([jnp.roll(eye, -1, 0), eye, jnp.roll(eye, 1, 0)], axis=-1)
        agent = Agent(
            [eye],
            [B],
            C=[jnp.array([0.0, 0.0, 4.0])],
            inference_algo="mmp",
            inference_horizon=3,
            gamma=4.0,
        )
        qs = agent.infer_states(
            [jnp.array([[0, 1, 2]])],
            empirical_prior=agent.D,
            past_actions=jnp.array([[[2], [2]]]),
        )
        self.assertEqual(qs[0].shape, (1, 3, num_states))

        search = mcts_policy_search(max_depth=1, num_simulations=16)
        q_window, _ = search(agent, qs, jr.PRNGKey(0))
        q_latest, _ = search(agent, jtu.tree_map(lambda x: x[:, -1:], qs), jr.PRNGKey(0))
        q_default, _ = agent.infer_policies(qs)

        np.testing.assert_allclose(q_window, q_latest)
        self.assertEqual(int(jnp.argmax(q_default[0])), 1)
        self.assertEqual(int(jnp.argmax(q_window[0])), 1)


if __name__ == "__main__":
    unittest.main()
