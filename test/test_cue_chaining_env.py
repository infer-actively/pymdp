import unittest

import jax.numpy as jnp
import jax.random as jr

from pymdp.agent import Agent
from pymdp.envs.cue_chaining import CueChainingEnv
from pymdp.envs.rollout import rollout


class TestCueChainingEnv(unittest.TestCase):
    def test_shapes_and_dependencies(self):
        env = CueChainingEnv()

        self.assertEqual(len(env.A), 4)
        self.assertEqual(len(env.B), 3)
        self.assertEqual(len(env.D), 3)

        self.assertEqual(env.A[0].shape, (35, 35))
        self.assertEqual(env.A[1].shape, (5, 35, 4))
        self.assertEqual(env.A[2].shape, (3, 35, 4, 2))
        self.assertEqual(env.A[3].shape, (3, 35, 2))
        self.assertEqual(env.A_dependencies, [[0], [0, 1], [0, 1, 2], [0, 2]])

        self.assertEqual(env.B[0].shape, (35, 35, 5))
        self.assertEqual(env.B[1].shape, (4, 4, 1))
        self.assertEqual(env.B[2].shape, (2, 2, 1))
        self.assertEqual(env.B_dependencies, [[0], [1], [2]])

        self.assertEqual(env.D[0].shape, (35,))
        self.assertEqual(env.D[1].shape, (4,))
        self.assertEqual(env.D[2].shape, (2,))

    def test_cue_and_reward_likelihood_semantics(self):
        env = CueChainingEnv()
        cue1_idx = env.cue1_index

        for cue_state in range(env.num_cue2_states):
            self.assertTrue(bool(jnp.isclose(env.A[1][0, cue1_idx, cue_state], 0.0)))
            self.assertTrue(bool(jnp.isclose(env.A[1][1 + cue_state, cue1_idx, cue_state], 1.0)))

        non_cue1_idx = env.start_index
        self.assertTrue(bool(jnp.isclose(env.A[1][0, non_cue1_idx, 0], 1.0)))

        for cue_state, cue_loc_idx in enumerate(env.cue2_indices):
            for reward_state in range(env.num_reward_states):
                self.assertTrue(
                    bool(
                        jnp.isclose(
                            env.A[2][1 + reward_state, cue_loc_idx, cue_state, reward_state],
                            1.0,
                        )
                    )
                )

        mismatched_state = 1
        cue0_loc_idx = env.cue2_indices[0]
        self.assertTrue(bool(jnp.isclose(env.A[2][0, cue0_loc_idx, mismatched_state, 0], 1.0)))

        top_idx, bottom_idx = env.reward_indices
        self.assertTrue(bool(jnp.isclose(env.A[3][1, top_idx, 0], 1.0)))
        self.assertTrue(bool(jnp.isclose(env.A[3][2, top_idx, 1], 1.0)))
        self.assertTrue(bool(jnp.isclose(env.A[3][2, bottom_idx, 0], 1.0)))
        self.assertTrue(bool(jnp.isclose(env.A[3][1, bottom_idx, 1], 1.0)))

    def test_env_params_broadcast(self):
        env = CueChainingEnv()
        batch_size = 3
        env_params = env.generate_env_params(batch_size=batch_size)

        self.assertEqual(env_params["A"][0].shape[0], batch_size)
        self.assertEqual(env_params["B"][0].shape[0], batch_size)
        self.assertEqual(env_params["D"][0].shape[0], batch_size)

    def test_rollout_smoke(self):
        env = CueChainingEnv(cue2_state=3, reward_condition=1)
        C = [jnp.zeros((a.shape[0],), dtype=jnp.float32) for a in env.A]
        C[3] = C[3].at[1].set(2.0)
        C[3] = C[3].at[2].set(-4.0)

        agent = Agent(
            A=env.A,
            B=env.B,
            C=C,
            D=env.D,
            A_dependencies=env.A_dependencies,
            B_dependencies=env.B_dependencies,
            policy_len=4,
            batch_size=1,
        )

        num_timesteps = 6
        _, info = rollout(agent, env, num_timesteps=num_timesteps, rng_key=jr.PRNGKey(0))

        self.assertEqual(info["action"].shape, (1, num_timesteps + 1, 3))
        self.assertEqual(info["observation"][0].shape, (1, num_timesteps + 1, 1))
        self.assertEqual(info["observation"][1].shape, (1, num_timesteps + 1, 1))
        self.assertEqual(info["observation"][2].shape, (1, num_timesteps + 1, 1))
        self.assertEqual(info["observation"][3].shape, (1, num_timesteps + 1, 1))

        loc_actions = info["action"][0, :, 0]
        self.assertTrue(bool(jnp.all(loc_actions >= 0)))
        self.assertTrue(bool(jnp.all(loc_actions < len(env.ACTION_LABELS))))
