#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import jax.numpy as jnp
import jax.random as jr
from pymdp.agent import Agent
from pymdp.envs import TMaze
from pymdp.envs.rollout import rollout
from pymdp.likelihoods import evolve_trials


class TestEvolveTrials(unittest.TestCase):
    def test_evolve_trials_with_different_batch_sizes(self):
        """Test that evolve_trials works with different batch sizes"""
        for batch_size in [1, 3]:
            env = TMaze(batch_size=batch_size)

            C_gm = [
                jnp.zeros((batch_size, 5)),
                jnp.broadcast_to(jnp.array([0., 1., -1.]), (batch_size, 3)),
                jnp.zeros((batch_size, 3))
            ]

            agent = Agent(
                env.params['A'], env.params['B'], C_gm, env.params['D'],
                A_dependencies=env.dependencies['A'],
                B_dependencies=env.dependencies['B'],
                control_fac_idx=[0],
                batch_size=batch_size
            )

            # Run rollout
            num_timesteps = 2
            rng_key = jr.PRNGKey(0)
            _, info, _ = rollout(agent, env, num_timesteps, rng_key)

            # Use rollout data directly
            probs, _, _ = evolve_trials(agent, info)

            # Verify correct shape
            assert probs.shape[0] == num_timesteps + 1
            assert probs.shape[1] == batch_size

            # Verify probabilities sum to 1 over actions
            prob_sums = jnp.sum(probs, axis=-1)
            assert jnp.allclose(prob_sums, 1.0, atol=1e-5), f"Probabilities don't sum to 1 for batch_size={batch_size}"
