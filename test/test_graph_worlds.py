#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for GraphEnv batched initial locations."""

import unittest
import warnings

import jax.numpy as jnp
import jax.random as jr
import networkx as nx

from pymdp.envs.graph_worlds import GraphEnv


def _env():
    graph = nx.path_graph(3)
    return GraphEnv(graph, object_location=0, agent_location=1), graph


class TestGraphEnvBatchLocations(unittest.TestCase):
    def test_partial_location_list_raises(self):
        env, graph = _env()
        with self.assertRaises(ValueError):
            env.generate_env_params(graph, object_locations=[0, 1])
        with self.assertRaises(ValueError):
            env.generate_env_params(graph, agent_locations=[0, 1])

    def test_mismatched_lengths_raise(self):
        env, graph = _env()
        with self.assertRaises(ValueError):
            env.generate_env_params(
                graph, object_locations=[0, 1], agent_locations=[0]
            )
        with self.assertRaises(ValueError):
            env.generate_env_params(
                graph,
                object_locations=[0, 1],
                agent_locations=[0, 1, 2],
                batch_size=2,
            )

    def test_length_must_match_batch_size(self):
        env, graph = _env()
        with self.assertRaises(ValueError):
            env.generate_env_params(
                graph,
                object_locations=[0, 1],
                agent_locations=[1, 2],
                batch_size=3,
            )

    def test_matching_lists_set_batched_priors(self):
        env, graph = _env()
        params = env.generate_env_params(
            graph, object_locations=[1, 3], agent_locations=[0, 2]
        )
        agent_prior, object_prior = params["D"]
        self.assertEqual(tuple(agent_prior.shape), (2, 3))
        self.assertEqual(tuple(object_prior.shape), (2, 4))
        self.assertEqual(float(agent_prior[0, 0]), 1.0)
        self.assertEqual(float(agent_prior[1, 2]), 1.0)
        self.assertEqual(float(object_prior[0, 1]), 1.0)
        self.assertEqual(float(object_prior[1, 3]), 1.0)
        self.assertTrue(bool(jnp.allclose(agent_prior.sum(axis=-1), 1.0)))
        self.assertTrue(bool(jnp.allclose(object_prior.sum(axis=-1), 1.0)))

    def test_omitted_locations_stay_unbatched(self):
        env, graph = _env()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            params = env.generate_env_params(graph)
        self.assertTrue(caught)
        self.assertEqual(params["D"][0].ndim, 1)
        self.assertEqual(params["D"][1].ndim, 1)

    def test_batch_size_samples_missing_locations(self):
        env, graph = _env()
        params = env.generate_env_params(graph, key=jr.PRNGKey(0), batch_size=4)
        self.assertEqual(params["D"][0].shape[0], 4)
        self.assertEqual(params["D"][1].shape[0], 4)
        self.assertTrue(bool(jnp.allclose(params["D"][0].sum(axis=-1), 1.0)))
        self.assertTrue(bool(jnp.allclose(params["D"][1].sum(axis=-1), 1.0)))


if __name__ == "__main__":
    unittest.main()
