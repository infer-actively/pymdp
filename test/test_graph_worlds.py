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

    def test_single_list_must_match_batch_size(self):
        env, graph = _env()
        key = jr.PRNGKey(0)
        with self.assertRaisesRegex(ValueError, "object_locations must have length batch_size"):
            env.generate_env_params(graph, key=key, object_locations=[1], batch_size=4)
        with self.assertRaisesRegex(ValueError, "object_locations must have length batch_size"):
            env.generate_env_params(graph, key=key, object_locations=[1, 2], batch_size=4)
        with self.assertRaisesRegex(ValueError, "agent_locations must have length batch_size"):
            env.generate_env_params(graph, key=key, agent_locations=[0, 1], batch_size=4)

    def test_out_of_range_locations_raise(self):
        env, graph = _env()
        with self.assertRaisesRegex(ValueError, "agent_locations values"):
            env.generate_env_params(graph, object_locations=[0, 1], agent_locations=[0, 7])
        with self.assertRaisesRegex(ValueError, "agent_locations values"):
            env.generate_env_params(graph, object_locations=[0, 1], agent_locations=[-1, 0])
        with self.assertRaisesRegex(ValueError, "agent_locations values"):
            env.generate_env_params(graph, object_locations=[0, 1], agent_locations=[0, 3])
        with self.assertRaisesRegex(ValueError, "object_locations values"):
            env.generate_env_params(graph, object_locations=[0, 4], agent_locations=[0, 1])

    def test_single_list_with_batch_size_samples_the_other(self):
        env, graph = _env()
        params = env.generate_env_params(
            graph, key=jr.PRNGKey(0), object_locations=[3, 0], batch_size=2
        )
        self.assertEqual(float(params["D"][1][0, 3]), 1.0)
        self.assertEqual(float(params["D"][1][1, 0]), 1.0)
        self.assertTrue(bool(jnp.allclose(params["D"][0].sum(axis=-1), 1.0)))

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


def _moves(env):
    """Moves (from, to) between different locations that B[0] allows."""
    B = env.B[0]  # (to, from, action)
    n = B.shape[0]
    return {(f, t) for f in range(n) for t in range(n) if f != t and float(B[t, f, t]) == 1.0}


def _expected_moves(graph):
    index = {node: i for i, node in enumerate(graph.nodes)}
    return {(index[u], index[v]) for u, v in graph.edges} | {(index[v], index[u]) for u, v in graph.edges}


class TestGraphEnvConstruction(unittest.TestCase):
    def test_non_integer_node_labels_follow_node_order(self):
        """Location i should be the i-th node of graph.nodes if nodes are not 0..N-1."""
        graphs = {
            "tuples": nx.grid_2d_graph(2, 2),
            "strings": nx.relabel_nodes(nx.path_graph(4), lambda i: f"room{i}"),
            "one-indexed": nx.relabel_nodes(nx.path_graph(4), lambda i: i + 1),
        }
        for name, graph in graphs.items():
            with self.subTest(graph=name):
                env = GraphEnv(graph, object_location=0, agent_location=0)
                self.assertEqual(_moves(env), _expected_moves(graph))

    def test_integer_node_labels_keep_their_index(self):
        """Nodes 0..N-1 should keep their label as index, whatever their order."""
        graph = nx.Graph([(2, 3), (0, 1), (1, 2)])
        env = GraphEnv(graph, object_location=0, agent_location=0)
        self.assertEqual(_moves(env), {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)})

    def test_sampled_locations_use_separate_keys(self):
        """Fixing one initial location should not change the other sampled one."""
        graph = nx.path_graph(5)
        for seed in (None, 3):
            key = None if seed is None else jr.PRNGKey(seed)
            with self.subTest(seed=seed):
                sampled = GraphEnv(graph, key=key)
                agent_loc = int(jnp.argmax(sampled.D[0]))
                object_loc = int(jnp.argmax(sampled.D[1]))
                with_object = GraphEnv(graph, object_location=object_loc, key=key)
                with_agent = GraphEnv(graph, agent_location=agent_loc, key=key)
                self.assertEqual(int(jnp.argmax(with_object.D[0])), agent_loc)
                self.assertEqual(int(jnp.argmax(with_agent.D[1])), object_loc)


if __name__ == "__main__":
    unittest.main()
