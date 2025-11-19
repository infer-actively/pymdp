#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

import unittest
import itertools
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from equinox import EquinoxRuntimeError

from pymdp import utils as jax_utils
from pymdp.legacy import utils as legacy_utils

class TestUtils(unittest.TestCase):

    def test_random_factorized_categorical(self):
        """
        Test `random_factorized_categorical`
        """
        key = jr.PRNGKey(0)
        dims_per_var = [3, 5, 2]

        dists = jax_utils.random_factorized_categorical(key, dims_per_var)
        repeat = jax_utils.random_factorized_categorical(key, dims_per_var)

        self.assertEqual(len(dists), len(dims_per_var))
        for idx, (dist, dim) in enumerate(zip(dists, dims_per_var)):
            with self.subTest(factor=idx):
                self.assertEqual(dist.shape, (dim,))
                self.assertTrue(bool(jnp.all(dist >= 0)))
                self.assertTrue(bool(jnp.isclose(dist.sum(), 1.0)))
                self.assertTrue(bool(jnp.allclose(dist, repeat[idx])))

    def test_random_A_array_shapes_and_normalization(self):
        """
        `random_A_array` should honor factor structure and yield normalized conditionals
        """
        key = jr.PRNGKey(42)
        num_obs = [3, 2]
        num_states = [4, 5, 6]
        A_dependencies = [[0, 2], [1]]  # modality 0 depends on states 0 & 2, modality 1 on state 1 only

        A = jax_utils.random_A_array(key, num_obs, num_states, A_dependencies=A_dependencies)

        self.assertEqual(len(A), len(num_obs))
        expected_shapes = [
            (num_obs[0], num_states[0], num_states[2]),
            (num_obs[1], num_states[1]),
        ]

        for idx, (A_m, expected_shape) in enumerate(zip(A, expected_shapes)):
            with self.subTest(modality=idx):
                self.assertEqual(A_m.shape, expected_shape)
                # each column over the observation axis should be a normalized categorical distribution
                marginal_sums = A_m.sum(axis=0)
                self.assertTrue(bool(jnp.allclose(marginal_sums, jnp.ones_like(marginal_sums))))

    def test_random_A_array_defaults_to_all_factors(self):
        """
        `random_A_array` should fallback to each modality depending on every hidden factor
        """
        key = jr.PRNGKey(7)
        num_obs = [2, 3]
        num_states = [4, 5]

        A = jax_utils.random_A_array(key, num_obs, num_states)

        self.assertEqual(len(A), len(num_obs))
        for idx, (A_m, n_o) in enumerate(zip(A, num_obs)):
            with self.subTest(modality=idx):
                self.assertEqual(A_m.shape, (n_o, *tuple(num_states)))
                marginal_sums = A_m.sum(axis=0)
                self.assertTrue(bool(jnp.allclose(marginal_sums, jnp.ones_like(marginal_sums))))

    def test_random_B_array_shapes_and_normalization(self):
        """`random_B_array` should honor provided dependencies and normalise per control slice"""

        key = jr.PRNGKey(11)
        num_states = [3, 2]
        num_controls = [2, 3]
        B_dependencies = [[0, 1], [1]]
        B_action_dependencies = [[0], [0, 1]]

        B = jax_utils.random_B_array(
            key,
            num_states,
            num_controls,
            B_dependencies=B_dependencies,
            B_action_dependencies=B_action_dependencies,
        )

        expected_shapes = [
            (num_states[0], num_states[0], num_states[1], num_controls[0]),
            (num_states[1], num_states[1], num_controls[0], num_controls[1]),
        ]

        self.assertEqual(len(B), len(num_states))
        for idx, (B_f, expected_shape) in enumerate(zip(B, expected_shapes)):
            with self.subTest(factor=idx):
                self.assertEqual(B_f.shape, expected_shape)
                marginal_sums = B_f.sum(axis=0)
                self.assertTrue(bool(jnp.allclose(marginal_sums, jnp.ones_like(marginal_sums))))

    def test_random_B_array_defaults_to_self_dependencies(self):
        """`random_B_array` should default to self-transition dependencies when none provided"""

        key = jr.PRNGKey(13)
        num_states = [4, 5]
        num_controls = [2, 3]

        B = jax_utils.random_B_array(key, num_states, num_controls)

        expected_shapes = [
            (num_states[0], num_states[0], num_controls[0]),
            (num_states[1], num_states[1], num_controls[1]),
        ]

        self.assertEqual(len(B), len(num_states))
        for idx, (B_f, expected_shape) in enumerate(zip(B, expected_shapes)):
            with self.subTest(factor=idx):
                self.assertEqual(B_f.shape, expected_shape)
                marginal_sums = B_f.sum(axis=0)
                self.assertTrue(bool(jnp.allclose(marginal_sums, jnp.ones_like(marginal_sums))))
    
    def test_norm_dist_list_version(self):
        """"
        Test `list_array_norm_dist`
        """
        dist_list = [jnp.array([0.2, 0.3, 0.5]), jnp.array([1.0, 2.0, 3.0, 4.0])]
        normed_list = jax_utils.list_array_norm_dist(dist_list)

        for idx, (orig, normed) in enumerate(zip(dist_list, normed_list)):
            with self.subTest(factor=idx):
                self.assertTrue(bool(jnp.all(normed >= 0)))
                self.assertTrue(bool(jnp.isclose(normed.sum(), 1.0)))
                expected = orig / orig.sum()
                self.assertTrue(bool(jnp.allclose(normed, expected)))

    def test_get_combination_index(self):
        """
        Test `get_combination_index`
        """
        num_controls = [10, 20]
        act = [5, 1]

        # make all combinations from itertools and find correct index
        action_map = list(itertools.product(*[list(range(i)) for i in num_controls]))
        true_act_flat = action_map.index(tuple(act))

        batch_size = 10
        act_vec = np.array(act)
        act_vec = np.broadcast_to(act_vec, (batch_size,) + act_vec.shape)

        # find flat index without itertools
        act_flat = jax_utils.get_combination_index(act_vec, num_controls)

        self.assertTrue(np.allclose(act_flat, true_act_flat))

    def test_index_to_combination(self):
        """
        Test `index_to_combination`
        """
        num_controls = [10, 20]
        act = [5, 1]

        # make all combinations from itertools and find correct index
        action_map = list(itertools.product(*[list(range(i)) for i in num_controls]))
        act_flat = action_map.index(tuple(act))

        batch_size = 10
        act_flat_vec = np.array([act_flat])
        act_flat_vec = np.broadcast_to(act_flat_vec, (batch_size,))

        # reconstruct categorical actions from flat index
        act_reconstruct = jax_utils.index_to_combination(act_flat_vec, num_controls)

        self.assertTrue(np.allclose(act_reconstruct - np.array([act]), 0))
    
    def test_validate_normalization_ok(self):
        """
        Distributions are properly normalized along axis=1 (default) -> no error
        """
        # shape: (batch=2, num_obs[m]=3, num_states=[4])
        tensor = jnp.ones((2, 3, 4)) / 3.0  # sums to 1 along axis=1
        # should not raise
        jax_utils.validate_normalization(tensor, axis=1, tensor_name="test_tensor")

    def test_validate_normalization_zero_filled_raises(self):
        """
        Zero-filled distributions along the checked axis should raise ValueError
        """
        tensor = jnp.zeros((2, 3, 4))
        with self.assertRaises(EquinoxRuntimeError):
            jax_utils.validate_normalization(tensor, axis=1, tensor_name="zero_tensor")

    def test_validate_normalization_not_normalised_raises(self):
        """
        Non-zero but not summing to 1 along the checked axis should raise ValueError
        """
        base = jnp.ones((2, 3, 4)) / 3.0   # normalized
        bad = base * 2.0                   # sums to 2.0 along axis=1
        with self.assertRaises(EquinoxRuntimeError):
            jax_utils.validate_normalization(bad, axis=1, tensor_name="bad_tensor")

    def test_validate_normalization_axis_argument(self):
        """
        Verify the 'axis' argument: normalized on axis=0 passes; same tensor on axis=1 fails
        """
        # shape: (3, 2); normalize along axis=0
        t = jnp.ones((3, 2)) / 3.0  # each column sums to 1
        # should pass on axis=0
        jax_utils.validate_normalization(t, axis=0, tensor_name="axis0_ok")
        # should fail on axis=1 (each row sums to 2/3)
        with self.assertRaises(EquinoxRuntimeError):
            jax_utils.validate_normalization(t, axis=1, tensor_name="axis1_bad")

    def test_create_controllable_B_matches_legacy(self):
        """`create_controllable_B` should reproduce the legacy construction semantics"""

        num_states = [2, 3]
        num_controls = [2, 3]

        legacy_B = legacy_utils.construct_controllable_B(num_states, num_controls)
        jax_B = jax_utils.create_controllable_B(num_states, num_controls)

        self.assertEqual(len(jax_B), len(num_states))
        for idx, (jax_factor, legacy_factor, ns, nc) in enumerate(zip(jax_B, legacy_B, num_states, num_controls)):
            with self.subTest(factor=idx):
                self.assertEqual(jax_factor.shape, (ns, ns, nc))
                factor_np = np.array(jax_factor)
                self.assertTrue(np.allclose(factor_np, legacy_factor))
                # summing over future states should yield ones for each (current state, action) pair
                marginal = factor_np.sum(axis=0)
                self.assertTrue(np.allclose(marginal, np.ones_like(marginal)))


if __name__ == "__main__":
    unittest.main()
