#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

import unittest
import itertools
import numpy as np
from jax import numpy as jnp

from pymdp.legacy import utils
from pymdp import utils as jax_utils

class TestUtils(unittest.TestCase):
    def test_obj_array_from_list(self):
        """
        Tests `obj_array_from_list`
        """
        # make arrays with same leading dimensions. naive method trigger numpy broadcasting error.
        arrs = [np.zeros((3, 6)), np.zeros((3, 4, 5))]
        obs_arrs = utils.obj_array_from_list(arrs)
        
        self.assertTrue(all([np.all(a == b) for a, b in zip(arrs, obs_arrs)]))
    
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
        with self.assertRaises(ValueError):
            jax_utils.validate_normalization(tensor, axis=1, tensor_name="zero_tensor")

    def test_validate_normalization_not_normalised_raises(self):
        """
        Non-zero but not summing to 1 along the checked axis should raise ValueError
        """
        base = jnp.ones((2, 3, 4)) / 3.0   # normalized
        bad = base * 2.0                   # sums to 2.0 along axis=1
        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
            jax_utils.validate_normalization(t, axis=1, tensor_name="axis1_bad")


if __name__ == "__main__":
    unittest.main()