#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

import unittest
import itertools
import numpy as np

from pymdp import utils
from pymdp.jax import utils as jax_utils

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

if __name__ == "__main__":
    unittest.main()