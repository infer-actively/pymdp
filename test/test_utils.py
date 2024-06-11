#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

import unittest
import itertools
import numpy as np

from pymdp import utils

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

        # find flat index without itertools
        act_flat = utils.get_combination_index(act, num_controls)

        self.assertEqual(act_flat, true_act_flat)

    def test_index_to_combination(self):
        """
        Test `index_to_combination`
        """
        num_controls = [10, 20]
        act = [5, 1]

        # make all combinations from itertools and find correct index
        action_map = list(itertools.product(*[list(range(i)) for i in num_controls]))
        act_flat = action_map.index(tuple(act))

        # reconstruct categorical actions from flat index
        act_reconstruct = utils.index_to_combination(act_flat, num_controls)

        self.assertEqual(act_reconstruct, act)

if __name__ == "__main__":
    unittest.main()