#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""
import os, sys
import unittest

import numpy as np

# import the library directly from local source (rather than relying on the library being installed)
# insert the dependency so it's prioritized over an installed variant
sys.path.insert(0, os.path.abspath('../pymdp'))

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

if __name__ == "__main__":
    unittest.main()