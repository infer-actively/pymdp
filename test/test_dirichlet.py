#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import os
import sys
import unittest

import numpy as np
from scipy.io import loadmat

from pymdp.distributions import Categorical, Dirichlet  

DATA_PATH = "test/matlab_crossval/output/"


class TestDirichlet(unittest.TestCase):
    def test_init_empty(self):
        d = Dirichlet()
        self.assertEqual(d.ndim, 2)

    def test_init_overload(self):
        with self.assertRaises(ValueError):
            values = np.random.rand(3, 2)
            _ = Dirichlet(dims=2, values=values)

    def test_float_conversion(self):
        values = np.array([2, 3])
        self.assertEqual(values.dtype, np.int)
        d = Dirichlet(values=values)
        self.assertEqual(d.values.dtype, np.float64)

    def test_init_dims_expand(self):
        d = Dirichlet(dims=[5])
        self.assertEqual(d.shape, (5, 1))

    def test_init_dims_int_expand(self):
        d = Dirichlet(dims=5)
        self.assertEqual(d.shape, (5, 1))

    def test_multi_factor_init_dims(self):
        d = Dirichlet(dims=[[5, 4], [4, 3]])
        self.assertEqual(d.shape, (2,))
        self.assertEqual(d[0].shape, (5, 4))
        self.assertEqual(d[1].shape, (4, 3))

    def test_multi_factor_init_values(self):
        values_1 = np.random.rand(5, 4)
        values_2 = np.random.rand(4, 3)
        values = np.array([values_1, values_2], dtype=object)
        d = Dirichlet(values=values)
        self.assertEqual(d.shape, (2,))
        self.assertEqual(d[0].shape, (5, 4))
        self.assertEqual(d[1].shape, (4, 3))

    def test_multi_factor_init_values_expand(self):
        values_1 = np.random.rand(5)
        values_2 = np.random.rand(4)
        values = np.array([values_1, values_2], dtype=object)
        d = Dirichlet(values=values)
        self.assertEqual(d.shape, (2,))
        self.assertEqual(d[0].shape, (5, 1))
        self.assertEqual(d[1].shape, (4, 1))

    def test_normalize_multi_factor(self):
        values_1 = np.random.rand(5)
        values_2 = np.random.rand(4, 3)
        values = np.array([values_1, values_2], dtype=object)
        d = Dirichlet(values=values)
        normed = Categorical(values=d.mean(return_numpy=True))
        self.assertTrue(normed.is_normalized())

    def test_normalize_single_dim(self):
        values = np.array([1.0, 1.0])
        d = Dirichlet(values=values)
        expected_values = np.array([[0.5], [0.5]])
        self.assertTrue(np.array_equal(d.mean(return_numpy=True), expected_values))

    def test_normalize_two_dim(self):
        values = np.array([[1.0, 1.0], [1.0, 1.0]])
        d = Dirichlet(values=values)
        expected_values = np.array([[0.5, 0.5], [0.5, 0.5]])
        self.assertTrue(np.array_equal(d.mean(return_numpy=True), expected_values))

    def test_remove_zeros(self):
        values = np.array([[1.0, 0.0], [1.0, 1.0]])
        d = Dirichlet(values=values)
        self.assertTrue((d.values == 0.0).any())
        d.remove_zeros()
        self.assertFalse((d.values == 0.0).any())

    def test_contains_zeros(self):
        values = np.array([[1.0, 0.0], [1.0, 1.0]])
        d = Dirichlet(values=values)
        self.assertTrue(d.contains_zeros())
        values = np.array([[1.0, 1.0], [1.0, 1.0]])
        d = Dirichlet(values=values)
        self.assertFalse(d.contains_zeros())

    def test_log(self):
        values = np.random.rand(3, 2)
        log_values = np.log(values)
        d = Dirichlet(values=values)
        self.assertTrue(np.array_equal(d.log(return_numpy=True), log_values))

    def test_copy(self):
        values = np.random.rand(3, 2)
        d = Dirichlet(values=values)
        d_copy = d.copy()
        self.assertTrue(np.array_equal(d_copy.values, d.values))
        d_copy.values = d_copy.values * 2
        self.assertFalse(np.array_equal(d_copy.values, d.values))

    def test_ndim(self):
        values = np.random.rand(3, 2)
        d = Dirichlet(values=values)
        self.assertEqual(d.ndim, d.values.ndim)

    def test_shape(self):
        values = np.random.rand(3, 2)
        d = Dirichlet(values=values)
        self.assertEqual(d.shape, (3, 2))

    """ TODO: these tests fail
    def test_expectation_single_factor(self):
        Tests implementation of expect_log method against matlab version (single factor)
       

        array_path = os.path.join(os.getcwd(), DATA_PATH + "wnorm_a.mat")
        mat_contents = loadmat(file_name=array_path)
        result = mat_contents["result"]

        d = Dirichlet(values=mat_contents["A"])
        result_py = d.expectation_of_log(return_numpy=True)
        self.assertTrue(np.isclose(result, result_py).all())

    def test_expectation_multi_factor(self):
        Tests implementation of expect_log method against matlab version (multi factor)
        

        array_path = os.path.join(os.getcwd(), DATA_PATH + "wnorm_b.mat")
        mat_contents = loadmat(file_name=array_path)
        result_1 = mat_contents["result_1"]
        result_2 = mat_contents["result_2"]

        d = Dirichlet(values=mat_contents["A"][0])
        result_py = d.expectation_of_log(return_numpy=True)

        self.assertTrue(
            np.isclose(result_1, result_py[0]).all() and np.isclose(result_2, result_py[1]).all()
        )
    """


if __name__ == "__main__":
    unittest.main()
