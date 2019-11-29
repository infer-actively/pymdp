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

from inferactively import Categorical


class TestCategorical(unittest.TestCase):
    def test_init_empty(self):
        c = Categorical()
        self.assertEqual(c.ndim, 2)

    def test_init_overload(self):
        with self.assertRaises(ValueError):
            values = np.random.rand(3, 2)
            _ = Categorical(dims=2, values=values)

    def test_float_conversion(self):
        values = np.array([2, 3])
        self.assertEqual(values.dtype, np.int)
        c = Categorical(values=values)
        self.assertEqual(c.values.dtype, np.float64)

    def test_init_dims_expand(self):
        c = Categorical(dims=[5])
        self.assertEqual(c.shape, (5, 1))

    def test_init_dims_int_expand(self):
        c = Categorical(dims=5)
        self.assertEqual(c.shape, (5, 1))

    def test_multi_factor_init_dims(self):
        c = Categorical(dims=[[5, 4], [4, 3]])
        self.assertEqual(c.shape, (2,))
        self.assertEqual(c[0].shape, (5, 4))
        self.assertEqual(c[1].shape, (4, 3))

    def test_multi_factor_init_values(self):
        values_1 = np.random.rand(5, 4)
        values_2 = np.random.rand(4, 3)
        values = np.array([values_1, values_2])
        c = Categorical(values=values)
        self.assertEqual(c.shape, (2,))
        self.assertEqual(c[0].shape, (5, 4))
        self.assertEqual(c[1].shape, (4, 3))

    def test_multi_factor_init_values_expand(self):
        values_1 = np.random.rand(5)
        values_2 = np.random.rand(4)
        values = np.array([values_1, values_2])
        c = Categorical(values=values)
        self.assertEqual(c.shape, (2,))
        self.assertEqual(c[0].shape, (5, 1))
        self.assertEqual(c[1].shape, (4, 1))

    def test_normalize_multi_factor(self):
        values_1 = np.random.rand(5)
        values_2 = np.random.rand(4, 3)
        values = np.array([values_1, values_2])
        c = Categorical(values=values)
        c.normalize()
        self.assertTrue(c.is_normalized())

    def test_normalize_single_dim(self):
        values = np.array([1.0, 1.0])
        c = Categorical(values=values)
        expected_values = np.array([[0.5], [0.5]])
        c.normalize()
        self.assertTrue(np.array_equal(c.values, expected_values))

    def test_normalize_two_dim(self):
        values = np.array([[1.0, 1.0], [1.0, 1.0]])
        c = Categorical(values=values)
        expected_values = np.array([[0.5, 0.5], [0.5, 0.5]])
        c.normalize()
        self.assertTrue(np.array_equal(c.values, expected_values))

    def test_is_normalized(self):
        values = np.array([[0.7, 0.5], [0.3, 0.5]])
        c = Categorical(values=values)
        self.assertTrue(c.is_normalized())
        values = np.array([[0.2, 0.8], [0.3, 0.5]])
        c = Categorical(values=values)
        self.assertFalse(c.is_normalized())

    def test_remove_zeros(self):
        values = np.array([[1.0, 0.0], [1.0, 1.0]])
        c = Categorical(values=values)
        self.assertTrue((c.values == 0.0).any())
        c.remove_zeros()
        self.assertFalse((c.values == 0.0).any())

    def test_contains_zeros(self):
        values = np.array([[1.0, 0.0], [1.0, 1.0]])
        c = Categorical(values=values)
        self.assertTrue(c.contains_zeros())
        values = np.array([[1.0, 1.0], [1.0, 1.0]])
        c = Categorical(values=values)
        self.assertFalse(c.contains_zeros())

    def test_entropy(self):
        values = np.random.rand(3, 2)
        entropy = -np.sum(values * np.log(values), 0)
        c = Categorical(values=values)
        self.assertTrue(np.array_equal(c.entropy(return_numpy=True), entropy))

    def test_log(self):
        values = np.random.rand(3, 2)
        log_values = np.log(values)
        c = Categorical(values=values)
        self.assertTrue(np.array_equal(c.log(return_numpy=True), log_values))

    def test_copy(self):
        values = np.random.rand(3, 2)
        c = Categorical(values=values)
        c_copy = c.copy()
        self.assertTrue(np.array_equal(c_copy.values, c.values))
        c_copy.values = c_copy.values * 2
        self.assertFalse(np.array_equal(c_copy.values, c.values))

    def test_ndim(self):
        values = np.random.rand(3, 2)
        c = Categorical(values=values)
        self.assertEqual(c.ndim, c.values.ndim)

    def test_shape(self):
        values = np.random.rand(3, 2)
        c = Categorical(values=values)
        self.assertEqual(c.shape, (3, 2))

    def test_dot_function_a(self):
        """ test with vectors and matrices, discrete state / outcomes """
        array_path = os.path.join(os.getcwd(), "tests/data/dot_a.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"]
        obs = mat_contents["o"]
        states = mat_contents["s"]
        states = np.array(states, dtype=object)

        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]
        result_3 = mat_contents["result3"]

        A = Categorical(values=A)
        result_1_py = A.dot(obs, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        result_2_py = A.dot(states, return_numpy=True)
        result_2_py = result_2_py.astype("float64")[:, np.newaxis]
        self.assertTrue(np.isclose(result_2, result_2_py).all())

        result_3_py = A.dot(states, dims_to_omit=[0], return_numpy=True)
        self.assertTrue(np.isclose(result_3, result_3_py).all())

        # now try by putting obs and states into Categoricals themselves
        obs = Categorical(values = mat_contents["o"])
        states = Categorical(values = mat_contents["s"][0])

        result_1_py_cat = A.dot(obs, return_numpy=True)
        self.assertTrue(np.isclose(result_1,result_1_py_cat).all())

        result_2_py_cat = A.dot(states, return_numpy=True)
        result_2_py_cat = result_2_py_cat.astype("float64")[:, np.newaxis]
        self.assertTrue(np.isclose(result_2, result_2_py_cat).all())

        result_3_py_cat = A.dot(states, dims_to_omit=[0], return_numpy=True)
        self.assertTrue(np.isclose(result_3_py_cat, result_3_py).all())

    def test_dot_function_b(self):
        """ continuous states and outcomes """
        array_path = os.path.join(os.getcwd(), "tests/data/dot_b.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"]
        obs = mat_contents["o"]
        states = mat_contents["s"]
        states = np.array(states, dtype=object)

        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]
        result_3 = mat_contents["result3"]

        A = Categorical(values=A)
        result_1_py = A.dot(obs, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        result_2_py = A.dot(states, return_numpy=True)
        result_2_py = result_2_py.astype("float64")[:, np.newaxis]
        self.assertTrue(np.isclose(result_2, result_2_py).all())

        result_3_py = A.dot(states, dims_to_omit=[0], return_numpy=True)
        self.assertTrue(np.isclose(result_3, result_3_py).all())

    def test_dot_function_c(self):
        """ DISCRETE states and outcomes, but also a third hidden state factor """
        array_path = os.path.join(os.getcwd(), "tests/data/dot_c.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"]
        obs = mat_contents["o"]
        states = mat_contents["s"]
        states_array_version = np.empty(states.shape[1],dtype=object)
        for i in range(states.shape[1]):
            states_array_version[i] = states[0][i][0]

        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]
        result_3 = mat_contents["result3"]

        A = Categorical(values=A)
        result_1_py = A.dot(obs, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        result_2_py = A.dot(states_array_version, return_numpy=True)
        result_2_py = result_2_py.astype("float64")[:, np.newaxis]
        self.assertTrue(np.isclose(result_2, result_2_py).all())

        result_3_py = A.dot(states_array_version, dims_to_omit=[0], return_numpy=True)
        self.assertTrue(np.isclose(result_3, result_3_py).all())

    def test_dot_function_d(self):
        """ CONTINUOUS states and outcomes, but also a third hidden state factor """
        array_path = os.path.join(os.getcwd(), "tests/data/dot_d.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"]
        obs = mat_contents["o"]
        states = mat_contents["s"]
        states_array_version = np.empty(states.shape[1],dtype=object)
        for i in range(states.shape[1]):
            states_array_version[i] = states[0][i][0]

        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]
        result_3 = mat_contents["result3"]

        A = Categorical(values=A)
        result_1_py = A.dot(obs, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        result_2_py = A.dot(states_array_version, return_numpy=True)
        result_2_py = result_2_py.astype("float64")[:, np.newaxis]
        self.assertTrue(np.isclose(result_2, result_2_py).all())

        result_3_py = A.dot(states_array_version, dims_to_omit=[0], return_numpy=True)
        self.assertTrue(np.isclose(result_3, result_3_py).all())

    def test_dot_function_e(self):
        """ CONTINUOUS states and outcomes, but add a final (fourth) hidden state factor """
        array_path = os.path.join(os.getcwd(), "tests/data/dot_e.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"]
        obs = mat_contents["o"]
        states = mat_contents["s"]
        states_array_version = np.empty(states.shape[1],dtype=object)
        for i in range(states.shape[1]):
            states_array_version[i] = states[0][i][0]

        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]
        result_3 = mat_contents["result3"]

        A = Categorical(values=A)
        result_1_py = A.dot(obs, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        result_2_py = A.dot(states_array_version, return_numpy=True)
        result_2_py = result_2_py.astype("float64")[:, np.newaxis]
        self.assertTrue(np.isclose(result_2, result_2_py).all())

        result_3_py = A.dot(states_array_version, dims_to_omit=[0], return_numpy=True)
        self.assertTrue(np.isclose(result_3, result_3_py).all())


if __name__ == "__main__":
    unittest.main()
