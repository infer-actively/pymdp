#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests

__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import os
import unittest

import numpy as np
from scipy.io import loadmat

from pymdp.distributions import Categorical

DATA_PATH = "test/data/"


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

    def test_sample_single(self):

        # values are already normalized
        values = np.array([1.0, 0.0])
        c = Categorical(values=values)
        self.assertEqual(0, c.sample())

        # values are not normalized
        values = np.array([0, 10.0])
        c = Categorical(values=values)
        self.assertEqual(1, c.sample())

    def test_sample_AoA(self):

        # values are already normalized
        values_1 = np.array([1.0, 0.0])
        values_2 = np.array([0.0, 1.0, 0.0])
        values = np.array([values_1, values_2])
        c = Categorical(values=values)
        self.assertTrue(np.isclose(np.array([0, 1]), c.sample()).all())

        # values are not normalized
        values_1 = np.array([10.0, 0.0])
        values_2 = np.array([0.0, 10.0, 0.0])
        values = np.array([values_1, values_2])
        c = Categorical(values=values)
        self.assertTrue(np.isclose(np.array([0, 1]), c.sample()).all())

    def test_dot_function_a(self):
        """ Test with vectors and matrices, discrete state / outcomes """

        array_path = os.path.join(os.getcwd(), DATA_PATH + "dot_a.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"]
        obs = mat_contents["o"]
        states = mat_contents["s"]
        states = np.array(states, dtype=object)

        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]
        result_3 = mat_contents["result3"]

        A = Categorical(values=A)
        result_1_py = A.dot(obs, obs_mode=True, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        result_2_py = A.dot(states, return_numpy=True)
        result_2_py = result_2_py.astype("float64")[:, np.newaxis] # type: ignore
        self.assertTrue(np.isclose(result_2, result_2_py).all())

        result_3_py = A.dot(states, dims_to_omit=[0], return_numpy=True)
        self.assertTrue(np.isclose(result_3, result_3_py).all())

    def test_dot_function_a_cat(self):
        """ Test with vectors and matrices, discrete state / outcomes
        Now, when arguments themselves are instances of Categorical
        """

        array_path = os.path.join(os.getcwd(), DATA_PATH + "dot_a.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"]
        obs = Categorical(values=mat_contents["o"])
        states = Categorical(values=mat_contents["s"][0])
        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]
        result_3 = mat_contents["result3"]

        A = Categorical(values=A)
        result_1_py = A.dot(obs, obs_mode=True, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        result_2_py = A.dot(states, return_numpy=True)
        result_2_py = result_2_py.astype("float64")[:, np.newaxis] # type: ignore
        self.assertTrue(np.isclose(result_2, result_2_py).all())

        result_3_py = A.dot(states, dims_to_omit=[0], return_numpy=True)
        self.assertTrue(np.isclose(result_3, result_3_py).all())

    def test_dot_function_b(self):
        """ Continuous states and outcomes """
        array_path = os.path.join(os.getcwd(), DATA_PATH + "dot_b.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"]
        obs = mat_contents["o"]
        states = mat_contents["s"]
        states = np.array(states, dtype=object)

        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]
        result_3 = mat_contents["result3"]

        A = Categorical(values=A)
        result_1_py = A.dot(obs, obs_mode=True, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        result_2_py = A.dot(states, return_numpy=True)
        result_2_py = result_2_py.astype("float64")[:, np.newaxis] # type: ignore
        self.assertTrue(np.isclose(result_2, result_2_py).all())

        result_3_py = A.dot(states, dims_to_omit=[0], return_numpy=True)
        self.assertTrue(np.isclose(result_3, result_3_py).all())

    def test_dot_function_c(self):
        """ Discrete states and outcomes, but also a third hidden state factor """
        array_path = os.path.join(os.getcwd(), DATA_PATH + "dot_c.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"]
        obs = mat_contents["o"]
        states = mat_contents["s"]
        states_array_version = np.empty(states.shape[1], dtype=object)
        for i in range(states.shape[1]):
            states_array_version[i] = states[0][i][0]

        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]
        result_3 = mat_contents["result3"]

        A = Categorical(values=A)
        result_1_py = A.dot(obs, obs_mode=True, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        result_2_py = A.dot(states_array_version, return_numpy=True)
        result_2_py = result_2_py.astype("float64")[:, np.newaxis] # type: ignore
        self.assertTrue(np.isclose(result_2, result_2_py).all())

        result_3_py = A.dot(states_array_version, dims_to_omit=[0], return_numpy=True)
        self.assertTrue(np.isclose(result_3, result_3_py).all())

    def test_dot_function_c_cat(self):
        """ Test with vectors and matrices, discrete state / outcomes but with a
        third hidden state factor. Now, when arguments themselves are
        instances of Categorical
        """

        array_path = os.path.join(os.getcwd(), DATA_PATH + "dot_c.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"]
        obs = Categorical(values=mat_contents["o"])
        states = mat_contents["s"]
        states_array_version = np.empty(states.shape[1], dtype=object)
        for i in range(states.shape[1]):
            states_array_version[i] = states[0][i][0]
        states_array_version = Categorical(values=states_array_version)
        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]
        result_3 = mat_contents["result3"]

        A = Categorical(values=A)
        result_1_py = A.dot(obs, obs_mode=True, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        result_2_py = A.dot(states_array_version, return_numpy=True)
        result_2_py = result_2_py.astype("float64")[:, np.newaxis] # type: ignore
        self.assertTrue(np.isclose(result_2, result_2_py).all())

        result_3_py = A.dot(states_array_version, dims_to_omit=[0], return_numpy=True)
        self.assertTrue(np.isclose(result_3, result_3_py).all())

    def test_dot_function_d(self):
        """ Continuous states and outcomes, but also a third hidden state factor """
        array_path = os.path.join(os.getcwd(), DATA_PATH + "dot_d.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"]
        obs = mat_contents["o"]
        states = mat_contents["s"]
        states_array_version = np.empty(states.shape[1], dtype=object)
        for i in range(states.shape[1]):
            states_array_version[i] = states[0][i][0]

        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]
        result_3 = mat_contents["result3"]

        A = Categorical(values=A)
        result_1_py = A.dot(obs, obs_mode=True, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        result_2_py = A.dot(states_array_version, return_numpy=True)
        result_2_py = result_2_py.astype("float64")[:, np.newaxis] # type: ignore
        self.assertTrue(np.isclose(result_2, result_2_py).all())

        result_3_py = A.dot(states_array_version, dims_to_omit=[0], return_numpy=True)
        self.assertTrue(np.isclose(result_3, result_3_py).all())

    def test_dot_function_e(self):
        """ Continuous states and outcomes, but add a final (fourth) hidden state factor """
        array_path = os.path.join(os.getcwd(), DATA_PATH + "dot_e.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"]
        obs = mat_contents["o"]
        states = mat_contents["s"]
        states_array_version = np.empty(states.shape[1], dtype=object)
        for i in range(states.shape[1]):
            states_array_version[i] = states[0][i][0]

        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]
        result_3 = mat_contents["result3"]

        A = Categorical(values=A)
        result_1_py = A.dot(obs, obs_mode=True, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        result_2_py = A.dot(states_array_version, return_numpy=True)
        result_2_py = result_2_py.astype("float64")[:, np.newaxis] # type: ignore
        self.assertTrue(np.isclose(result_2, result_2_py).all())

        result_3_py = A.dot(states_array_version, dims_to_omit=[0], return_numpy=True)
        self.assertTrue(np.isclose(result_3, result_3_py).all())

    def test_dot_function_e_cat(self):
        """ Continuous states and outcomes, but add a final (fourth) hidden state factor
        Now, when arguments themselves are instances of Categorical """
        array_path = os.path.join(os.getcwd(), DATA_PATH + "dot_e.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"]
        obs = Categorical(values=mat_contents["o"])
        states = mat_contents["s"]
        states_array_version = np.empty(states.shape[1], dtype=object)
        for i in range(states.shape[1]):
            states_array_version[i] = states[0][i][0]
        states_array_version = Categorical(values=states_array_version)

        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]
        result_3 = mat_contents["result3"]

        A = Categorical(values=A)
        result_1_py = A.dot(obs, obs_mode=True, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        result_2_py = A.dot(states_array_version, return_numpy=True)
        result_2_py = result_2_py.astype("float64")[:, np.newaxis] # type: ignore
        self.assertTrue(np.isclose(result_2, result_2_py).all())

        result_3_py = A.dot(states_array_version, dims_to_omit=[0], return_numpy=True)
        self.assertTrue(np.isclose(result_3, result_3_py).all())

    def test_dot_function_f(self):
        """ Test for when the outcome modality is a trivially one-dimensional vector, 
        meaningthe return of spm_dot is a scalar - this tests that the spm_dot function
        successfully wraps such scalar returns into an array """

        states = np.empty(2, dtype=object)
        states[0] = np.array([0.75, 0.25])
        states[1] = np.array([0.5, 0.5])
        No = 1
        A = Categorical(values=np.ones([No] + list(states.shape)))
        A.normalize()

        # return the result as a Categorical
        result_cat = A.dot(states, return_numpy=False)
        self.assertTrue(np.prod(result_cat.shape) == 1)

        # return the result as a numpy array
        result_np = A.dot(states, return_numpy=True)
        self.assertTrue(np.prod(result_np.shape) == 1)

    def test_cross_function_a(self):
        """Test case a: passing in a single-factor hidden state vector -
        under both options of Categorical: where self.AOA is True or False """
        array_path = os.path.join(os.getcwd(), DATA_PATH + "cross_a.mat")
        mat_contents = loadmat(file_name=array_path)
        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]

        states = np.empty(1, dtype=object)
        states[0] = mat_contents["s"][0, 0].squeeze()
        # This would create a 1-dimensional array of arrays (namely, self.IS_AOA == True)
        states = Categorical(values=states)
        result_1_py = states.cross(return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        # this creates a simple single-factor Categorical (namely, self.IS_AOA == False)
        states = Categorical(values=mat_contents["s"][0, 0].squeeze())
        result_2_py = states.cross(return_numpy=True)
        self.assertTrue(np.isclose(result_2, result_2_py).all())

    def test_cross_function_b(self):
        """Test case b: outer-producting two vectors together:
        Options:
        - both vectors are stored in single Categorical (with two entries, where self.AoA == True)
        - first vector is a Categorical (self.AoA = False) and second array is a numpy ndarray
        (non-object array)
        - first vector is a Categorical, second vector is also Categorical
        """
        array_path = os.path.join(os.getcwd(), DATA_PATH + "cross_b.mat")
        mat_contents = loadmat(file_name=array_path)
        result_1 = mat_contents["result1"]
        result_2 = mat_contents["result2"]

        # first way, where both arrays as stored as two entries in a single AoA Categorical
        states = Categorical(values=mat_contents["s"][0])
        result_1_py = states.cross(return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1_py).all())

        # second way (type 1), where first array is a Categorical, second array is a
        # straight numpy array
        states_first_factor = Categorical(values=mat_contents["s"][0][0])
        states_second_factor = mat_contents["s"][0][1]
        result_2a_py = states_first_factor.cross(states_second_factor, return_numpy=True)
        self.assertTrue(np.isclose(result_2, result_2a_py).all())

        # second way (type 2), where first array is a Categorical, second array
        # is another Categorical
        states_first_factor = Categorical(values=mat_contents["s"][0][0])
        states_second_factor = Categorical(values=mat_contents["s"][0][1])
        result_2b_py = states_first_factor.cross(states_second_factor, return_numpy=True)
        self.assertTrue(np.isclose(result_2, result_2b_py).all())

    def test_cross_function_c(self):
        """Test case c: outer-producting a vector and a matrix together:
        Options:
        - First vector is a Categorical, and the matrix argument is a numpy ndarray
        (non-object array)
        - First vector is a Categorical, and the matrix argument is also a Categorical
        """
        array_path = os.path.join(os.getcwd(), DATA_PATH + "cross_c.mat")
        mat_contents = loadmat(file_name=array_path)
        result_1 = mat_contents["result1"]
        random_vec = Categorical(values=mat_contents["random_vec"])

        # first way, where first array is a Categorical, second array is a numpy ndarray
        random_matrix = mat_contents["random_matrix"]
        result_1a_py = random_vec.cross(random_matrix, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1a_py).all())

        # second way, where first array is a Categorical, second array is a Categorical
        random_matrix = Categorical(values=mat_contents["random_matrix"])
        result_1b_py = random_vec.cross(random_matrix, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1b_py).all())

    def test_cross_function_d(self):
        """ Test case d: outer-producting a vector and a sequence of vectors:
        Options:
        - First vector is a Categorical, second sequence of vectors is a numpy ndarray
        (dtype = object)
        - First vector is a Categorical, second sequence of vectors is a Categorical
        (where self.IS_AOA = True))
        """
        array_path = os.path.join(os.getcwd(), DATA_PATH + "cross_d.mat")
        mat_contents = loadmat(file_name=array_path)
        result_1 = mat_contents["result1"]
        random_vec = Categorical(values=mat_contents["random_vec"])
        states = mat_contents["s"]
        for i in range(len(states)):
            states[i] = states[i].squeeze()

        # First way, where first array is a Categorical, second array is a numpy ndarray
        # (dtype = object)
        result_1a_py = random_vec.cross(states, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1a_py).all())

        # Second way, where first array is a Categorical, second array is a Categorical
        # (where self.IS_AOA = True)
        states = Categorical(values=states[0])
        result_1b_py = random_vec.cross(states, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1b_py).all())

    def test_cross_function_e(self):
        """Test case e: outer-producting two sequences of vectors:
        Options:
        - First sequence is an AoA Categorical, second sequence is a numpy ndarray
        (dtype = object)
        - First sequence is NOT an AoA Categorical, second sequence is a numpy ndarray
        (dtype = object)
        - First sequence is an AoA Categorical, second sequence is an AoA Categorical
        - First sequence is NOT an AoA Categorical, second sequence is an AoA Categorical
        """
        array_path = os.path.join(os.getcwd(), DATA_PATH + "cross_e.mat")
        mat_contents = loadmat(file_name=array_path)
        result_1 = mat_contents["result1"]

        s2 = mat_contents["s2"]
        s2_new = np.empty(s2.shape[1], dtype=object)
        for i in range(len(s2_new)):
            s2_new[i] = s2[0][i].squeeze()

        # First way (type 1), first sequence is a Categorical (self.AOA = True) and
        # second sequence is a numpy ndarray (dtype = object)
        s1 = Categorical(values=mat_contents["s1"][0])
        result_1aa_py = s1.cross(s2_new, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1aa_py).all())

        # First way (type 2), first sequence is a Categorical (self.AOA = False) and
        # second sequence is a numpy ndarray (dtype = object)
        s1 = Categorical(values=mat_contents["s1"][0][0])
        result_1ab_py = s1.cross(s2_new, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_1ab_py).all())

        s2_new = Categorical(values=mat_contents["s2"][0])
        # Second way (type 1), first sequence is a Categorical (self.AOA = True)
        # and second sequence is a Categorical
        s1 = Categorical(values=mat_contents["s1"][0])
        result_2aa_py = s1.cross(s2_new, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_2aa_py).all())

        # Second way (type 2), first sequence is a Categorical (self.AOA = False)
        # and second sequence is a Categorical
        s1 = Categorical(values=mat_contents["s1"][0][0])
        result_2ab_py = s1.cross(s2_new, return_numpy=True)
        self.assertTrue(np.isclose(result_1, result_2ab_py).all())


if __name__ == "__main__":
    unittest.main()
