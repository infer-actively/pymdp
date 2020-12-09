#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=E1136

""" Dirichlet
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import numpy as np
from scipy import special
import warnings
import copy
from pymdp.distributions import Categorical
from pymdp.core import maths


class Dirichlet(object):
    """ A Dirichlet distribution """

    def __init__(self, dims=None, values=None):
        """Initialize a Dirichlet distribution

        `IS_AOA` refers to whether this class uses the `array-of-array` formulation

        Parameters
        ----------
       -  dims: list of int _or_ list of list (where each list is a list of int)
            Specify the number and size of dimensions
            This will initialize the parameters with zero values
        - values: np.ndarray
            The parameters of the distribution
        """

        self.IS_AOA = False

        if values is not None and dims is not None:
            raise ValueError("please provide _either_ :dims: or :values:, not both")

        if values is None and dims is None:
            self.values = np.zeros([1, 1])

        if values is not None:
            self.construct_values(values)

        if dims is not None:
            self.construct_dims(dims)

        if self.IS_AOA:
            self.n_arrays = len(self.values)
        else:
            self.n_arrays = 1

    def construct_values(self, values):
        """Initialize a Dirichlet distribution with `values` argument

        Parameters
        ----------
        - values: np.ndarray
            The parameters of the distribution

        """
        if not isinstance(values, np.ndarray):
            raise ValueError(":values: must be a :numpy.ndarray:")

        if values.dtype == "object":
            self.IS_AOA = True

        if self.IS_AOA:
            self.values = np.empty(len(values), dtype="object")
            for i, array in enumerate(values):
                if array.ndim == 1:
                    values[i] = np.expand_dims(values[i], axis=1)
                self.values[i] = values[i].astype("float64")
        else:
            if values.ndim == 1:
                values = np.expand_dims(values, axis=1)
            self.values = values.astype("float64")

    def construct_dims(self, dims):
        """Initialize a Dirichlet distribution with `values` argument

        Parameters
        ----------
        - dims: list of int
            Specify the number and size of dimensions
            This will initialize the parameters with zero values
        """

        if isinstance(dims, list):
            if any(isinstance(el, list) for el in dims):
                if not all(isinstance(el, list) for el in dims):
                    raise ValueError(":list: of :dims: must only contains :lists:")
                self.IS_AOA = True

            if self.IS_AOA:
                self.values = np.empty(len(dims), dtype="object")
                for i in range(len(dims)):
                    if len(dims[i]) == 1:
                        self.values[i] = np.zeros([dims[i][0], 1])
                    else:
                        self.values[i] = np.zeros(dims[i])
            else:
                if len(dims) == 1:
                    self.values = np.zeros([dims[0], 1])
                else:
                    self.values = np.zeros(dims)
        elif isinstance(dims, int):
            self.values = np.zeros([dims, 1])
        else:
            raise ValueError(":dims: must be either :list: or :int:")

    def mean(self, return_numpy=False):
        """ Normalize distribution

        This function will ensure the distribution(s) integrate to 1.0
        In the case `ndims` >= 2, normalization is performed along the columns of the arrays
        """
        if self.IS_AOA:
            normed = np.empty(len(self.values), dtype=object)
            for i in range(len(self.values)):
                arr = self.values[i]
                column_sums = np.sum(arr, axis=0)
                arr = np.divide(arr, column_sums)
                arr[np.isnan(arr)] = np.divide(1.0, arr.shape[0])
                normed[i] = arr
        else:
            column_sums = np.sum(self.values, axis=0)
            normed = np.divide(self.values, column_sums)
            normed[np.isnan(normed)] = np.divide(1.0, normed.shape[0])

        if return_numpy:
            return normed
        else:
            return Categorical(values=normed)

    def remove_zeros(self):
        """ Remove zeros by adding a small number
        @NOTE exp(-16) is used as the minimum value

        """
        self.values += 1e-16

    def expectation_of_log(self, return_numpy=True):
        """ Expectation of the log 

        @TODO this might need renaming
        """

        if self.IS_AOA:
            wA = np.empty(len(self.values), dtype=object)
            for i in range(len(self.values)):
                A = Dirichlet(values=self[i].values)
                A.remove_zeros()
                wA[i] = maths.spm_wnorm(A.values)
            if return_numpy:
                return wA
            else:
                return Dirichlet(values=wA)
        else:
            A = Dirichlet(values=self.values)
            A.remove_zeros()
            wA = maths.spm_wnorm(A.values)
            if return_numpy:
                return wA
            else:
                return Dirichlet(values=wA)

    def contains_zeros(self):
        """ Checks if any values are zero

        Returns
        ----------
        - bool
            Whether there are any zeros

        """
        if not self.IS_AOA:
            return (self.values == 0.0).any()
        else:
            for i in range(len(self.values)):
                if (self.values[i] == 0.0).any():
                    return True
            return False

    def entropy(self):
        """ Entropy of distribution 

        """
        if not self.IS_AOA:
            values = np.copy(self.values)
            if values.ndim > 1:
                output = np.zeros(values.shape[1])
                for col_i in range(values.shape[1]):
                    first_term = maths.spm_betaln(values[:, col_i])
                    a0 = values[:, col_i].sum(axis=0)
                    second_term = (a0 - values.shape[0]) * special.digamma(a0)
                    third_term = -np.sum(
                        (values[:, col_i] - 1) * special.digamma(values[:, col_i]), axis=0
                    )
                    output[col_i] = first_term + second_term + third_term
            else:
                first_term = maths.spm_betaln(values)
                a0 = values.sum(axis=0)
                second_term = (a0 - values.shape[0]) * special.digamma(a0)
                third_term = -np.sum((values - 1) * special.digamma(values), axis=0)
                output = first_term + second_term + third_term

        else:
            output = np.zeros(len(self.values))
            for i in range(len(self.values)):
                output[i] = self.values[i].entropy()
        return output

    def variance(self, return_numpy=False):
        """ Variance 

        """

        if not self.IS_AOA:
            values = np.copy(self.values)
            a_mean = values / values.sum(axis=0)
            a_0 = values.sum(axis=0)
            numerator = a_mean * (1.0 - a_mean)
            denominator = a_0 + 1.0
            if a_mean.shape[0] > 1:
                output = numerator / np.tile(denominator, (a_mean.shape[0], 1))
            else:
                output = numerator / denominator
            if return_numpy:
                return output
            else:
                return Dirichlet(values=output)
        else:
            result_array = Dirichlet(dims=[list(el.shape) for el in self.values])
            for i in range(len(self.values)):
                result_array[i] = self.values[i].variance(return_numpy=True)
            return result_array

    def log(self, return_numpy=False):
        """ Return the log of the parameters

        Parameters
        ----------
        - return_numpy: bool
            Whether to return a :np.ndarray: or :Dirichlet: object

        Returns
        ----------
        - np.ndarray or Dirichlet
            The log of the parameters
        """

        if self.contains_zeros():
            self.remove_zeros()
            warnings.warn(
                "You have called :log: on a Dirichlet that contains zeros. \
                     We have removed zeros."
            )

        if not self.IS_AOA:
            values = np.copy(self.values)
            log_values = np.log(values)
        else:
            log_values = np.empty(len(self.values), dtype="object")
            for i in range(len(self.values)):
                values = np.copy(self.values[i])
                log_values[i] = np.log(values)

        if return_numpy:
            return log_values
        else:
            return Dirichlet(values=log_values)

    def copy(self):
        """Returns a copy of this object

        Returns
        ----------
        Dirichlet
            Returns a copy of this object
        """
        if not self.IS_AOA:
            values = np.copy(self.values)
        else:
            values = copy.deepcopy(self.values)
        return Dirichlet(values=values)

    def print_shape(self):
        if not self.IS_AOA:
            print("Shape: {}".format(self.values.shape))
        else:
            string = [str(el.shape) for el in self.values]
            print("Shape: {} {}".format(self.values.shape, string))

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def shape(self):
        return self.values.shape

    def __add__(self, other):
        if isinstance(other, Dirichlet):
            values = self.values + other.values
            return Dirichlet(values=values)
        else:
            values = self.values + other
            return Dirichlet(values=values)

    def __radd__(self, other):
        if isinstance(other, Dirichlet):
            values = self.values + other.values
            return Dirichlet(values=values)
        else:
            values = self.values + other
            return Dirichlet(values=values)

    def __sub__(self, other):
        if isinstance(other, Dirichlet):
            values = self.values - other.values
            return Dirichlet(values=values)
        else:
            values = self.values - other
            return Dirichlet(values=values)

    def __rsub__(self, other):
        if isinstance(other, Dirichlet):
            values = self.values - other.values
            return Dirichlet(values=values)
        else:
            values = self.values - other
            return Dirichlet(values=values)

    def __mul__(self, other):
        if isinstance(other, Dirichlet):
            values = self.values * other.values
            return Dirichlet(values=values)
        else:
            values = self.values * other
            return Dirichlet(values=values)

    def __rmul__(self, other):
        if isinstance(other, Dirichlet):
            values = self.values * other.values
            return Dirichlet(values=values)
        else:
            values = self.values * other
            return Dirichlet(values=values)

    def __contains__(self, value):
        pass

    def __getitem__(self, key):
        values = self.values[key]
        if isinstance(values, np.ndarray):
            if values.ndim == 1:
                values = values[:, np.newaxis]
            return Dirichlet(values=values)
        else:
            return values

    def __setitem__(self, idx, value):
        if isinstance(value, Dirichlet):
            value = value.values
        self.values[idx] = value

    def __repr__(self):
        if not self.IS_AOA:
            return "<Dirichlet Distribution> \n {}".format(np.round(self.values, 3))
        else:
            string = [np.round(el, 3) for el in self.values]
            return "<Dirichlet Distribution> \n {}".format(string)
