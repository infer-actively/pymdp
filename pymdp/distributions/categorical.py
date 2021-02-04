#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=E1136

""" Categorical
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import warnings
import copy
import numpy as np
import matplotlib.pyplot as plt
from pymdp.core import maths, utils


class Categorical(object):
    """ A Categorical distribution
    
        A discrete probability distribution over K possible events    
        Parameters are in the range 0 to 1 and sum to 1
            
        This class assumes that columns encode probability distributions, such that
        multiple columns represents a set of distributions
        This can be useful for representing conditional distributions

        @NOTE It is common to represent arrays of vectors and matrices
        In this case, we use `np.dtype == object`, and set `IS_AOA` flag true

    """

    def __init__(self, dims=None, values=None):
        """Initialize a Categorical distribution

        Parameters
        ----------
        - `dims` [list of ints] || [list of lists of ints] 
            Specifies the number and size of dimensions
        - `values` [np.ndarray]
            The parameters of the distribution
        """

        # whether we use `array of array` formulation
        self.IS_AOA = False

        if values is not None and dims is not None:
            raise ValueError("please provide _either_ `dims` or `values`, not both")

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
        """Initialize a Categorical distribution with `values` argument
        
        Parameters
        ----------
        - `values` [np.ndarray]
            The parameters of the distribution
        """

        if not isinstance(values, np.ndarray):
            raise ValueError("`values` must be a `np.ndarray`")

        if utils.is_arr_of_arr(values):
            self.IS_AOA = True

        if self.IS_AOA:
            self.values = np.empty(len(values), dtype="object")
            for i, array in enumerate(values):
                if array.ndim == 1:
                    # repo generally uses column vectors
                    values[i] = np.expand_dims(values[i], axis=1)
                self.values[i] = values[i].astype("float64")
        else:
            if values.ndim == 1:
                values = np.expand_dims(values, axis=1)
            self.values = values.astype("float64")

    def construct_dims(self, dims):
        """Initialize a Categorical distribution with `dims` argument
        
        @NOTE distributions are initialized with zero values
        
        Parameters
        ----------
        `dims` [list of ints || list of lists of ints]
            Specifies the number and sizes of dimensions
        """

        # check whether array of array
        if isinstance(dims, list):
            if any(isinstance(el, list) for el in dims):
                if not all(isinstance(el, list) for el in dims):
                    raise ValueError("`list` of `dims` must only contains `lists`")
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
            raise ValueError("`dims` must be either `list` or `int`")

    def dot_old(self, x, dims_to_omit=None, return_numpy=False, obs_mode=False):
        """ Dot product of a this distribution with `x`
        
            @NOTE see `spm_dot` in core.maths
            @TODO create better workaround for `obs_mode`

            The dimensions in `dims_to_omit` will not be summed across during the dot product
        
        Parameters
        ----------
        - `x` [1D np.ndarray || Categorical]
            The array to perform the dot product with
        - `dims_to_omit` [list of ints] (optional)
            Which dimensions to omit
        - `return_numpy` [bool] (optional)
            Whether to return `np.ndarray` or `Categorical` - defaults to `Categorical`
        - 'obs_mode' [bool] (optional)
            Whether to perform the inner product of `x` with the leading dimension of self
            
            @NOTE We call this `obs_mode` because it's often used to get the likelihood 
                  of an observation (leading dimension) under different settings of 
                  hidden states (lagging dimensions)
        """
        x = utils.to_numpy(x)

        # perform dot product on each sub-array
        if self.IS_AOA:
            y = np.empty(self.n_arrays, dtype=object)
            for i in range(self.n_arrays):
                y[i] = maths.spm_dot(self[i].values, x, dims_to_omit, obs_mode)
        else:
            y = maths.spm_dot(self.values, x, dims_to_omit, obs_mode)

        if return_numpy:
            return y
        else:
            return Categorical(values=y)
    
    def dot(self, x, dims_to_omit=None, return_numpy=False):
        """ Dot product of distribution encoded in self.values  with `x`
        
            @NOTE see `spm_dot` in core.maths

            The dimensions in `dims_to_omit` will not be summed across during the dot product
        
        Parameters
        ----------
        - `x` [1D np.ndarray || Categorical]
            The array to perform the dot product with
        - `dims_to_omit` [list of ints] (optional)
            Which dimensions to omit
        - `return_numpy` [bool] (optional)
            Whether to return `np.ndarray` or `Categorical` - defaults to `Categorical`
        """
        x = utils.to_numpy(x)

        # perform dot product on each sub-array
        if self.IS_AOA:
            y = np.empty(self.n_arrays, dtype=object)
            for i in range(self.n_arrays):
                y[i] = maths.spm_dot(self[i].values, x, dims_to_omit)
        else:
            y = maths.spm_dot(self.values, x, dims_to_omit)

        if return_numpy:
            return y
        else:
            return Categorical(values=y)

    def dot_likelihood(self, obs, dims_to_omit=None, return_numpy=False):
        """ Product of delta distribution encoded in obs with self.values  with `x`
        
            @NOTE see `dot_likelihood` in core.maths
        
        Parameters
        ----------
        - `obs` [1D np.ndarray || Categorical]
            The observations (delta vector, one-hot vector) to evaluate the likelihood of
        - `return_numpy` [bool] (optional)
            Whether to return `np.ndarray` or `Categorical` - defaults to `Categorical`
        """
        obs = utils.to_numpy(obs)

        # perform dot product on each sub-array
        if self.IS_AOA:
            y = np.empty(self.n_arrays, dtype=object)
            for i in range(self.n_arrays):
                y[i] = maths.dot_likelihood(self[i].values, obs[i])
        else:
            y = maths.dot_likelihood(self.values, obs)

        if return_numpy:
            return y
        else:
            return Categorical(values=y)

    def cross(self, x=None, return_numpy=False, *args):
        """ Multi-dimensional outer product
            
            If no `x` argument is passed, the function returns the "auto-outer product" 
            of self. Otherwise, the function will recursively take the outer product 
            of the initial entry of `x` with `self` until it has depleted the possible 
            entries of `x` that it can outer-product

        Parameters
        ----------
        - `x` [np.ndarray || [Categorical] (optional)
            The values to perform the outer-product with
        - `args` [np.ndarray] || Categorical] (optional)
            Perform the outer product of the `args` with self
       
        Returns
        -------
        - `y` [np.ndarray || Categorical]
            The result of the outer-product
        """
        x = utils.to_numpy(x)

        if x is not None:
            if len(args) > 0 and utils.is_distribution(args[0]):
                arg_array = []
                for arg in args:
                    arg_array.append(arg.values)
                y = maths.spm_cross(self.values, x, *arg_array)
            else:
                y = maths.spm_cross(self.values, x, *args)
        else:
            y = maths.spm_cross(self.values)

        if return_numpy:
            return y
        else:
            return Categorical(values=y)

    def normalize(self):
        """ Normalize distribution (i.e. columns)

            This function will ensure the distribution(s) integrate to 1.0
            In the case `ndims` >= 2, normalization is performed along 
            the columns of the arrays
        """
        if self.is_normalized():
            return
        if self.IS_AOA:
            for i in range(self.n_arrays):
                arr = self.values[i]
                column_sums = np.sum(arr, axis=0)
                arr = np.divide(arr, column_sums)
                arr[np.isnan(arr)] = np.divide(1.0, arr.shape[0])
                self.values[i] = arr
        else:
            column_sums = np.sum(self.values, axis=0)
            self.values = np.divide(self.values, column_sums)
            self.values[np.isnan(self.values)] = np.divide(1.0, self.values.shape[0])

    def is_normalized(self):
        """ Checks whether columns sum to 1
            @NOTE this operates within some margin of error (10^-4)
        """
        if self.IS_AOA:
            array_is_normed = np.zeros(self.n_arrays, dtype=bool)
            for i in range(self.n_arrays):
                error = np.abs(1 - np.sum(self.values[i], axis=0))
                array_is_normed[i] = (error < 0.0001).all()
            return array_is_normed.all()
        else:
            error = np.abs(1 - np.sum(self.values, axis=0))
            return (error < 0.0001).all()

    def remove_zeros(self):
        """ Remove zeros by adding a small number
            @NOTE exp(-16) is used as the minimum value
        """
        self.values += 1e-16

    def contains_zeros(self):
        """ Checks if any values are zero
       
        Returns
        ----------
        - `bool`
            Whether there are any zeros
        """
        if not self.IS_AOA:
            return (self.values == 0.0).any()
        else:
            for i in range(self.n_arrays):
                if (self.values[i] == 0.0).any():
                    return True
            return False

    def entropy(self, return_numpy=False):
        """ Return the entropy of each column
       
        Parameters
        ----------
        -  `return_numpy` [bool] (optional)
            Whether to return as `np.ndarray` 
            Defaults to `False 
        
        Returns
        ----------
        - `np.ndarray` or `Categorical`
            The entropy of the columns
        """

        if self.contains_zeros():
            self.remove_zeros()
            self.normalize()
            warnings.warn(
                "You have called :entropy: on a Categorical that contains zeros. \
                     We have removed zeros and normalized."
            )

        if not self.IS_AOA:
            values = np.copy(self.values)
            entropy = -np.sum(values * np.log(values), 0)
        else:
            entropy = np.empty(self.n_arrays, dtype="object")
            for i in range(self.n_arrays):
                values = np.copy(self.values[i])
                entropy[i] = -np.sum(values * np.log(values), 0)

        if return_numpy:
            return entropy
        else:
            return Categorical(values=entropy)

    def log(self, return_numpy=False):
        """ Return the log of the parameters
        
        Parameters
        ----------
        -  `return_numpy` [bool] (optional)
            Whether to return as `np.ndarray` 
            Defaults to `False 

       Returns
        ----------
        - `np.ndarray` or `Categorical`
            The log of the parameters 
        """

        if self.contains_zeros():
            self.remove_zeros()
            warnings.warn(
                "You have called :log: on a Categorical that contains zeros. \
                     We have removed zeros by adding a small non-negative scalar to each value."
            )

        if not self.IS_AOA:
            values = np.copy(self.values)
            log_values = np.log(values)
        else:
            log_values = np.empty(self.n_arrays, dtype="object")
            for i in range(self.n_arrays):
                values = np.copy(self.values[i])
                log_values[i] = np.log(values)

        if return_numpy:
            return log_values
        else:
            return Categorical(values=log_values)

    def copy(self):
        """Returns a copy of this object
        
        Returns
        ----------
        - `Categorical`
            A copy of this object
        """
        if not self.IS_AOA:
            values = np.copy(self.values)
        else:
            values = copy.deepcopy(self.values)
        return Categorical(values=values)

    def print_shape(self):
        """ Print shape of distribution """
        if not self.IS_AOA:
            print("Shape: {}".format(self.values.shape))
        else:
            string = [str(el.shape) for el in self.values]
            print("Shape: {} {}".format(self.values.shape, string))

    def sample(self):
        """ Sample from the distribution

        In the case that `IS_AOA` is true, a tuple of samples is returned for each array 
        @TODO sampling from arbitrary shape distributions  
        
        Returns
        ----------
        - `int` or `tuple`
            Samples from distribution
        """

        if not self.is_normalized():
            self.normalize()

        if self.IS_AOA:
            sample_array = np.zeros(self.n_arrays)
            for i in range(self.n_arrays):
                probabilities = np.copy(self.values[i])
                try:
                    sample_onehot = np.random.multinomial(1, probabilities.squeeze())
                except:
                    sample_onehot = np.random.multinomial(1, probabilities[0])
                sample_array[i] = np.where(sample_onehot == 1)[0][0]
            return tuple(sample_array.astype(int))
        else:
            if self.ndim != 2 or self.shape[1] != 1:
                raise ValueError("Can only currently sample from [n x 1] distribution")
            probabilities = np.copy(self.values)
            sample_onehot = np.random.multinomial(1, probabilities.squeeze())
            return np.where(sample_onehot == 1)[0][0]

    def plot(self, title=None, array_idx=None, leading_dim=0, index=0):
        """ Plot distribution

            @TODO currently doesn't work with AoA or arbitrary tensor 
            If you have AoA, use the `array_idx` arg to select an
            array to plot

            Leading dim [int] is used in `n_dim = 3` case, specifies whether
            plotting uses mat = x[0, :, :] / x[:, 1, :] / x[:, :, 2] 

            @TODO move plotting to utils, and generally destroy this function
        """

        if self.IS_AOA and array_idx is None:
            raise ValueError("Plotting AOA not implemented, see `array_idx` arg")
        values = self.values if array_idx is None else self.values[array_idx]

        if self.ndim == 2:
            plt.bar(values.shape[0], values)
            plt.yticks([], [])
            plt.title(title)
            plt.show()
        elif self.ndim == 3:
            # @TODO
            matrix = self._get_matrix_from_dim_index(values, leading_dim, index)
            plt.imshow(matrix, cmap="OrRd")
            plt.yticks([], [])
            plt.show()
        else:
            raise ValueError("Plotting for n_dim > 4 not implemented")

    def _get_matrix_from_dim_index(self, matrix, dim, i):
        if dim == 0:
            return matrix[i, :, :]
        elif dim == 1:
            return matrix[:, i, :]
        elif dim == 3:
            return matrix[:, :, i]
        else:
            raise ValueError()

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def shape(self):
        return self.values.shape

    def __add__(self, other):
        if isinstance(other, Categorical):
            values = self.values + other.values
            return Categorical(values=values)
        else:
            values = self.values + other
            return Categorical(values=values)

    def __radd__(self, other):
        if isinstance(other, Categorical):
            values = self.values + other.values
            return Categorical(values=values)
        else:
            values = self.values + other
            return Categorical(values=values)

    def __sub__(self, other):
        if isinstance(other, Categorical):
            values = self.values - other.values
            return Categorical(values=values)
        else:
            values = self.values - other
            return Categorical(values=values)

    def __rsub__(self, other):
        if isinstance(other, Categorical):
            values = self.values - other.values
            return Categorical(values=values)
        else:
            values = self.values - other
            return Categorical(values=values)

    def __mul__(self, other):
        if isinstance(other, Categorical):
            values = self.values * other.values
            return Categorical(values=values)
        else:
            values = self.values * other
            return Categorical(values=values)

    def __rmul__(self, other):
        if isinstance(other, Categorical):
            values = self.values * other.values
            return Categorical(values=values)
        else:
            values = self.values * other
            return Categorical(values=values)

    def __contains__(self, value):
        pass

    def __getitem__(self, key):
        values = self.values[key]
        if isinstance(values, np.ndarray):
            if values.ndim == 1:
                values = values[:, np.newaxis]
            return Categorical(values=values)
        else:
            return values

    def __setitem__(self, idx, value):
        if isinstance(value, Categorical):
            value = value.values
        self.values[idx] = value

    def __repr__(self):
        if not self.IS_AOA:
            return "<Categorical Distribution> \n {}".format(np.round(self.values, 3))
        else:
            string = [np.round(el, 3) for el in self.values]
            return "<Categorical Distribution> \n {}".format(string)
