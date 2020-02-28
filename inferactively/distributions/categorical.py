#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Categorical
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import numpy as np
import warnings
import inferactively.core as F

class Categorical(object):
    """ A Categorical distribution
    A discrete probability distribution over K possible events.
    Parameters are in the range 0 to 1 and sum to 1.
    This class assumes that columns encode probability distributions.
    Multiple columns represents a set of distributions.
    This can be useful for representing conditional distributions.
    @TODO: Describe what is happening with `arrays of arrays`
    
    """

    def __init__(self, dims=None, values=None):
        """Initialize a Categorical distribution
        Parameters
        ----------
        `dims` [list :: int] || [list :: list] 
            Specifies the number and size of dimensions
        `values` [np.ndarray]
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

    def construct_values(self, values):
        """Initialize a Categorical distribution with `values` argument
        Parameters
        ----------
        `values` [np.ndarray]
            The parameters of the distribution
        """

        if not isinstance(values, np.ndarray):
            raise ValueError("`values` must be a `numpy.ndarray`")

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
        """Initialize a Categorical distribution with `dims` argument
        Note we initialise distributions with zero values
        Parameters
        ----------
        `dims` [list :: int]
            Specify the number and size of dimensions
        """

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

    def dot(self, x, dims_to_omit=None, return_numpy=False, obs_mode = False):
        """ Dot product of a Categorical distribution with `x`
        The dimensions in `dims_to_omit` will not be summed across during the dot product
        Parameters
        ----------
        `x` [1D numpy.ndarray] || [Categorical]
            The alternative array to perform the dot product with
        `dims_to_omit` [list :: int] (optional)
            Which dimensions to omit
        `return_numpy` [bool] (optional)
            Whether to return `np.ndarray` or `Categorical`
        'obs_mode' [bool] (optional)
            Whether to perform the inner product of 'x' with the leading dimension of self. 
            We call this 'obs_mode' because it's often used to get the likelihood of an observation (leading dimension)
            under different settings of hidden states (lagging dimensions)
        """

        if isinstance(x, Categorical):
            x = x.values

        # perform dot product on each sub-array
        if self.IS_AOA:
            y = np.empty(len(self.values), dtype=object)
            for g in range(len(self.values)):
                X = self[g].values
                y[g] = F.spm_dot(X, x, dims_to_omit, obs_mode)
        else:
            X = self.values
            y = F.spm_dot(X, x, dims_to_omit, obs_mode)

        if return_numpy:
            return y
        else:
            return Categorical(values=y)

    def cross(self, x=None, return_numpy=False, *args):
        """ Multi-dimensional outer product
        If no `x` argument is passed, the function returns the "auto-outer product" of self
        Otherwise, the function will recursively take the outer product of the initial entry
        of `x` with `self` until it has depleted the possible entries of `x` that it can outer-product
        Parameters
        ----------
        `x` [np.ndarray] || [Categorical] (optional)
            The values to perform the outer-product with
        `args` [np.ndarray] || [Categorical] (optional)
            Perform the outer product of the `args` with self
       
        Returns
        -------
        `y` [np.ndarray] || [Categorical]
            The result of the outer-product
        """

        X = self.values

        if isinstance(x, Categorical):
            x = x.values
        if x is not None and len(args) > 0 and isinstance(args[0], Categorical):
            args_arrays = []
            for i in args:
                args_arrays.append(i.values)
            Y = F.spm_cross(X, x, *args_arrays)
        else:
            Y = F.spm_cross(X, x, *args)

        if return_numpy:
            return Y
        else:
            return Categorical(values=Y)

    def normalize(self):
        """ Normalize distribution
        This function will ensure the distribution(s) integrate to 1.0
        In the case `ndims` >= 2, normalization is performed along the columns of the arrays
        """
        if self.is_normalized():
            return
        if self.IS_AOA:
            for i in range(len(self.values)):
                array_i = self.values[i]
                column_sums = np.sum(array_i, axis=0)
                array_i = np.divide(array_i, column_sums)
                array_i[np.isnan(array_i)] = np.divide(1.0, array_i.shape[0])
                self.values[i] = array_i
        else:
            column_sums = np.sum(self.values, axis=0)
            self.values = np.divide(self.values, column_sums)
            self.values[np.isnan(self.values)] = np.divide(1.0, self.values.shape[0])

    def is_normalized(self):
        """ Checks whether columns sum to 1
        Note this operates within some margin of error (10^-4)
        """
        if self.IS_AOA:
            array_is_normed = np.zeros(len(self.values), dtype=bool)
            for i in range(len(self.values)):
                error = np.abs(1 - np.sum(self.values[i], axis=0))
                array_is_normed[i] = (error < 0.0001).all()
            return array_is_normed.all()
        else:
            error = np.abs(1 - np.sum(self.values, axis=0))
            return (error < 0.0001).all()

    def remove_zeros(self):
        """ Remove zeros by adding a small number
        This function avoids division by zero
        exp(-16) is used as the minimum value
        """
        # self.values += np.exp(-16)
        self.values += 1e-16
        
    def contains_zeros(self):
        """ Checks if any values are zero
        Returns
        ----------
        bool
            Whether there are any zeros
        """
        if not self.IS_AOA:
            return (self.values == 0.0).any()
        else:
            for i in range(len(self.values)):
                if (self.values[i] == 0.0).any():
                    return True
            return False

    def entropy(self, return_numpy=False):
        """ Return the entropy of each column
        Parameters
        ----------
        return_numpy: bool
            Whether to return a :np.ndarray: or :Categorical: object
        Returns
        ----------
        np.ndarray or Categorical
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
            entropy = np.empty(len(self.values), dtype="object")
            for i in range(len(self.values)):
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
        return_numpy: bool
            Whether to return a :np.ndarray: or :Categorical: object
        Returns
        ----------
        np.ndarray or Categorical
            The log of the parameters
        """

        if self.contains_zeros():
            self.remove_zeros()
            # self.normalize()
            # warnings.warn(
            #     "You have called :log: on a Categorical that contains zeros. \
            #          We have removed zeros and normalized."
            # )
            warnings.warn(
                "You have called :log: on a Categorical that contains zeros. \
                     We have removed zeros by adding a small non-negative scalar to each value."
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
            return Categorical(values=log_values)

    def copy(self):
        """Returns a copy of this object
        Returns
        ----------
        Categorical
            Returns a copy of this object
        """
        values = np.copy(self.values)
        return Categorical(values=values)

    def print_shape(self):
        if not self.IS_AOA:
            print("Shape: {}".format(self.values.shape))
        else:
            string = [str(el.shape) for el in self.values]
            print("Shape: {} {}".format(self.values.shape, string))

    def sample(self):
        """ Draws a sample from a Categorical distribution or set of samples from a set of distributions 
            (in case that self.IS_AOA is True)
        """

        if not self.is_normalized():
            self.normalize()

        if self.IS_AOA:
            """ @TODO: In case of self.IS_AOA, how should we store a multinomial sample - a list, an array of arrays, a simple 1D array...?
                Here, we decided to use a tuple of integers. This can be revised, however."""
            sample_array = np.zeros(len(self.values))
            for i in range(len(self.values)):
                probabilities = np.copy(self.values[i])
                try:
                    sample_onehot = np.random.multinomial(1, probabilities.squeeze())
                except:
                    sample_onehot = np.random.multinomial(1, probabilities[0])
                sample_array[i] = np.where(sample_onehot == 1)[0][0]
            # returning a tuple of indices is good in the case when you're sampling observations - consistent with update_posterior_states function
            return tuple(sample_array.astype(int))
        else:
            if self.ndim != 2 or self.shape[1] != 1:
                raise ValueError("Can only currently sample from [n x 1] distribution")
            probabilities = np.copy(self.values)
            sample_onehot = np.random.multinomial(1, probabilities.squeeze())
            return np.where(sample_onehot == 1)[0][0]

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
