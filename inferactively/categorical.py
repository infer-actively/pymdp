#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Categorical

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

import numpy as np
import warnings


class Categorical(object):
    """ A Categorical distribution

    A discrete probability distribution that describes the possible results of a random variable
    that can take on one of K possible categories.
    Parameters are all in the range 0 to 1, and all sum to 1

    This class assumes that the columns encode probability distributions
    This means that multiple columns represents a set of distributions
    TODO: `Describe what is happening with arrays of arrays`

    Attributes
    ----------
    values : np.ndarray
        The parameters of the Categorical distribution

    Methods
    -------

    """

    def __init__(self, dims=None, values=None):
        """Initialize a Categorical distribution

        `IS_AOA` refers to whether this class uses the `array-of-array` formulation

        Parameters
        ----------
        dims: list of int _or_ list of list (where each list is a list of int)
            Specify the number and size of dimensions
            This will initialize the parameters with zero values
        values: np.ndarray
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

    def construct_values(self, values):
        """Initialize a Categorical distribution with `values` argument

        Parameters
        ----------
        values: np.ndarray
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
        """Initialize a Categorical distribution with `values` argument

        TODO: allow other iterables (such as tuple)

        Parameters
        ----------
        dims: list of int
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

    def dot(self, x, dims_to_omit=None, return_numpy=False):
        """ Dot product of a Categorical distribution with `x`

        The dimensions in `dims_to_omit` will not be summed across during the dot product

        Parameters
        ----------
        x: 1d numpy.ndarray or Categorical
            The alternative array to perform the dot product with
        dims_to_omit: list (optional)
            a list of `ints` specifying which dimensions to omit
        return_numpy: Boolean (optional)
            whether the output will be a numpy `ndarray` or a `Categorical`
        """

        if self.IS_AOA:
            y = np.empty(len(self.values), dtype=object)
            for g in range(len(self.values)):
                X = self[g]
                y[g] = X.dot(x, dims_to_omit, return_numpy=True)
            if return_numpy:
                return y
            else:
                return Categorical(values=y)
        else:
            if isinstance(x, Categorical):
                x = x.values
            if x.dtype == object:
                dims = (np.arange(0, len(x)) + self.ndim - len(x)).astype(int)
            else:

                """
                This else handles cases when `x` is a numpy `ndarray`
                After first being extracted from a Categorical or as passed in initially
                The first dimension of `x` is likely the same as the first dimension of `A`
                E.g. inverting the generative model using observations
                Equivalent to something like self.values[np.where(x),:]
                When `x` is a discrete 'one-hot' observation vector
                """

                if x.shape[0] != self.shape[1]:
                    dims = np.array([0], dtype=int)
                else:

                    """
                    Case when `x` leading dimension matches the lagging dimension of `values`
                    E.g. a more 'classical' dot product of a likelihood with hidden states

                    """

                    dims = np.array([1], dtype=int)
                x_new = np.empty(1, dtype=object)
                x_new[0] = x.squeeze()
                x = x_new

            if dims_to_omit is not None:
                if not isinstance(dims_to_omit, list):
                    raise ValueError("dims_to_omit must be a :list:")
                dims = np.delete(dims, dims_to_omit)
                if len(x) == 1:
                    x = np.empty([0], dtype=object)
                else:
                    x = np.delete(x, dims_to_omit)

            y = self.values
            for d in range(len(x)):
                s = np.ones(np.ndim(y), dtype=int)
                s[dims[d]] = np.shape(x[d])[0]
                y = y * x[d].reshape(tuple(s))
                y = np.sum(y, axis=dims[d], keepdims=True)
            y = np.squeeze(y)

            if return_numpy:
                return y
            else:
                return Categorical(values=y)

    def cross(self, x=None, *args):
        """ Equivalent of spm_cross -- multidimensional outer product

        Parameters
        ----------
        x : numpy ndarray (including arrays-of-arrays), Categorical, or None (default):
            If None, the function simply returns the "auto-outer product" of self
            (i.e. self.cross(self.values))
            Otherwise (i.e. if x is an numpy ndarray or Categorical), the function will recursively
            take the outer product of the initial entry
            of x with self until it has depleted the possible entries of x
            (which become the next (x, *args) for future function calls) that it can outer-product.
        args : numpy ndarray (including arrays-of-arrays), Categorical or None:
            If an numpy ndarray (including dtype = object) or Categorical is passed
            into the function, it will take the outer product of the ndarray
            or the first entry of the array-object (and the corresponding values of
            respectively-constructed Categorical) with self. Otherwise,
            if this is None, then the result will simply be self.cross(x)
        Returns
        ________
        Y: always returns a numpy ndarray
        """

        if len(args) == 0 and x is None:

            if self.IS_AOA:
                """
                In the case that the Categorical is an AoA take the first array
                in the AoA and cross it against the remaining elements in the AoA
                """

                """
                TODO Two options:
                (A) The remaining elements of the AoA are passed in as AoA
                (numpy arrays with dtype == object).
                This would lead to be x being the first of the arrays in the AoAs,
                and *args would be the following ones
                """
                Y = self[0].cross(*list(self.values[1:]))
                """
                (B) The remaining elements of the AoA are passed in as an AoA Categorical.
                This would lead to x being a single Categorical variable in the
                second call to cross, and *args would be empty)
                """
                # Y = self[0].cross(self[1:])
            elif np.issubdtype(self.values.dtype, np.number):
                Y = self.values

            return Y

        if self.IS_AOA:
            """
            TODO Two options:
            (A) The remaining elements of the AoA are passed in as AoA
            (numpy arrays with dtype == object).
            This would lead to be x being the first of the arrays in the AoAs,
            and *args would be the following ones
            """
            X = self[0].cross(*list(self.values[1:]))
            """
            (B) The remaining elements of the AoA are passed in as an AoA Categorical.
            This would lead to x being a single Categorical variable in the
            second call to cross, and *args would be empty)
            """
            # X = self[0].cross(self[1:])
        else:
            X = self.values

        if x is not None:
            if isinstance(x, np.ndarray):
                if x.dtype == "object":
                    x_first = Categorical(values=x[0])
                    x = x_first.cross(*list(x[1:]))
            elif isinstance(x, Categorical):
                if x.IS_AOA:
                    """
                    TODO Two options:
                    (A) The remaining elements of the AoA are passed in as AoA
                    (numpy arrays with dtype == object).
                    This would lead to be x being the first of the arrays in the AoAs,
                    and *args would be the following ones
                    """
                    x = x[0].cross(*list(x.values[1:]))
                    """
                    (B) The remaining elements of the AoA are passed in as an AoA Categorical.
                    This would lead to x being a single Categorical variable in the
                    second call to cross, and *args would be empty)
                    """
                    # x = x[0].cross(x[1:])
                else:
                    x = x.values

        reshape_dims = tuple(list(X.shape) + list(np.ones(x.ndim, dtype=int)))
        A = X.reshape(reshape_dims)

        reshape_dims = tuple(list(np.ones(X.ndim, dtype=int)) + list(x.shape))
        B = x.reshape(reshape_dims)

        Y = np.squeeze(A * B)

        Y = Categorical(values=Y)
        for x in args:
            Y = Y.cross(x)

        if isinstance(Y, Categorical):
            Y = Y.values
            return Y
        else:
            return Y

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
        self.values += np.exp(-16)

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
            self.normalize()
            warnings.warn(
                "You have called :log: on a Categorical that contains zeros. \
                     We have removed zeros and normalized."
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
        """ Draws a sample from the distribution. Assumes a [n x 1] vector

        TODO: make this work with multi-dimension arrays
        """
        if self.ndim != 2 or self.shape[1] != 1:
            raise ValueError("can only currently sample from [n x 1] distribution")
        self.normalize()
        values = np.copy(self.values)
        samples = np.random.multinomial(1, values.squeeze())
        return np.where(samples == 1)[0][0]

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
