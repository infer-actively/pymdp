#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Categorical

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

import numpy as np


class Categorical(object):
    """ A Categorical distribution

    A discrete probability distribution that describes the possible results of a random variable
    that can take on one of K possible categories.
    Parameters are all in the range 0 to 1, and all sum to 1

    This class assumes that the columns encode probability distributions
    This means that multiple columns represents a set of distributions
    TODO: Describe what is happening with arrays of arrays

    ...

    Attributes
    ----------
    values : np.ndarray
        The parameters of the Categorical distribution
    TODO:

    Methods
    -------
    TODO:

    """

    def __init__(self, dims=None, values=None):
        """Initialize a Categorical distribution

        Note: _either_ dims or values should be provided, providing both will raise an error
        Note: for single dimension :values:, the dimension is changed to [n x 1]
        Note: if neither :dims: or :values: is provided, a `dims=[1, 1]` distribution is created
        TODO: Describe what is happening with arrays of arrays

        Parameters
        ----------
        dims: list of int OR list of lists (where each list is a list of ints)
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

        if dims is not None:
            if isinstance(dims, list):
                if any(isinstance(el, list) for el in dims):
                    if not all(isinstance(el, list) for el in dims):
                        raise ValueError("you have mixed :int: and :list: in your list of :dims:")
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

    def dot(self,x,dims2omit):
        """
        Dot product of the Categorical distribution with a vector of set of vectors x, with optional argument 'dims2omit,' which specifies
        which dimensions will not be summed out during the dot product. 
        """



    def normalize(self):
        """
        Normalization Categorical distribution of set of Categorical distributions so that they are probability distributions (integrate to 1.0)
        In the case of ndarrays with ndims >= 2, normalization is performed along the columns of the respective array or set of arrays.
        """
        if self.is_normalized():
            print('This array is already normalized!')
            return
        if self.IS_AOA:
            for i in range(len(self.values)):
                array_i = self.values[i]
                column_sums = np.sum(array_i, axis=0)
                array_i = np.divide(array_i, column_sums)
                array_i[np.isnan] = np.divide(1.0,array_i.shape[0])
                self.values[i] = array_i
        else:
            column_sums = np.sum(self.values, axis=0)
            self.values = np.divide(self.values, column_sums)
            self.values[np.isnan] = np.divide(1.0, self.values.shape[0])

    def is_normalized(self):
        """
        Checks if a Cateogrical distribution or set of such distributions are normalized, 
        and returns True if this is satisfied for all such distributions, and False if any distribution is not normalized.
        """
        if self.IS_AOA:
            array_isNormed = np.zeros(len(self.values),dtype=bool)
            for i in range(len(self.values)):
                error = np.abs(1 - np.sum(self.values[i], axis=0))
                array_isNormed[i] =  (error < 0.0001).all()
            return array_isNormed.all()
        else:
            error = np.abs(1 - np.sum(self.values, axis=0))
            return (error < 0.0001).all()

    def remove_zeros(self):
        pass

    def contains_zeros(self):
        pass

    def entropy(self, return_numpy=False):
        pass

    def log(self, return_numpy=False):
        pass

    def dot(self, other, return_numpy=False):
        pass

    def copy(self):
        pass

    @property
    def ndim(self):
        return None

    @property
    def shape(self):
        return self.values.shape

    def __add__(self, other):
        pass

    def __radd__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __rsub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __rmul__(self, other):
        pass

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
        """ TODO:

        """
        return str(self.values)
