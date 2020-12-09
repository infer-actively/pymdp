#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

""" Generic functions
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import itertools
import numpy as np
import torch
from scipy import special
from pymdp.core import utils
from pymdp.distributions import Dirichlet, Categorical


def spm_dot(X, y, dims_to_omit=None, obs_mode=False):
    """ Dot product of a multidimensional array `X` with `y`
    The dimensions in `dims_to_omit` will not be summed across during  dot product
   
    @TODO: we need documentation describing `obs_mode`
        Ideally, we could find a way to avoid it altogether 

    Parameters
    ----------
    `y` [1D numpy.ndarray] 
        Either vector or array of arrays
    `dims_to_omit` [list :: int] (optional) 
        Which dimensions to omit
    """

    X = utils.to_numpy(X)
    y = utils.to_numpy(y)

    # if `X` is array of array, we need to construct the dims to sum
    if utils.is_arr_of_arr(X):
        dims = (np.arange(0, len(y)) + X.ndim - len(y)).astype(int)
    else:
        """ 
        Deal with particular use case - see above @TODO 
        """
        if obs_mode is True:
            """
            Case when you're getting the likelihood of an observation under model.
            Equivalent to something like self.values[np.where(x),:]
            where `y` is a discrete 'one-hot' observation vector
            """
            dims = np.array([0], dtype=int)
        else:
            """
            Case when `y` leading dimension matches the lagging dimension of `values`
            E.g. a more 'classical' dot product of a likelihood with hidden states
            """
            dims = np.array([1], dtype=int)

        # convert `y` to array of array
        y = utils.to_arr_of_arr(y)

    # omit dims not needed for dot product
    if dims_to_omit is not None:
        if not isinstance(dims_to_omit, list):
            raise ValueError("`dims_to_omit` must be a `list` of `int`")

        # delete dims
        dims = np.delete(dims, dims_to_omit)
        if len(y) == 1:
            y = np.empty([0], dtype=object)
        else:
            y = np.delete(y, dims_to_omit)

    print(dims)
    # perform dot product
    for d in range(len(y)):
        s = np.ones(np.ndim(X), dtype=int)
        s[dims[d]] = np.shape(y[d])[0]
        X = X * y[d].reshape(tuple(s))
        X = np.sum(X, axis=dims[d], keepdims=True)
    X = np.squeeze(X)

    # perform check to see if `x` is a scalar
    if np.prod(X.shape) <= 1.0:
        X = X.item()
        X = np.array([X]).astype("float64")

    return X


def softmax(dist, return_numpy=True):
    """ 
    Computes the softmax function on a set of values
    """

    dist = utils.to_numpy(dist)

    output = []
    if utils.is_arr_of_arr(dist):
        for i in range(len(dist.values)):
            output.append(softmax(dist[i]), return_numpy=True)

    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    if return_numpy:
        return output
    else:
        return utils.to_categorical(output)


def kl_divergence(q, p):
    """
    TODO: make this work for multi-dimensional arrays
    """
    if not isinstance(type(q), type(Categorical)) or not isinstance(type(p), type(Categorical)):
        raise ValueError("`kl_divergence` function takes `Categorical` objects")
    q.remove_zeros()
    p.remove_zeros()
    q = np.copy(q.values)
    p = np.copy(p.values)
    kl = np.sum(q * np.log(q / p), axis=0)[0]
    return kl

    # perform check to see if result  is a number
    if np.prod(a.shape) <= 1.0:
        a = a.item()
        a = np.array([a]).astype("float64")

    return a


def spm_dot_torch(X, x, dims_to_omit=None):
    """ Dot product of a multidimensional array with `x` -- Pytorch version, using Tensor instances
    @TODO: Instead of a separate function, this should be integrated with spm_dot so that it can either take torch.Tensors or nd.arrays

    The dimensions in `dims_to_omit` will not be summed across during the dot product

    Parameters
    ----------
    'X' [torch.Tensor]
    `x` [1D torch.Tensor or numpy object array containing 1D torch.Tensors]
        The array(s) to dot X with
    `dims_to_omit` [list :: int] (optional)
        Which dimensions to omit from summing across
    """

    if x.dtype == object:
        dims = (np.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    else:
        if x.shape[0] != X.shape[1]:
            """
            Case when the first dimension of `x` is likely the same as the first dimension of `A`
            e.g. inverting the generative model using observations.
            Equivalent to something like self.values[np.where(x),:]
            when `x` is a discrete 'one-hot' observation vector
            """
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
            raise ValueError("dims_to_omit must be a `list`")
        dims = np.delete(dims, dims_to_omit)
        if len(x) == 1:
            x = np.empty([0], dtype=object)
        else:
            x = np.delete(x, dims_to_omit)

    Y = X
    for d in range(len(x)):
        s = np.ones(Y.ndim, dtype=int)
        s[dims[d]] = max(x[d].shape)
        Y = Y * x[d].view(tuple(s))
        Y = Y.sum(dim=int(dims[d]), keepdim=True)
    Y = Y.squeeze()

    # perform check to see if `y` is a number
    if Y.numel() <= 1:
        Y = np.asscalar(Y)
        Y = torch.Tensor([Y])

    return Y


def spm_cross(X, x=None, *args):
    """ Multi-dimensional outer product
    If no `x` argument is passed, the function returns the "auto-outer product" of self
    Otherwise, the function will recursively take the outer product of the initial entry
    of `x` with `self` until it has depleted the possible entries of `x` that it can outer-product
    Parameters
    ----------
    `x` [np.ndarray] || [Categorical] (optional)
        The values to perfrom the outer-product with
    `args` [np.ndarray] || [Categorical] (optional)
        Perform the outer product of the `args` with self
    
    Returns
    -------
    `y` [np.ndarray] || [Categorical]
        The result of the outer-product
    """

    if len(args) == 0 and x is None:
        if X.dtype == "object":
            Y = spm_cross(*list(X))

        elif np.issubdtype(X.dtype, np.number):
            Y = X

        return Y

    if X.dtype == "object":
        X = spm_cross(*list(X))

    if x is not None and x.dtype == "object":
        x = spm_cross(*list(x))

    reshape_dims = tuple(list(X.shape) + list(np.ones(x.ndim, dtype=int)))
    A = X.reshape(reshape_dims)

    reshape_dims = tuple(list(np.ones(X.ndim, dtype=int)) + list(x.shape))
    B = x.reshape(reshape_dims)

    Y = np.squeeze(A * B)

    for x in args:
        Y = spm_cross(Y, x)

    return Y


def spm_wnorm(A):
    """
    Normalization of a prior over Dirichlet parameters, used in updates for information gain
    """

    A = A + 1e-16

    norm = np.divide(1.0, np.sum(A, axis=0))

    avg = np.divide(1.0, A)

    wA = norm - avg

    return wA


def spm_betaln(z):
    """
    Returns the log of the multivariate beta function of a vector.
    FORMAT y = spm_betaln(z)
     y = spm_betaln(z) computes the natural logarithm of the beta function
     for corresponding elements of the vector z. if concerned is a matrix,
     the logarithm is taken over the columns of the matrix z.
    """

    y = np.sum(special.gammaln(z), axis=0) - special.gammaln(np.sum(z, axis=0))

    return y
