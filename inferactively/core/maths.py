#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

""" Functions
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import itertools
import numpy as np
import torch
from scipy import special
from inferactively.core import utils


def spm_dot(X, x, dims_to_omit=None, obs_mode=False):
    """ Dot product of a multidimensional array with `x`
        The dimensions in `dims_to_omit` will not be summed across during the dot product

    @TODO: we should look for an alternative to obs_mode
    
    Parameters
    ----------
    - `x` [1D numpy.ndarray] - either vector or array of arrays
        The alternative array to perform the dot product with
    - `dims_to_omit` [list :: int] (optional)
        Which dimensions to omit
    
    Returns 
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """

    # construct dims to perform dot product on
    if utils.is_arr_of_arr(x):
        dims = (np.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    else:
        if obs_mode is True:
            """
            @NOTE Case when you're getting the likelihood of an observation under the generative model.
                  Equivalent to something like self.values[np.where(x),:]
                  when `x` is a discrete 'one-hot' observation vector
            """
            dims = np.array([0], dtype=int)
        else:
            """
            @NOTE Case when `x` leading dimension matches the lagging dimension of `values`
                  E.g. a more 'classical' dot product of a likelihood with hidden states
            """
            dims = np.array([1], dtype=int)

        x = utils.to_arr_of_arr(x)

    # delete ignored dims
    if dims_to_omit is not None:
        if not isinstance(dims_to_omit, list):
            raise ValueError("`dims_to_omit` must be a `list` of `int`")
        dims = np.delete(dims, dims_to_omit)
        if len(x) == 1:
            x = np.empty([0], dtype=object)
        else:
            x = np.delete(x, dims_to_omit)

    # compute dot product
    for d in range(len(x)):
        s = np.ones(np.ndim(X), dtype=int)
        s[dims[d]] = np.shape(x[d])[0]
        X = X * x[d].reshape(tuple(s))
        X = np.sum(X, axis=dims[d], keepdims=True)
    Y = np.squeeze(X)

    # check to see if `Y` is a scalar
    if np.prod(Y.shape) <= 1.0:
        Y = Y.item()
        Y = np.array([Y]).astype("float64")

    return Y


def spm_cross(X, x=None, *args):
    """ Multi-dimensional outer product
    
        @NOTE: If no `x` argument is passed, the function returns the "auto-outer product" of self
               Otherwise, the function recursively take the outer product of the initial entry
               of `x` with `self` until it has depleted the possible entries of `x` that it can outer-product

    @TODO the `args` parameter is a bit confusing, we could maybe make this clearer

    Parameters
    ----------
    - `x` [np.ndarray] || [Categorical] (optional)
        The values to perfrom the outer-product with
    `args` [np.ndarray] || [Categorical] (optional)
        Perform the outer product of the `args` with self
    
    Returns
    -------
    - `y` [np.ndarray] || [Categorical]
          The result of the outer-product
    """

    if len(args) == 0 and x is None:
        if utils.is_arr_of_arr(X):
            Y = spm_cross(*list(X))
        elif np.issubdtype(X.dtype, np.number):
            Y = X
        return Y

    if utils.is_arr_of_arr(X):
        X = spm_cross(*list(X))

    if x is not None and utils.is_arr_of_arr(x):
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
    """ Normalize Dirichlet parameters

    @TODO need to update the description of this function
    """
    A = A + 1e-16
    norm = np.divide(1.0, np.sum(A, axis=0))
    avg = np.divide(1.0, A)
    wA = norm - avg
    return wA


def spm_betaln(z):
    """ Log of the multivariate beta function of a vector.

     @NOTE this function computes across columns if `z` is a matrix
    """
    return np.sum(special.gammaln(z), axis=0) - special.gammaln(np.sum(z, axis=0))


def calc_free_energy(qs, prior, n_factors, likelihood=None):
    """ Calculate variational free energy

    @TODO Primarily used in FPI algorithm, needs to be made general
    """
    free_energy = 0
    for factor in range(n_factors):
        term_a = -qs[factor].dot(np.log(qs[factor][:, np.newaxis] + 1e-16))
        term_b = -qs[factor].dot(prior[factor][:, np.newaxis])
        free_energy += term_a + term_b

    if likelihood is not None:
        accuracy = spm_dot(likelihood, qs)[0]
        free_energy -= accuracy
    return free_energy


def softmax(dist, return_numpy=True):
    """ Computes the softmax function on a set of values

    """
    if utils.is_distribution(dist):
        if dist.IS_AOA:
            output = []
            for i in range(len(dist.values)):
                output[i] = softmax(dist.values[i], return_numpy=True)
            output = utils.to_categorical(np.array(output))
        else:
            dist = np.copy(dist.values)
    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    if return_numpy:
        return output
    else:
        return utils.to_categorical(output)


def kl_divergence(q, p):
    """ Calculate KL divdivergence between two distributions

    @TODO: make this work for multi-dimensional arrays
    """
    q.remove_zeros()
    p.remove_zeros()
    q = np.copy(q.values)
    p = np.copy(p.values)
    kl = np.sum(q * np.log(q / p), axis=0)[0]
    return kl
