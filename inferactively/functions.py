#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Functions

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

import itertools
import numpy as np
from inferactively.distributions import Categorical


def softmax(values, return_numpy=False):
    """ Computes the softmax function on a set of values

    TODO: make this work for multi-dimensional arrays

    """
    if isinstance(values, Categorical):
        values = np.copy(values.values)
    values = values - values.max()
    values = np.exp(values)
    values = values / np.sum(values)
    if return_numpy:
        return values
    else:
        return Categorical(values=values)


def generate_policies(n_actions, policy_len):
    """ Generate of possible combinations of N actions for policy length T

    Returns
    -------
    `policies` [list]
        A list of tuples, each specifying a list of actions [int]

    """

    x = [n_actions] * policy_len
    return list(itertools.product(*[list(range(i)) for i in x]))


def kl_divergence(q, p):
    """
    TODO: make this work for multi-dimensional arrays
    """
    if not isinstance(type(q), type(Categorical)) or not isinstance(
        type(p), type(Categorical)
    ):
        raise ValueError("`kl_divergence` function takes `Categorical` objects")
    q.remove_zeros()
    p.remove_zeros()
    q = np.copy(q.values)
    p = np.copy(p.values)
    kl = np.sum(q * np.log(q / p), axis=0)[0]
    return kl


def spm_dot(X, x, dims_to_omit):
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
        s = np.ones(np.ndim(Y), dtype=int)
        s[dims[d]] = np.shape(x[d])[0]
        Y = Y * x[d].reshape(tuple(s))
        Y = np.sum(Y, axis=dims[d], keepdims=True)
    Y = np.squeeze(Y)

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
