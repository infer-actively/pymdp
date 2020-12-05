#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Utils

__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import itertools
import numpy as np
from inferactively.distributions import Categorical, Dirichlet


def to_numpy(dist, flatten=False):
    """
    If flatten is True, then the individual entries of the object array will be 
    flattened into row vectors(common operation when dealing with array of arrays 
    with 1D numpy array entries)
    """
    if isinstance(dist, (Categorical, Dirichlet)):
        values = np.copy(dist.values)
        if flatten:
            if dist.IS_AOA:
                for idx, arr in enumerate(values):
                    values[idx] = arr.flatten()
            else:
                values = values.flatten()
    else:
        values = dist
        if flatten:
            if is_arr_of_arr(values):
                for idx, arr in enumerate(values):
                    values[idx] = arr.flatten()
            else:
                values = values.flatten()
    return values


def is_distribution(obj):
    return isinstance(obj, (Categorical, Dirichlet))


def is_arr_of_arr(arr):
    return arr.dtype == "object"


def to_arr_of_arr(arr):
    """ @TODO make general """
    arr_of_arr = np.empty(1, dtype=object)
    arr_of_arr[0] = arr.squeeze()
    return arr_of_arr


def to_categorical(values):
    return Categorical(values=values)


def to_dirichlet(values):
    return Dirichlet(values=values)

