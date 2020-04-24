#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-an-iterable

""" Utils
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import itertools
import numpy as np
from inferactively.distributions import Categorical, Dirichlet


def to_numpy(dist):
    if isinstance(dist, Categorical):
        values = np.copy(dist.values)
    elif isinstance(dist, Dirichlet):
        values = np.copy(dist.values)
    else:
        values = dist
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

