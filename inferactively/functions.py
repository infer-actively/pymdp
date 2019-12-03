#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Functions

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

import itertools
import numpy as np
from . import Categorical


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
    policies: `list`
        A list of tuples, each specifying a list of actions (`int`)

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
        raise ValueError("[kl_divergence] function takes [Categorical] objects")
    q.remove_zeros()
    p.remove_zeros()
    q = np.copy(q.values)
    p = np.copy(p.values)
    kl = np.sum(q * np.log(q / p), axis=0)[0]
    return kl
