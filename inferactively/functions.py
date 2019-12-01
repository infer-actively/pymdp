#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Functions

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

import numpy as np
from . import Categorical


def softmax(values):
    if isinstance(values, Categorical):
        values = np.copy(values.values)
    values = values - values.max()
    values = np.exp(values)
    values = values / np.sum(values)
    return Categorical(values=values)
