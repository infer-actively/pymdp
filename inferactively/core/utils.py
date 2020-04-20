import numpy as np
from inferactively.distributions import Categorical, Dirichlet

def to_numpy(values):
    if isinstance(values, Categorical):
        values = np.copy(values.values)

    if isinstance(values, Dirichlet):
        values = np.copy(values.values)

    return values

def is_arr_of_arr(arr):
    return arr.dtype == "object"


def to_categorical(values):
    return Categorical(values=values)