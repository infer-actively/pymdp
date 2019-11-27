#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Categorical

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""


class Categorical(object):
    def __init__(self):
        pass

    def normalize(self):
        pass

    def is_normalized(self):
        pass

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
        return None

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
        pass

    def __setitem__(self, idx, value):
        pass

    def __repr__(self):
        return "-"
