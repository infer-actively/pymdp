#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Environment Base Class

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""


class Env(object):
    action_space = None
    observation_space = None

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed=None):
        return

    def get_likelihood_dist(self):
        pass

    def get_transition_dist(self):
        pass

    def __str__(self):
        return "<{} instance>".format(type(self).__name__)
