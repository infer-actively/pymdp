#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Environment Base Class

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""


class Env(object):
    def reset(self, state=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        pass

    def sample_action(self):
        pass

    def get_likelihood_dist(self):
        raise ValueError(
            "<{}> does not provide a model specification".format(type(self).__name__)
        )

    def get_transition_dist(self):
        raise ValueError(
            "<{}> does not provide a model specification".format(type(self).__name__)
        )

    def get_uniform_posterior(self):
        raise ValueError(
            "<{}> does not provide a model specification".format(type(self).__name__)
        )

    def get_rand_likelihood_dist(self):
        raise ValueError(
            "<{}> does not provide a model specification".format(type(self).__name__)
        )

    def get_rand_transition_dist(self):
        raise ValueError(
            "<{}> does not provide a model specification".format(type(self).__name__)
        )

    def __str__(self):
        return "<{} instance>".format(type(self).__name__)
