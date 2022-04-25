#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Environment Base Class

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""


class Env(object):
    """ 
    The Env base class, loosely-inspired by the analogous ``env`` class of the OpenAIGym framework. 

    A typical workflow is as follows:

    >>> my_env = MyCustomEnv(<some_params>)
    >>> initial_observation = my_env.reset(initial_state)
    >>> my_agent.infer_states(initial_observation)
    >>> my_agent.infer_policies()
    >>> next_action = my_agent.sample_action()
    >>> next_observation = my_env.step(next_action)

    This would be the first step of an active inference process, where a sub-class of ``Env``, ``MyCustomEnv`` is initialized, 
    an initial observation is produced, and these observations are fed into an instance of ``Agent`` in order to produce an action,
    that can then be fed back into the the ``Env`` instance.

    """

    def reset(self, state=None):
        """
        Resets the initial state of the environment. Depending on case, it may be common to return an initial observation as well.
        """
        raise NotImplementedError

    def step(self, action):
        """
        Steps the environment forward using an action.

        Parameters
        ----------
        action
            The action, the type/format of which depends on the implementation.

        Returns
        ---------
        observation
            Sensory observations for an agent, the type/format of which depends on the implementation of ``step`` and the observation space of the agent.
        """
        raise NotImplementedError

    def render(self):
        """
        Rendering function, that typically creates a visual representation of the state of the environment at the current timestep.
        """
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
