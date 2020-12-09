#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Cube world environment

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pymdp.envs import Env


class GridWorldEnv(Env):
    """ Basic grid-world implementation """

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4

    CONTROL_NAMES = ["UP", "RIGHT", "DOWN", "LEFT", "STAY"]

    def __init__(self, shape=[2, 2], init_state=None):
        self.shape = shape
        self.n_states = np.prod(shape)
        self.n_observations = self.n_states
        self.n_control = 5
        self.max_y = shape[0]
        self.max_x = shape[1]
        self._build()
        self.set_init_state(init_state)
        self.last_action = None

    def reset(self, init_state=None):
        self.set_init_state(init_state)
        self.last_action = None
        return self.state

    def set_state(self, state):
        self.state = state
        return state

    def step(self, action):
        state = self.P[self.state][action]
        self.state = state
        self.last_action = action
        return state

    def render(self, title=None):
        values = np.zeros(self.shape)
        values[self.position] = 1.0
        _, ax = plt.subplots(figsize=(3, 3))
        if self.shape[0] is 1 or self.shape[1] is 1:
            ax.imshow(values, cmap="OrRd")
        else:
            _ = sns.heatmap(values, cmap="OrRd", linewidth=2.5, cbar=False, ax=ax)
        plt.xticks(range(self.shape[1]))
        plt.yticks(range(self.shape[0]))
        if title is not None:
            plt.title(title)
        plt.show()

    def set_init_state(self, init_state=None):
        if init_state is not None:
            if init_state > (self.n_states - 1) or init_state < 0:
                raise ValueError("`init_state` is greater than number of states")
            if not isinstance(init_state, (int, float)):
                raise ValueError("`init_state` must be [int/float]")
            self.init_state = int(init_state)
        else:
            self.init_state = np.random.randint(0, self.n_states)
        self.state = self.init_state

    def _build(self):
        P = {}
        grid = np.arange(self.n_states).reshape(self.shape)
        it = np.nditer(grid, flags=["multi_index"])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            P[s] = {a: [] for a in range(self.n_control)}

            next_up = s if y == 0 else s - self.max_x
            next_right = s if x == (self.max_x - 1) else s + 1
            next_down = s if y == (self.max_y - 1) else s + self.max_x
            next_left = s if x == 0 else s - 1
            next_stay = s

            P[s][self.UP] = next_up
            P[s][self.RIGHT] = next_right
            P[s][self.DOWN] = next_down
            P[s][self.LEFT] = next_left
            P[s][self.STAY] = next_stay

            it.iternext()

        self.P = P

    def get_init_state_dist(self, init_state=None):
        init_state_dist = np.zeros(self.n_states)
        if init_state is None:
            init_state_dist[self.init_state] = 1.0
        else:
            init_state_dist[init_state] = 1.0

    def get_transition_dist(self):
        B = np.zeros([self.n_states, self.n_states, self.n_control])
        for s in range(self.n_states):
            for a in range(self.n_control):
                ns = int(self.P[s][a])
                B[ns, s, a] = 1
        return B

    def get_likelihood_dist(self):
        A = np.eye(self.n_observations, self.n_states)
        return A

    def sample_action(self):
        return np.random.randint(self.n_control)

    @property
    def position(self):
        """ @TODO might be wrong w.r.t (x & y) """
        return np.unravel_index(np.array(self.state), self.shape)


class DGridWorldEnv(object):
    """ Only one dimension (three actions) """

    LEFT = 0
    STAY = 1
    RIGHT = 2

    CONTROL_NAMES = ["LEFT", "STAY", "RIGHT"]

    def __init__(self, shape=[2, 2], init_state=None):
        self.shape = shape
        self.n_states = np.prod(shape)
        self.n_observations = self.n_states
        self.n_control = 3
        self.max_y = shape[0]
        self.max_x = shape[1]
        self._build()
        self.set_init_state(init_state)
        self.last_action = None

    def reset(self, init_state=None):
        self.set_init_state(init_state)
        self.last_action = None
        return self.state

    def set_state(self, state):
        self.state = state
        return state

    def step(self, action):
        state = self.P[self.state][action]
        self.state = state
        self.last_action = action
        return state

    def render(self, title=None):
        values = np.zeros(self.shape)
        values[self.position] = 1.0
        _, ax = plt.subplots(figsize=(3, 3))
        if self.shape[0] is 1 or self.shape[1] is 1:
            ax.imshow(values, cmap="OrRd")
        else:
            _ = sns.heatmap(values, cmap="OrRd", linewidth=2.5, cbar=False, ax=ax)
        plt.xticks(range(self.shape[1]))
        plt.yticks(range(self.shape[0]))
        if title is not None:
            plt.title(title)
        plt.show()

    def set_init_state(self, init_state=None):
        if init_state is not None:
            if init_state > (self.n_states - 1) or init_state < 0:
                raise ValueError("`init_state` is greater than number of states")
            if not isinstance(init_state, (int, float)):
                raise ValueError("`init_state` must be [int/float]")
            self.init_state = int(init_state)
        else:
            self.init_state = np.random.randint(0, self.n_states)
        self.state = self.init_state

    def _build(self):
        P = {}
        grid = np.arange(self.n_states).reshape(self.shape)
        it = np.nditer(grid, flags=["multi_index"])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            P[s] = {a: [] for a in range(self.n_control)}

            next_right = s if x == (self.max_x - 1) else s + 1
            next_left = s if x == 0 else s - 1
            next_stay = s

            P[s][self.LEFT] = next_left
            P[s][self.STAY] = next_stay
            P[s][self.RIGHT] = next_right

            it.iternext()

        self.P = P

    def get_init_state_dist(self, init_state=None):
        init_state_dist = np.zeros(self.n_states)
        if init_state is None:
            init_state_dist[self.init_state] = 1.0
        else:
            init_state_dist[init_state] = 1.0

    def get_transition_dist(self):
        B = np.zeros([self.n_states, self.n_states, self.n_control])
        for s in range(self.n_states):
            for a in range(self.n_control):
                ns = int(self.P[s][a])
                B[ns, s, a] = 1
        return B

    def get_likelihood_dist(self):
        A = np.eye(self.n_observations, self.n_states)
        return A

    def sample_action(self):
        return np.random.randint(self.n_control)

    @property
    def position(self):
        return self.state
