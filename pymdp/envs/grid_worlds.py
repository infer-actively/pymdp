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
    """ 2-dimensional grid-world implementation with 5 actions (the 4 cardinal directions and staying put)."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4

    CONTROL_NAMES = ["UP", "RIGHT", "DOWN", "LEFT", "STAY"]

    def __init__(self, shape=[2, 2], init_state=None):
        """
        Initialization function for 2-D grid world

        Parameters
        ----------
        shape: ``list`` of ``int``, where ``len(shape) == 2``
            The dimensions of the grid world, stored as a list of integers, storing the discrete dimensions of the Y (vertical) and X (horizontal) spatial dimensions, respectively.
        init_state: ``int`` or ``None``
            Initial state of the environment, i.e. the location of the agent in grid world. If not ``None``, must be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the initial location of the agent in grid world.
            If ``None``, then an initial location will be randomly sampled from the grid.
        """
        
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
        """
        Reset the state of the 2-D grid world. In other words, resets the location of the agent, and wipes the current action.

        Parameters
        ----------
        init_state: ``int`` or ``None``
            Initial state of the environment, i.e. the location of the agent in grid world. If not ``None``, must be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the initial location of the agent in grid world.
            If ``None``, then an initial location will be randomly sampled from the grid.

        Returns
        ----------
        self.state: ``int``
            The current state of the environment, i.e. the location of the agent in grid world. Will be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the location of the agent in grid world.
        """
        self.set_init_state(init_state)
        self.last_action = None
        return self.state

    def set_state(self, state):
        """
        Sets the state of the 2-D grid world.

        Parameters
        ----------
        state: ``int`` or ``None``
            State of the environment, i.e. the location of the agent in grid world. If not ``None``, must be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the location of the agent in grid world.
            If ``None``, then a location will be randomly sampled from the grid.

        Returns
        ----------
        self.state: ``int``
            The current state of the environment, i.e. the location of the agent in grid world. Will be a discrete index  in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the location of the agent in grid world.
        """
        self.state = state
        return state

    def step(self, action):
        """
        Updates the state of the environment, i.e. the location of the agent, using an action index that corresponds to one of the 5 possible moves.

        Parameters
        ----------
        action: ``int`` 
            Action index that refers to which of the 5 actions the agent will take. Actions are, in order: "UP", "RIGHT", "DOWN", "LEFT", "STAY".

        Returns
        ----------
        state: ``int``
            The new, updated state of the environment, i.e. the location of the agent in grid world after the action has been made. Will be discrete index in the range ``(0, (shape[0] * shape[1])-1)``. It is thus a "linear index" of the location of the agent in grid world.
        """
        state = self.P[self.state][action]
        self.state = state
        self.last_action = action
        return state

    def render(self, title=None):
        """
        Creates a heatmap showing the current position of the agent in the grid world.

        Parameters
        ----------
        title: ``str`` or ``None``
            Optional title for the heatmap.
        """
        values = np.zeros(self.shape)
        values[self.position] = 1.0
        _, ax = plt.subplots(figsize=(3, 3))
        if self.shape[0] == 1 or self.shape[1] == 1:
            ax.imshow(values, cmap="OrRd")
        else:
            _ = sns.heatmap(values, cmap="OrRd", linewidth=2.5, cbar=False, ax=ax)
        plt.xticks(range(self.shape[1]))
        plt.yticks(range(self.shape[0]))
        if title != None:
            plt.title(title)
        plt.show()

    def set_init_state(self, init_state=None):
        if init_state != None:
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
        if init_state == None:
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
    """ 1-dimensional grid-world implementation with 3 possible movement actions ("LEFT", "STAY", "RIGHT")"""

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
        if self.shape[0] == 1 or self.shape[1] == 1:
            ax.imshow(values, cmap="OrRd")
        else:
            _ = sns.heatmap(values, cmap="OrRd", linewidth=2.5, cbar=False, ax=ax)
        plt.xticks(range(self.shape[1]))
        plt.yticks(range(self.shape[0]))
        if title != None:
            plt.title(title)
        plt.show()

    def set_init_state(self, init_state=None):
        if init_state != None:
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
        if init_state == None:
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
