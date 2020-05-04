#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Cube world environment

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

import numpy as np

from inferactively.envs import Env


class GridWorldEnv(object):
    """ Basic grid-world implementation """

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4

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

    def step(self, action):
        state = self.P[self.state][action]
        self.state = state
        self.last_action = action
        return state

    def render(self):
        """ @TODO render """
        print(f"Current agent state {self.state} (x {self.position[0]} y {self.position[1]})")

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
        """ @TODO might be wrong w.r.t x & y """
        return np.unravel_index(np.array(self.state), self.shape)


class NDGridWorldEnv(Env):
    """ N-dimensional Grid World Environment

    This environment can be understood as an agent operating in multiple grid worlds simultaneously
    When `n_dimensions == 1`, this environment operates as a standard grid world.

        > Let `N` be the number of dimensions. Each dimension defines a grid world of size (`w` x `h`)
        > Agents have access to `N` modalities, which give rise to size (1,) observations of position in that dimension
        > Agents maintain `N` belief factors, i.e. about their position in each dimension
        > Agent's can control all `N` factors. There are 5 control states for each factor:
            {UP : 0 RIGHT : 1 DOWN : 2 LEFT : 3 STAY 4}

    """

    def __init__(self, shape=[3, 3], n_dims=1, shapes=None, init_state=None):
        """Initialize grid world environment

        @NOTE If `shape` is passed and `n_dims` > 1, the same shape is used for all dims

        Parameters
        ----------
        - `shape` [list of ints] (optional)
            Specifies the width and height of the grid world (default [3, 3])
        - `n_dims` [np.ndarray] (optional)
            Number of simultaneous grid-world environments 
        - `shapes` [list of list of ints] (optional)
            Specifies the width and height of each dimension
        - `init_state` [int] (optional)
            Initial state - defaults to random
        """

        if shapes is not None:
            assert n_dims == len(shapes), "Provide a shape for each dimension"
        else:
            shapes = [shape]

        for shape in shapes:
            if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
                raise ValueError("`shape` must be a list/tuple of length 2")

        if len(shapes) == n_dims:
            self.grids = [GridWorldEnv(shapes[s], init_state) for s in range(n_dims)]
        else:
            self.grids = [GridWorldEnv(shape, init_state) for s in range(n_dims)]

        self.n_states = [grid.n_states for grid in self.grids]
        self.n_observations = [grid.n_states for grid in self.grids]
        self.n_control = [grid.n_control for grid in self.grids]
        self.n_dims = n_dims

    def reset(self, init_state=None):
        states = [grid.reset(init_state) for grid in self.grids]
        self.states = np.array(states)
        return self.to_observation(self.states)

    def step(self, controls):
        assert len(controls) == self.n_dims, "f`len(controls)` is not {self.n_dims}"
        states = []
        for control, grid in zip(controls, self.grids):
            state = grid.step(control)
            states.append(state)
        self.states = np.array(states)
        return self.to_observation(self.states)

    def to_observation(self, states):
        """ 
        @TODO Here we can put arbitrary mappings between states and observations
                For now, its identity
        """
        return states

    def render(self):
        pass

    def sample_action(self):
        controls = []
        for grid in self.grids:
            controls.append(np.random.randint(0, grid.n_control))
        return np.array(controls)

    def get_state(self):
        """ @TODO - somtimes state hasn't been set """
        states = [grid.state for grid in self.grids]
        return np.array(states)
