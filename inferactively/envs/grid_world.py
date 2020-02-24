#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Grid world environment

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

import numpy as np
import sys

class GridWorldEnv(object):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4

    def __init__(self, shape=[3, 3]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError("`shape` argument must be a list/tuple of length 2")

        self.shape = shape
        self.n_states = np.prod(shape)
        self.n_observations = self.n_states
        self.n_actions = 5

        self.max_y = shape[0]
        self.max_x = shape[1]

        self.reward_id = 0
        self.initial_state = np.random.randint(0, self.n_states)

        self.build()

    def set_reward_state(self, reward_id):
        if reward_id > (self.n_states - 1) or reward_id < 0:
            raise ValueError("`reward_id` is greater than number of states")
        if not isinstance(reward_id, (int, float)):
            raise ValueError("`reward_id` must be [int/float]")

        self.reward_id = int(reward_id)

    def set_initial_state(self, initial_state):
        if initial_state > (self.n_states - 1) or initial_state < 0:
            raise ValueError("`initial_state` is greater than number of states")
        if not isinstance(initial_state, (int, float)):
            raise ValueError("`initial_state` must be [int/float]")

        self.initial_state = int(initial_state)

    def get_reward(self, s):
        if s == self.reward_id:
            reward = 1.0
        else:
            reward = -1.0
        return reward

    def build(self):
        P = {}
        grid = np.arange(self.n_states).reshape(self.shape)
        it = np.nditer(grid, flags=["multi_index"])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            P[s] = {a: [] for a in range(self.n_actions)}

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
        self.initial_state_dist = np.zeros(self.n_states)
        self.initial_state_dist[self.initial_state] = 1.0
        self.last_action = None
        self.s = self.initial_state

    def reset(self):
        self.s = self.initial_state
        self.last_action = None
        return self.s

    def step(self, a):
        s = self.P[self.s][a]
        self.s = s
        self.last_action = a
        r = self.get_reward(s)
        return (s, r)

    def render(self):
        outfile = sys.stdout
        grid = np.arange(self.n_states).reshape(self.shape)
        it = np.nditer(grid, flags=["multi_index"])

        while not it.finished:
            s = it.iterindex
            _, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == self.reward_id:
                output = " R "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)
            if x == self.shape[1] - 1:
                outfile.write("\n")
            it.iternext()
        outfile.write("\n")

    def categorical_sample(self, prob_n):
        prob_n = np.asarray(prob_n)
        csprob_n = np.cumsum(prob_n)
        return (csprob_n > np.random.rand()).argmax()

    def get_transition_matrix(self):
        B = np.zeros([self.n_actions, self.n_states, self.n_states])
        for s in range(self.n_states):
            for a in range(self.n_actions):
                ns = int(self.P[s][a])
                B[a, ns, s] = 1
        return B

    # def get_transition_matrix(self):
    #     # storing actions in the third dimension - makes things easier with the way other functions work
    #     B = np.zeros([self.n_states, self.n_states, self.n_actions])
    #     for s in range(self.n_states):
    #         for a in range(self.n_actions):
    #             ns = int(self.P[s][a])
    #             B[ns, s, a] = 1
    #     return B

    def get_likelihood_matrix(self):
        A = np.eye(self.n_observations, self.n_states)
        return A
