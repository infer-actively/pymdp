#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

import numpy as np

from inferactively.envs import GridWorldEnv
from inferactively.distributions import Categorical

""" A tutorial on Active Inference for Markov Decision Processes 

    This tutorial focuses on a single factor MDP problem. 
    Specifically, an agent exists in a grid world with 3 states, and can perform one of three actions (move left, move right, stay).

    __author__: Alexander Tschantz, Conor Heins, Beren Millidge
"""

""" (1) Creating a grid-world environment """

# shape of environment
env_shape = [2, 2]
# number of hidden states
n_states = np.prod(env_shape)
print(f"> {env_shape} grid world environment ({n_states})")

# init our environment
env = GridWorldEnv(shape=env_shape)
# reset environment a
state = env.reset()
print(f"> Initial state {state}")

# random actions to show the dynamics of the environment
# time horizon
T = 4 
for t in range(T):
    # sample random action
    action = env.sample_action() 
    # step environment 
    state = env.step(action)
    print(f"> Time {t}: state {state} action {action}")

""" (2) Initialise the generative model """

# create likelihood distribution (A matrix)
# n observations
n_observations = n_states

values = np.eye(n_states)
A = Categorical(values=values)
A.plot()



# Inference - agent infers its own state through observatinos
# for t in range(T):
    