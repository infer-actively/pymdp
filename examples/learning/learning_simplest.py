#!/usr/bin/env python
# coding: utf-8

# # Simplest Environment Demo
# 
# In this script, we demonstrate a simple active inference agent in JAX solving the simplest possible environment using the `jax-pymdp` library.
# 
# The simplest environment has:
# - Two states (locations): left (0) and right (1)
# - Two observations: left (0) and right (1)
# - Two actions: go left (0) and go right (1)
# 
# The environment is fully observed (the observation likelihood matrix A is the identity matrix) and deterministic 
# (actions always lead to their corresponding states).

# ### Imports
#
# First, import `pymdp` and the modules we'll need.

# %% Importing necessary libraries
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random as jr
from pymdp.envs.simplest import SimplestEnv, print_rollout, plot_beliefs, plot_A_learning, print_parameter_learning
from pymdp.envs.rollout import rollout
from pymdp.agent import Agent
from pymdp.maths import dirichlet_expected_value

# if __name__ == "__main__":
key_idx = 1 # Initialize master random key index at the start

#%% Initialise environment
batch_size = 3
env = SimplestEnv(batch_size=batch_size)

# %% ### 1. Basic Demo
#
# This demo shows how to use the simplest environment with an active inference agent.
# The environment consists of two states (left and right) and two actions (go left and go right).
# The agent can observe which state it is in perfectly (identity A matrix).

# Set up random key
key = jr.PRNGKey(key_idx)

# Create the agent's generative model based on the environment parameters
# Get A tensors (observation model) from environment
A = [jnp.array(a, dtype=jnp.float32) for a in env.params["A"]]
A_dependencies = env.dependencies["A"]

# Get B tensors (transition model) from environment  
B = [jnp.array(b, dtype=jnp.float32) for b in env.params["B"]]
B_dependencies = env.dependencies["B"]

# Get D tensors (initial state beliefs) from environment
D = [jnp.array(d, dtype=jnp.float32) for d in env.params["D"]]

# Create C tensors (preferences) - all zeros means no preference
C = [jnp.zeros((batch_size, a.shape[1]), dtype=jnp.float32) for a in A]

# Create the agent
agent = Agent(
    A, B, C, D,
    policy_len=1,  # Plan one step ahead
    A_dependencies=A_dependencies,
    B_dependencies=B_dependencies, 
    batch_size=batch_size,
    learn_A=False,
    learn_B=False
)

# %% Print setup
print("=== Basic Demo Setup ===")
print(f"Environment A matrix (observation model):\n{env.params['A'][0][0]}")
print(f"Environment B matrix (transition model):\n{env.params['B'][0][0]}")
print(f"Agent starting beliefs: {D[0][0]}")

# %% Run basic simulation
key = jr.PRNGKey(key_idx)
T = 5  # Number of timesteps
last, info, _ = rollout(agent, env, num_timesteps=T, rng_key=key)

# %%  Print rollout
print_rollout(info, batch_idx=0)

