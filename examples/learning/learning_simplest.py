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
batch_size = 1
env = SimplestEnv(batch_size=batch_size)

# %% ### 1. Basic Demo
#
# This demo shows how to use the simplest environment with an active inference agent.
# The environment consists of two states (left and right) and two actions (stay and move).
# The agent can observe which state it is in perfectly.
#
# First, we'll create an instance of the simplest environment

# Set up random key
key = jr.PRNGKey(key_idx)

# Initialize agent's learning config
learning_config = LearningConfig(learn_A=False, learn_B=False, learn_D=False)

# Initialise POMDP model from environment and learning config
model, key = POMDPModel.from_env(
    env=env,
    learning=learning_config,
    key=key,
    T=100               #can play with this
)

# Update initial beliefs (D): Equal probability for all states
model = model.set_uniform_D()

# Set up preference (C) matrix
# C = [jnp.zeros((batch_size, 2), dtype=jnp.float32).at[:, 1].set(1.0)]  # The agent prefers to be in the right state (state 1)
C = [jnp.zeros((batch_size, model.structure.num_obs[0]), dtype=jnp.float32)]  # All states equally preferred

# Initialize the agent based on model and other parameters
agent = Agent.from_model(
    model=model,
    C=C,
    policy_len=1,            # Plan one step ahead
    inference_algo="fpi",
    apply_batch=False,
    action_selection="stochastic"
)

# Run simulation
key, rollout_key = jr.split(key)
final_state, info, _ = rollout(agent, env, num_timesteps=model.structure.T, rng_key=rollout_key)

# Print rollout and visualize results
plot_beliefs(info, agent)
render_rollout(env, info)  # Optionally: render_rollout(env, info, save_gif=True, filename="figures/simplest.gif")
print_rollout(info)