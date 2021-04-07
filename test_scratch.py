# %%
import os
import unittest

import numpy as np
from scipy.io import loadmat

from pymdp.agent import Agent
from pymdp.core.utils import random_A_matrix, random_B_matrix, obj_array_zeros, get_model_dimensions, convert_observation_array
from pymdp.core.utils import to_arr_of_arr, to_numpy
from pymdp.core import control

import matplotlib.pyplot as plt

# %%

"""
Test against output of SPM_MDP_VB_X Case A - one hidden state factor, one observation modality, policy_len = 1
"""

DATA_PATH = "test/matlab_crossval/output/"

array_path = os.path.join(os.getcwd(), DATA_PATH + "vbx_test_a.mat")
mat_contents = loadmat(file_name=array_path)

A = mat_contents["A"][0]
B = mat_contents["B"][0]
C = to_arr_of_arr(mat_contents["C"][0][0][:,0])
obs_matlab = mat_contents["obs"].astype("int64")
policy = mat_contents["policies"].astype("int64") - 1
t_horizon = mat_contents["t_horizon"][0, 0].astype("int64")
actions = mat_contents["actions"].astype("int64") - 1
qs = mat_contents["qs"][0]
likelihoods = mat_contents["likelihoods"][0]

num_obs, num_states, _, num_factors = get_model_dimensions(A, B)
obs = convert_observation_array(obs_matlab, num_obs)

agent = Agent(A=A, B=B, C=C, inference_algo="MMP", policy_len=1, 
                inference_horizon=t_horizon, use_BMA = False, 
                policy_sep_prior = True)

T = len(obs)

# %% Run loop over time
for t in range(T):

    o_t = (np.where(obs[t])[0][0],)
    qx = agent.infer_states(o_t)
    agent.infer_policies()

    # action = agent.sample_action() # we're skipping this because it has randomness in it

    # this is what happens in that line `agent.sample()` above
    # action = control.sample_action(
    #         agent.q_pi, agent.policies, agent.n_controls, agent.action_sampling
    #     )
    # agent.action = action
    # agent.step_time()

    # we're gonna set the action to the one given by SPM

    agent.action = actions[:,t].T
    agent.step_time()


# %%
