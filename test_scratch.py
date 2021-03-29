import os
import unittest

import numpy as np
from scipy.io import loadmat

from pymdp.agent import Agent
from pymdp.core.utils import random_A_matrix, random_B_matrix, obj_array_zeros, get_model_dimensions, convert_observation_array

import matplotlib.pyplot as plt

DATA_PATH = "test/matlab_crossval/output/"

"""
Test against output of SPM_MDP_VB_X Case A - one hidden state factor, one observation modality, policy_len = 1
"""

array_path = os.path.join(os.getcwd(), DATA_PATH + "vbx_test_a.mat")
mat_contents = loadmat(file_name=array_path)

A = mat_contents["A"][0]
B = mat_contents["B"][0]
prev_obs = mat_contents["obs"].astype("int64")
policy = mat_contents["policies"].astype("int64") - 1
t_horizon = mat_contents["t_horizon"][0, 0].astype("int64")
actions = mat_contents["actions"].astype("int64") - 1
qs = mat_contents["qs"][0]
likelihoods = mat_contents["likelihoods"][0]

num_obs, num_states, _, num_factors = get_model_dimensions(A, B)
prev_obs = convert_observation_array(
    prev_obs[:, max(0, curr_t - t_horizon) : (curr_t + 1)], num_obs
)

agent = Agent(A=A, B=B, C=C, inference_algo="MMP", policy_len=1, inference_horizon=t_horizon, use_BMA = False, policy_sep_prior = True)
