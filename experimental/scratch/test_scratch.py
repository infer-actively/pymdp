# %%
import os
import unittest

import numpy as np
from scipy.io import loadmat

from pymdp.agent import Agent, build_belief_array, build_xn_vn_array
from pymdp.core.utils import random_A_matrix, random_B_matrix, obj_array_zeros, get_model_dimensions, convert_observation_array
from pymdp.core.utils import to_arr_of_arr
from pymdp.core import control

import matplotlib.pyplot as plt

# %% Test against output of SPM_MDP_VB_X Case A - one hidden state factor, one observation modality, policy_len = 1

DATA_PATH = "test/matlab_crossval/output/"

array_path = os.path.join(os.getcwd(), DATA_PATH + "vbx_test_1a.mat")
mat_contents = loadmat(file_name=array_path)

A = mat_contents["A"][0]
B = mat_contents["B"][0]
C = to_arr_of_arr(mat_contents["C"][0][0][:,0])
obs_matlab = mat_contents["obs"].astype("int64")
policy = mat_contents["policies"].astype("int64") - 1
t_horizon = mat_contents["t_horizon"][0, 0].astype("int64")
actions = mat_contents["actions"].astype("int64") - 1
qs = mat_contents["qs"][0]
xn = mat_contents["xn"][0]
vn = mat_contents["vn"][0]

likelihoods = mat_contents["likelihoods"][0]

num_obs, num_states, _, num_factors = get_model_dimensions(A, B)
obs = convert_observation_array(obs_matlab, num_obs)

agent = Agent(A=A, B=B, C=C, inference_algo="MMP", policy_len=1, 
                inference_horizon=t_horizon, use_BMA = False, 
                policy_sep_prior = True)

T = len(obs)

# %% Run loop

# all_actions = np.zeros(T)

# for t in range(T):
#     o_t = (np.where(obs[t])[0][0],)
#     qx, xn_t, vn_t = agent.infer_states_test(o_t)
#     q_pi = agent.infer_policies()
#     action = agent.sample_action()
#     all_actions[t] = action

# %% Run first timestep timesteps

all_actions = np.zeros(T)

for t in range(T):
    o_t = (np.where(obs[t])[0][0],)
    qx, xn_t, vn_t = agent.infer_states_test(o_t)
    q_pi, efe= agent.infer_policies()
    print(q_pi)
    action = agent.sample_action()

    all_actions[t] = action

    # qx_reshaped = build_belief_array(qx)
    xn_reshaped = build_xn_vn_array(xn_t)
    vn_reshaped = build_xn_vn_array(vn_t)

    if t == T-1:
        xn_reshaped = xn_reshaped[:,:,:-1,:]
        vn_reshaped = vn_reshaped[:,:,:-1,:]

    start_tstep = max(0, agent.curr_timestep - agent.inference_horizon)
    end_tstep = min(agent.curr_timestep + agent.policy_len, T)

    xn_validation = xn[0][:,:,start_tstep:end_tstep,t,:]
    vn_validation = vn[0][:,:,start_tstep:end_tstep,t,:]

    assert np.isclose(xn_reshaped, xn_validation).all(), "Arrays not the same!"
    assert np.isclose(vn_reshaped, vn_validation).all(), "Arrays not the same!"

# %%
# t = 0

# o_t = (np.where(obs[t])[0][0],)
# print(f'Observations used for timestep {t}\n')
# qx, xn_t, vn_t = agent.infer_states_test(o_t)
# q_pi = agent.infer_policies()

# action = agent.sample_action()

# qx_reshaped = build_belief_array(qx)
# xn_reshaped = build_xn_vn_array(xn_t)
# vn_reshaped = build_xn_vn_array(vn_t)

# start_tstep = max(0, agent.curr_timestep - agent.inference_horizon)
# end_tstep = min(agent.curr_timestep + agent.policy_len, T)

# xn_validation = xn[0][:,:,start_tstep:end_tstep,t,:]
# vn_validation = vn[0][:,:,start_tstep:end_tstep,t,:]

# assert np.isclose(xn_reshaped, xn_validation).all(), "Arrays not the same!"
# assert np.isclose(vn_reshaped, vn_validation).all(), "Arrays not the same!"

# %% Run timesteps

# t = 1

# o_t = (np.where(obs[t])[0][0],)
# print(f'Observations used for timestep {t}\n')
# qx, xn_t, vn_t = agent.infer_states_test(o_t)
# q_pi = agent.infer_policies()

# action = agent.sample_action()

# qx_reshaped = build_belief_array(qx)
# xn_reshaped = build_xn_vn_array(xn_t)
# vn_reshaped = build_xn_vn_array(vn_t)

# start_tstep = max(0, agent.curr_timestep - agent.inference_horizon)
# end_tstep = min(agent.curr_timestep + agent.policy_len, T)

# xn_validation = xn[0][:,:,start_tstep:end_tstep,t,:]
# vn_validation = vn[0][:,:,start_tstep:end_tstep,t,:]

# assert np.isclose(xn_reshaped, xn_validation).all(), "Arrays not the same!"
# assert np.isclose(vn_reshaped, vn_validation).all(), "Arrays not the same!"

# %% Run timesteps

# t = 2

# o_t = (np.where(obs[t])[0][0],)
# print(f'Observations used for timestep {t}\n')
# qx, xn_t, vn_t = agent.infer_states_test(o_t)
# q_pi = agent.infer_policies()

# action = agent.sample_action()

# qx_reshaped = build_belief_array(qx)
# xn_reshaped = build_xn_vn_array(xn_t)
# vn_reshaped = build_xn_vn_array(vn_t)

# start_tstep = max(0, agent.curr_timestep - agent.inference_horizon)
# end_tstep = min(agent.curr_timestep + agent.policy_len, T)

# xn_validation = xn[0][:,:,start_tstep:end_tstep,t,:]
# vn_validation = vn[0][:,:,start_tstep:end_tstep,t,:]

# assert np.isclose(xn_reshaped, xn_validation).all(), "Arrays not the same!"
# assert np.isclose(vn_reshaped, vn_validation).all(), "Arrays not the same!"

# # %%

# t = 3

# o_t = (np.where(obs[t])[0][0],)
# print(f'Observations used for timestep {t}\n')
# qx, xn_t, vn_t = agent.infer_states_test(o_t)
# q_pi = agent.infer_policies()

# action = agent.sample_action()

# qx_reshaped = build_belief_array(qx)
# xn_reshaped = build_xn_vn_array(xn_t)
# vn_reshaped = build_xn_vn_array(vn_t)

# start_tstep = max(0, agent.curr_timestep - agent.inference_horizon)
# end_tstep = min(agent.curr_timestep + agent.policy_len, T)

# xn_validation = xn[0][:,:,start_tstep:end_tstep,t,:]
# vn_validation = vn[0][:,:,start_tstep:end_tstep,t,:]

# assert np.isclose(xn_reshaped, xn_validation).all(), "Arrays not the same!"
# assert np.isclose(vn_reshaped, vn_validation).all(), "Arrays not the same!"

# # %%

# t = 4

# o_t = (np.where(obs[t])[0][0],)
# qx, xn_t, vn_t = agent.infer_states_test(o_t)
# q_pi = agent.infer_policies()

# action = agent.sample_action()

# qx_reshaped = build_belief_array(qx)
# xn_reshaped = build_xn_vn_array(xn_t)
# vn_reshaped = build_xn_vn_array(vn_t)

# start_tstep = max(0, agent.curr_timestep - agent.inference_horizon)
# end_tstep = min(agent.curr_timestep + agent.policy_len, T)

# xn_validation = xn[0][:,:,start_tstep:end_tstep,t,:]
# vn_validation = vn[0][:,:,start_tstep:end_tstep,t,:]

# assert np.isclose(xn_reshaped, xn_validation).all(), "Arrays not the same!"
# assert np.isclose(vn_reshaped, vn_validation).all(), "Arrays not the same!"

