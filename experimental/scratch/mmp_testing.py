# %%
import sys
import pathlib
import numpy as np
import copy
from scipy.io import loadmat
import os

from pymdp import algos
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

# %% define some auxiliary functions that help generate likelihoods and other variables useful for testing

def construct_generic_A(num_obs, n_states):
    """
    Generates a random likelihood array
    """ 

    num_modalities = len(num_obs)

    if num_modalities == 1: # single modality case
        A = np.random.rand(*(num_obs + n_states))
        A = np.divide(A,A.sum(axis=0))
    elif num_modalities > 1: # multifactor case
        A = np.empty(num_modalities, dtype = object)
        for modality,no in enumerate(num_obs):
            tmp = np.random.rand((*([no] + n_states)))
            tmp = np.divide(tmp,tmp.sum(axis=0))
            A[modality] = tmp
    return A


def construct_generic_B(n_states, n_control):
    """
    Generates a fully controllable transition likelihood array, where each action (control state) corresponds to a move to the n-th state from any other state, for each control factor
    """ 

    num_factors = len(n_states)

    if num_factors == 1: # single factor case
        B = np.eye(n_states[0])[:, :, np.newaxis]
        B = np.tile(B, (1, 1, n_control[0]))
        B = B.transpose(1, 2, 0)
    elif num_factors > 1: # multifactor case
        B = np.empty(num_factors, dtype = object)
        for factor,nc in enumerate(n_control):
            tmp = np.eye(nc)[:, :, np.newaxis]
            tmp = np.tile(tmp, (1, 1, nc))
            B[factor] = tmp.transpose(1, 2, 0)

    return B


def construct_init_qs(n_states):
    """
    Creates a random initial posterior
    """

    num_factors = len(n_states)
    if num_factors == 1: 
        qs = np.random.rand(n_states[0])
        qs = qs / qs.sum()
    elif num_factors > 1:
        qs = np.empty(num_factors, dtype = object)
        for factor, ns in enumerate(n_states):
            tmp = np.random.rand(ns)
            qs[factor] = tmp / tmp.sum()

    return qs

# %%

num_obs = [2, 2]
num_states = [2, 2, 3]
n_controls = [2, 2, 3]

A = construct_generic_A(num_obs, num_states)
B = construct_generic_B(num_states, n_controls)
# qs = construct_init_qs(num_states)

# print(A[0].shape)
# print(A[1].shape)

# for f in range(len(num_states)):
    #print(qs[f].shape)
    # print(B[f].shape)

observation = np.empty(len(num_obs),dtype=object)
for g in range(len(num_obs)):
    observation[g] = np.eye(num_obs[g])[np.random.choice(num_obs[g]),:]
    print(observation[g].shape)

obs_t = [observation]

policy = np.array([[1], [1], [2]]).reshape(1,len(n_controls))
curr_t = 0
t_horizon = 1
T = 1

qs1, qss, F, F_pol = algos.run_mmp(A, B, obs_t, policy, curr_t, t_horizon,T)
print(f"qs1 len {len(qs1)} qs1 {qs1[0]} qss len {len(qss)} qss [0] {qss[0]} F {F} F_pol {F_pol}")

qs2 = algos.run_fpi(A, obs_t[0], num_obs, num_states, prior=None, num_iter=10, dF=1.0, dF_tol=0.001)
print(f"qs2 len {len(qs2)} qs2 {qs2}")


# %% matlab comparison

