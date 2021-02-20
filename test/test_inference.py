import os
import unittest

import numpy as np
from scipy.io import loadmat

from pymdp import core
from pymdp.core.utils import (
    get_model_dimensions,
    random_A_matrix,
    random_B_matrix,
    obj_array,
    onehot,
    norm_dist,
)
from pymdp.core.maths import get_joint_likelihood_seq
from pymdp.core.inference import update_posterior_states_v2, average_states_over_policies
from pymdp.core.control import update_posterior_policies_v2
from pymdp.core import utils


def rand_onehot_obs(num_obs):
    if type(num_obs) is int:
        num_obs = [num_obs]
    obs = obj_array(len(num_obs))
    for i in range(len(num_obs)):
        ob = np.random.randint(num_obs[i])
        obs[i] = onehot(ob, num_obs[i])
    return obs


def rand_controls(num_controls):
    if type(num_controls) is int:
        num_controls = [num_controls]
    controls = np.zeros(len(num_controls))
    for i in range(len(num_controls)):
        controls[i] = np.random.randint(num_controls[i])
    return controls


def rand_dist_states(num_states):
    if type(num_states) is int:
        num_states = [num_states]
    states = obj_array(len(num_states))
    for i in range(len(num_states)):
        states[i] = norm_dist(np.random.rand(num_states[i]))
    return states


class Inference(unittest.TestCase):
    def test_update_posterior_states_v2(self):
        """
        Testing our SPM-ified version of `run_MMP` with
            1 hidden state factor & 1 outcome modality, at a random fixed
            timestep during the generative process
        """

        past_len = 3
        future_len = 4
        num_policies = 5
        num_states = [6, 7, 8]
        num_controls = [9, 10, 11]
        num_obs = [12, 13, 14]
        num_modalities = len(num_obs)

        A = random_A_matrix(num_obs, num_states)
        B = random_B_matrix(num_states, num_controls)
        prev_obs = [rand_onehot_obs(num_obs) for _ in range(past_len)]
        prev_actions = np.array([rand_controls(num_controls) for _ in range(past_len)])
        policies = [
            np.array([rand_controls(num_controls) for _ in range(future_len)])
            for _ in range(num_policies)
        ]
        prior = rand_dist_states(num_states)

        qs_seq_pi, VFE_policies = update_posterior_states_v2(A, B, prev_obs, policies, prev_actions, prior=prior, policy_sep_prior = False)

        qs_seq_pi_future = utils.obj_array(num_policies)
        for p_idx in range(num_policies):
            qs_seq_pi_future[p_idx] = qs_seq_pi[p_idx][(1 + past_len) :]
        
        # create  C matrix
        horizon = len(qs_seq_pi_future[0])
        C = utils.obj_array(horizon)
        for t in range(horizon):
            C[t] = utils.obj_array(num_modalities)
            for g in range(num_modalities):
                C[t][g] = np.ones(num_obs[g]) 
        
        q_pi, efe = update_posterior_policies_v2(
            qs_seq_pi_future,
            A,
            B,
            C,
            policies,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=False,
            prior = None,
            pA=None,
            pB=None,
            F = VFE_policies,
            E = None,
            gamma=16.0,
            return_numpy=True,
        )

        qs_pi_curr_t = utils.obj_array(num_policies)
        for p_idx in range(num_policies):
            qs_pi_curr_t[p_idx] = qs_seq_pi[p_idx][past_len]
        
        qs_bma = average_states_over_policies(qs_pi_curr_t, q_pi) # Bayesian model average of hidden states across policies

if __name__ == "__main__":
    unittest.main()

