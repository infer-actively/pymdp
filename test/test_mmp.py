#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Test of `run_mmp` function under various parameterisations

__date__: 25/11/2020
__author__: Conor Heins, Alexander Tschantz
"""

import os
import unittest

import numpy as np
from scipy.io import loadmat

from pymdp import core
from pymdp.core.utils import get_model_dimensions
from pymdp.core.algos import run_mmp_v2
from pymdp.core.maths import get_joint_likelihood_seq

DATA_PATH = "test/matlab_crossval/output/"


def onehot(value, num_values):
    """
    Quick way to convert an integer index to its corresponding one-hot vector 
    representation using `np.eye()`
    """
    arr = np.eye(num_values)[value]
    return arr


def convert_observation_array(obs, num_obs):
    """
    Converts from SPM-style observation array to infer-actively one-hot object arrays.
    
    Parameters
    ----------
    - 'obs' [numpy 2-D nd.array]:
        SPM-style observation arrays are of shape (num_modalities, T), where each row 
        contains observation indices for a different modality, and columns indicate 
        different timepoints. Entries store the indices of the discrete observations 
        within each modality. 

    - 'num_obs' [list]:
        List of the dimensionalities of the observation modalities. `num_modalities` 
        is calculated as `len(num_obs)` in the function to determine whether we're 
        dealing with a single- or multi-modality 
        case.

    Returns
    ----------
    - `obs_t`[list]: 
        A list with length equal to T, where each entry of the list is either a) an object 
        array (in the case of multiple modalities) where each sub-array is a one-hot vector 
        with the observation for the correspond modality, or b) a 1D numpy array (in the case
        of one modality) that is a single one-hot vector encoding the observation for the 
        single modality.
    """

    T = obs.shape[1]
    num_modalities = len(num_obs)

    # Initialise the output
    obs_t = []
    # Case of one modality
    if num_modalities == 1:
        for t in range(T):
            obs_t.append(onehot(obs[0, t] - 1, num_obs[0]))
    else:
        for t in range(T):
            obs_AoA = np.empty(num_modalities, dtype=object)
            for g in range(num_modalities):
                # Subtract obs[g,t] by 1 to account for MATLAB vs. Python indexing
                # (MATLAB is 1-indexed)
                obs_AoA[g] = onehot(obs[g, t] - 1, num_obs[g])
            obs_t.append(obs_AoA)

    return obs_t


class MMP(unittest.TestCase):
    def test_mmp_a(self):
        """
        Testing our SPM-ified version of `run_MMP` with
            1 hidden state factor & 1 outcome modality, at a random fixed
            timestep during the generative process
        """

        array_path = os.path.join(os.getcwd(), DATA_PATH + "mmp_a.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"][0]
        B = mat_contents["B"][0]
        prev_obs = mat_contents["obs_idx"].astype("int64")
        policy = mat_contents["policy"].astype("int64") - 1
        curr_t = mat_contents["t"][0, 0].astype("int64") - 1
        t_horizon = mat_contents["t_horizon"][0, 0].astype("int64")
        prev_actions = mat_contents["previous_actions"].astype("int64") - 1
        result_spm = mat_contents["qs"][0]
        likelihoods = mat_contents["likelihoods"][0]

        num_obs, num_states, _, num_factors = get_model_dimensions(A, B)
        prev_obs = convert_observation_array(
            prev_obs[:, max(0, curr_t - t_horizon) : (curr_t + 1)], num_obs
        )

        prev_actions = prev_actions[(max(0, curr_t - t_horizon) - 1) :, :]
        prior = np.empty(num_factors, dtype=object)
        for f in range(num_factors):
            uniform = np.ones(num_states[f]) / num_states[f]
            prior[f] = B[f][:, :, prev_actions[0, f]].dot(uniform)

        ll_seq = get_joint_likelihood_seq(A, prev_obs, num_states)
        qs_seq = run_mmp_v2(
            A, B, ll_seq, policy, prev_actions[1:], prior=prior, num_iter=5, grad_descent=True
        )

        result_pymdp = qs_seq[-1]
        for f in range(num_factors):
            self.assertTrue(np.isclose(result_spm[f].squeeze(), result_pymdp[f]).all())

    def test_mmp_b(self):
        """ Testing our SPM-ified version of `run_MMP` with
        2 hidden state factors & 2 outcome modalities, at a random fixed
        timestep during the generative process"""

        array_path = os.path.join(os.getcwd(), DATA_PATH + "mmp_b.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"][0]
        B = mat_contents["B"][0]
        obs = mat_contents["obs_idx"].astype("int64")
        policy = mat_contents["policy"].astype("int64") - 1
        curr_t = mat_contents["t"][0, 0].astype("int64") - 1
        t_horizon = mat_contents["t_horizon"][0, 0].astype("int64")
        T = obs.shape[1]
        previous_actions = mat_contents["previous_actions"].astype("int64") - 1
        result_spm = mat_contents["qs"][0]
        _ = mat_contents["likelihoods"][0]

        # Convert matlab index-style observations into list of array of arrays over time
        num_obs = [A[g].shape[0] for g in range(len(A))]
        obs_t = convert_observation_array(obs, num_obs)

        qs, _, _, _ = core.algos.run_mmp(
            A,
            B,
            obs_t,
            policy,
            curr_t,
            t_horizon,
            T,
            use_gradient_descent=True,
            num_iter=5,
            previous_actions=previous_actions,
        )

        # Just check whether the latest beliefs (about curr_t, held at curr_t) match up
        result_pymdp = qs[-1]
        for f in range(len(B)):
            self.assertTrue(np.isclose(result_spm[f].squeeze(), result_pymdp[f]).all())
    
    def test_mmp_c(self):
        """ Testing our SPM-ified version of `run_MMP` with
        2 hidden state factors & 2 outcome modalities, at the first
        timestep of the generative process (boundary condition test)"""

        array_path = os.path.join(os.getcwd(), DATA_PATH + "mmp_c.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"][0]
        B = mat_contents["B"][0]
        obs = mat_contents["obs_idx"].astype("int64")
        policy = mat_contents["policy"].astype("int64") - 1
        curr_t = mat_contents["t"][0, 0].astype("int64") - 1
        t_horizon = mat_contents["t_horizon"][0, 0].astype("int64")
        T = obs.shape[1]
        previous_actions = mat_contents["previous_actions"].astype("int64") - 1
        result_spm = mat_contents["qs"][0]
        _ = mat_contents["likelihoods"][0]

        # Convert matlab index-style observations into list of array of arrays over time
        num_obs = [A[g].shape[0] for g in range(len(A))]
        obs_t = convert_observation_array(obs, num_obs)

        qs, _, _, _ = core.algos.run_mmp(
            A,
            B,
            obs_t,
            policy,
            curr_t,
            t_horizon,
            T,
            use_gradient_descent=True,
            num_iter=5,
            previous_actions=previous_actions,
        )

        # Just check whether the latest beliefs (about curr_t, held at curr_t) match up
        result_pymdp = qs[-1]
        for f in range(len(B)):
            self.assertTrue(np.isclose(result_spm[f].squeeze(), result_pymdp[f]).all())
    
    def test_mmp_d(self):
        """ Testing our SPM-ified version of `run_MMP` with
        2 hidden state factors & 2 outcome modalities, at the final
        timestep of the generative process (boundary condition test)"""

        array_path = os.path.join(os.getcwd(), DATA_PATH + "mmp_d.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"][0]
        B = mat_contents["B"][0]
        obs = mat_contents["obs_idx"].astype("int64")
        policy = mat_contents["policy"].astype("int64") - 1
        curr_t = mat_contents["t"][0, 0].astype("int64") - 1
        t_horizon = mat_contents["t_horizon"][0, 0].astype("int64")
        T = obs.shape[1]
        previous_actions = mat_contents["previous_actions"].astype("int64") - 1
        result_spm = mat_contents["qs"][0]
        _ = mat_contents["likelihoods"][0]

        # Convert matlab index-style observations into list of array of arrays over time
        num_obs = [A[g].shape[0] for g in range(len(A))]
        obs_t = convert_observation_array(obs, num_obs)

        # print('policy:')
        # print(policy)
        # print('-----------------')

        # print('Previous actions:')
        # print(previous_actions)
        # print('-----------------')

        # print('Obs_t:')
        # print(obs_t)
        # print('-----------------')

        qs, _, _, _ = core.algos.run_mmp(
            A,
            B,
            obs_t,
            policy,
            curr_t,
            t_horizon,
            T,
            use_gradient_descent=True,
            num_iter=5,
            previous_actions=previous_actions,
        )

        # print("number of timesteps stored in posterior:")
        # print(len(qs))
        # print('-----------------')

        # Just check whether the latest beliefs (about curr_t, held at curr_t) match up
        result_pymdp = qs[-1]

        # print("factor 1 first timestep:")
        # print(qs[0][0])
        # print('-----------------')

        # print("factor 2 first timestep:")
        # print(qs[0][1])
        # print('-----------------')

        # print("factor 1 final timestep:")
        # print(result_pymdp[0])
        # print('-----------------')

        # print("factor 2 final timestep:")
        # print(result_pymdp[1])
        # print('-----------------')
        for f in range(len(B)):
            self.assertTrue(np.isclose(result_spm[f].squeeze(), result_pymdp[f]).all())


    def test_mmp_v2(self):
        array_path = os.path.join(os.getcwd(), DATA_PATH + "mmp_b.mat")
        mat_contents = loadmat(file_name=array_path)
        A = mat_contents["A"][0]
        B = mat_contents["B"][0]
        prev_obs = mat_contents["obs_idx"].astype("int64")
        policy = mat_contents["policy"].astype("int64") - 1
        curr_t = mat_contents["t"][0, 0].astype("int64") - 1
        t_horizon = mat_contents["t_horizon"][0, 0].astype("int64")
        prev_actions = mat_contents["previous_actions"].astype("int64") - 1
        result_spm = mat_contents["qs"][0]
        _ = mat_contents["likelihoods"][0]

        num_obs, num_states, _, num_factors = get_model_dimensions(A, B)
        prev_obs = convert_observation_array(
            prev_obs[:, max(0, curr_t - t_horizon) : (curr_t + 1)], num_obs
        )

        ll_seq = get_joint_likelihood_seq(A, prev_obs, num_states)
        qs_seq = run_mmp_v2(A, B, ll_seq, policy, prev_actions, num_iter=5, grad_descent=True)

        result_pymdp = qs_seq[-1]
        for f in range(num_factors):
            self.assertTrue(np.isclose(result_spm[f].squeeze(), result_pymdp[f]).all())


if __name__ == "__main__":
    unittest.main()
