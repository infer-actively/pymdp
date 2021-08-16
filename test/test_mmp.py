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

from pymdp.utils import get_model_dimensions, convert_observation_array
from pymdp.algos import run_mmp
from pymdp.maths import get_joint_likelihood_seq

DATA_PATH = "test/matlab_crossval/output/"

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

        prev_actions = prev_actions[(max(0, curr_t - t_horizon) -1) :, :]
        prior = np.empty(num_factors, dtype=object)
        for f in range(num_factors):
            uniform = np.ones(num_states[f]) / num_states[f]
            prior[f] = B[f][:, :, prev_actions[0, f]].dot(uniform)

        lh_seq = get_joint_likelihood_seq(A, prev_obs, num_states)
        qs_seq, _ = run_mmp(
            lh_seq, B, policy, prev_actions[1:], prior=prior, num_iter=5, grad_descent=True
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

        prev_actions = prev_actions[(max(0, curr_t - t_horizon)) :, :]
        lh_seq = get_joint_likelihood_seq(A, prev_obs, num_states)
        qs_seq, _ = run_mmp(lh_seq, 
             B, policy, prev_actions=prev_actions, prior=None, num_iter=5, grad_descent=True
        )

        result_pymdp = qs_seq[-1]
        for f in range(num_factors):
            self.assertTrue(np.isclose(result_spm[f].squeeze(), result_pymdp[f]).all())

    def test_mmp_c(self):
        """ Testing our SPM-ified version of `run_MMP` with
         2 hidden state factors & 2 outcome modalities, at the very first
         timestep of the generative process (boundary condition test). So there 
         are no previous actions"""

        array_path = os.path.join(os.getcwd(), DATA_PATH + "mmp_c.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"][0]
        B = mat_contents["B"][0]
        prev_obs = mat_contents["obs_idx"].astype("int64")
        policy = mat_contents["policy"].astype("int64") - 1
        curr_t = mat_contents["t"][0, 0].astype("int64") - 1
        t_horizon = mat_contents["t_horizon"][0, 0].astype("int64")
        # prev_actions = mat_contents["previous_actions"].astype("int64") - 1
        result_spm = mat_contents["qs"][0]
        likelihoods = mat_contents["likelihoods"][0]

        num_obs, num_states, _, num_factors = get_model_dimensions(A, B)
        prev_obs = convert_observation_array(
            prev_obs[:, max(0, curr_t - t_horizon) : (curr_t + 1)], num_obs
        )

        # prev_actions = prev_actions[(max(0, curr_t - t_horizon)) :, :]
        lh_seq = get_joint_likelihood_seq(A, prev_obs, num_states)
        qs_seq, _ = run_mmp(
            lh_seq, B, policy, prev_actions=None, prior=None, num_iter=5, grad_descent=True
        )

        result_pymdp = qs_seq[-1]
        for f in range(num_factors):
            self.assertTrue(np.isclose(result_spm[f].squeeze(), result_pymdp[f]).all())
    
    def test_mmp_d(self):
        """ Testing our SPM-ified version of `run_MMP` with
        2 hidden state factors & 2 outcome modalities, at the final
        timestep of the generative process (boundary condition test)
        @NOTE: mmp_d.mat test has issues with the prediction errors. But the future messages are 
        totally fine (even at the last timestep of variational iteration."""

        array_path = os.path.join(os.getcwd(), DATA_PATH + "mmp_d.mat")
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
        
        prev_actions = prev_actions[(max(0, curr_t - t_horizon) -1) :, :]
        prior = np.empty(num_factors, dtype=object)
        for f in range(num_factors):
            uniform = np.ones(num_states[f]) / num_states[f]
            prior[f] = B[f][:, :, prev_actions[0, f]].dot(uniform)

        lh_seq = get_joint_likelihood_seq(A, prev_obs, num_states)

        qs_seq, _ = run_mmp(
            lh_seq, B, policy, prev_actions[1:], prior=prior, num_iter=5, grad_descent=True, last_timestep=True
        )
    
        result_pymdp = qs_seq[-1] 

        for f in range(num_factors):
            self.assertTrue(np.isclose(result_spm[f].squeeze(), result_pymdp[f]).all())
    
    """"
    @ NOTE (from Conor Heins 07.04.2021)
    Please keep this uncommented code below here. We need to figure out how to re-include optional arguments e.g. `save_vfe_seq` 
    into `run_mmp` so that important tests like these can run again some day. My only dumb solution for now would be to just have a 'UnitTest variant' of the MMP function
    that has extra optional outputs that slow down run-time (e.g. `save_vfe_seq`), and are thus excluded from the deployable version of `pymdp`,
    but are useful for benchmarking the performance/ accuracy of the algorithm
    """
    # def test_mmp_fixedpoints(self):

    #     array_path = os.path.join(os.getcwd(), DATA_PATH + "mmp_a.mat")
    #     mat_contents = loadmat(file_name=array_path)

    #     A = mat_contents["A"][0]
    #     B = mat_contents["B"][0]
    #     prev_obs = mat_contents["obs_idx"].astype("int64")
    #     policy = mat_contents["policy"].astype("int64") - 1
    #     curr_t = mat_contents["t"][0, 0].astype("int64") - 1
    #     t_horizon = mat_contents["t_horizon"][0, 0].astype("int64")
    #     prev_actions = mat_contents["previous_actions"].astype("int64") - 1
    #     result_spm = mat_contents["qs"][0]
    #     likelihoods = mat_contents["likelihoods"][0]

    #     num_obs, num_states, _, num_factors = get_model_dimensions(A, B)
    #     prev_obs = convert_observation_array(
    #         prev_obs[:, max(0, curr_t - t_horizon) : (curr_t + 1)], num_obs
    #     )

    #     prev_actions = prev_actions[(max(0, curr_t - t_horizon) -1) :, :]
    #     prior = np.empty(num_factors, dtype=object)
    #     for f in range(num_factors):
    #         uniform = np.ones(num_states[f]) / num_states[f]
    #         prior[f] = B[f][:, :, prev_actions[0, f]].dot(uniform)

    #     lh_seq = get_joint_likelihood_seq(A, prev_obs, num_states)
    #     qs_seq, F = run_mmp(
    #         lh_seq, B, policy, prev_actions[1:], prior=prior, num_iter=5, grad_descent=False, save_vfe_seq=True
    #     )

    #     self.assertTrue((np.diff(np.array(F)) < 0).all())


if __name__ == "__main__":
    unittest.main()
