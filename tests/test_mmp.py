#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Test of `run_mmp` function under various parameterisations
__date__: 25/11/2020
__author__: Conor Heins
"""

import os
import sys
import unittest

import numpy as np
from scipy.io import loadmat

sys.path.append(".")
from inferactively import core


def onehot(value, num_values):
    """
    Quick way to convert an integer index to its corresponding one-hot vector representation using `np.eye()`
    """
    arr = np.eye(num_values)[value]
    return arr

def convert_observation_array(obs, num_obs):
    """
    Converts from SPM-style observation array to infer-actively one-hot object arrays.
    Parameters
    ----------
    - 'obs' [numpy 2-D nd.array]:
        SPM-style observation arrays are of shape (num_modalities, T), where
        each row contains observation indices for a different modality, and columns 
        indicate different timepoints. Entries store the indices of the discrete observations within
        each modality. 
    - 'num_obs' [list]:
        List of the dimensionalities of the observation modalities. `num_modalities` is calculated as 
        `len(num_obs)` in the function to determine whether we're dealing with a single- or multi-modality 
        case.
    Returns
    ----------
    - `obs_t`[list]: 
        A list with length equal to T, where each entry of the list is either a) an object array 
        (in the case of multiple modalities) where each sub-array is a one-hot vector with the observation 
        for the correspond modality, or b) a 1D numpy array (in the case of one modality) 
        that is a single one-hot vector encoding the observation for the single modality.
    """

    T = obs.shape[1]
    num_modalities = len(num_obs)
    
    obs_t = [] # initialise the output
    if num_modalities == 1: # case of one modality
        for t in range(T):
            obs_t.append(onehot(obs[0,t],num_obs[0]))
    else:
        for t in range(T):
            obs_AoA = np.empty(num_modalities,dtype=object)
            for g in range(num_modalities):
                obs_AoA[g] = onehot(obs[g,t]-1,num_obs[g]) # subtract obs[g,t] by 1 to account for MATLAB vs. Python indexing (MATLAB is 1-indexed)
            obs_t.append(obs_AoA)
    
    return obs_t
                
class mmp(unittest.TestCase):
    
    def test_mmp_b(self):
        """ 2 hidden state factors, 2 outcome modalities"""

        array_path = os.path.join(os.getcwd(), "tests/data/mmp_b.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"][0]
        B = mat_contents["B"][0]
        obs = mat_contents["obs_idx"].astype('int64')
        policy = mat_contents["policy"].astype('int64') - 1
        curr_t = mat_contents["t"][0,0].astype('int64') - 1
        t_horizon = mat_contents["t_horizon"][0,0].astype('int64')
        T = obs.shape[1]
        previous_actions = mat_contents["previous_actions"].astype('int64') - 1
        result_spm = mat_contents["qs"][0]
        likelihoods = mat_contents["likelihoods"][0]
        for t in range(len(likelihoods)):
            print(f"likelihood matlab {likelihoods[t]}")

        # convert matlab index-style observations into list of array of arrays over time
        num_obs = [A[g].shape[0] for g in range(len(A))]
        obs_t = convert_observation_array(obs, num_obs)

        # print(previous_actions.shape)
        # print(policy.shape)
        # print(curr_t.shape)
        qs, _, _, _ = core.algos.run_mmp(A, B, obs_t, policy, curr_t, t_horizon,T, use_gradient_descent = True, num_iter = 5, previous_actions = previous_actions)

        print(f"final qs {qs[-2]}")
        print(f"matlab qs {result_spm}")

        result_inferactively = qs[-2] # just check whether the latest beliefs (about curr_t, held at curr_t) match up
        for f in range(len(B)):
            self.assertTrue(np.isclose(result_spm[f], result_inferactively[f]).all())


if __name__ == "__main__":
    unittest.main()
