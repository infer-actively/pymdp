#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import os
import sys
import unittest

import numpy as np

sys.path.append(".")
from inferactively.distributions import Categorical, Dirichlet  # nopep8
from inferactively import core

# define some auxiliary functions that help generate likelihoods and other variables useful for testing

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

def construct_pA(num_obs, n_states, prior_scale = 1.0):
    """
    Generates Dirichlet prior over a observation likelihood distribution (initialized to all ones * prior_scale parameter)
    """ 

    num_modalities = len(num_obs)

    if num_modalities == 1: # single modality case
        pA = prior_scale * np.ones((num_obs + n_states))
    elif num_modalities > 1: # multifactor case
        pA = np.empty(num_modalities, dtype = object)
        for modality,no in enumerate(num_obs):
            pA[modality] = prior_scale * np.ones((no, *n_states))

    return pA

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

def construct_pB(n_states, n_control, prior_scale = 1.0):
    """
    Generates Dirichlet prior over a transition likelihood distribution (initialized to all ones * prior_scale parameter)
    """ 

    num_factors = len(n_states)

    if num_factors == 1: # single factor case
        pB = prior_scale * np.ones( (n_states[0], n_states[0]) )[:, :, np.newaxis]
        pB = np.tile(pB, (1, 1, n_control[0]))
        pB = pB.transpose(1, 2, 0)
    elif num_factors > 1: # multifactor case
        pB = np.empty(num_factors, dtype = object)
        for factor,nc in enumerate(n_control):
            tmp = prior_scale * np.ones( (nc, nc) )[:, :, np.newaxis]
            tmp = np.tile(tmp, (1, 1, nc))
            pB[factor] = tmp.transpose(1, 2, 0)

    return pB

def construct_generic_C(num_obs):
    """
    Generates a random C matrix
    """ 

    num_modalities = len(num_obs)

    if num_modalities == 1: # single modality case
        C = np.random.rand(num_obs[0])
        C = np.divide(C,C.sum(axis=0))
    elif num_modalities > 1: # multifactor case
        C = np.empty(num_modalities, dtype = object)
        for modality,no in enumerate(num_obs):
            tmp = np.random.rand(no)
            tmp = np.divide(tmp,tmp.sum())
            C[modality] = tmp

    return C

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

class TestControl(unittest.TestCase):

    def test_onestep_singlefac_posteriorPolicies(self):
        """
        Test for computing posterior over policies (and associated expected free energies)
        in the case of a posterior over hidden states with a single hidden state factor. 
        This version tests using a policy horizon of 1 step ahead
        """

        n_states = [3]
        n_control = [3]

        qs = Categorical(values = construct_init_qs(n_states))
        B = Categorical(values = construct_generic_B(n_states, n_control))
        pB = Dirichlet(values = construct_pB(n_states,n_control))

        # single timestep
        n_step = 1
        policies = core.construct_policies(n_states, n_control, policy_len=n_step)

        # single observation modality
        num_obs = [4]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))
        C = Categorical(values = construct_generic_C(num_obs))

        q_pi, efe = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
        use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

        self.assertEqual(len(q_pi), len(policies))
        self.assertEqual(len(efe), len(policies))

        # multiple observation modalities
        num_obs = [3, 2]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))
        C = Categorical(values = construct_generic_C(num_obs))

        q_pi, efe = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
        use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

        self.assertEqual(len(q_pi), len(policies))
        self.assertEqual(len(efe), len(policies))

    def test_multistep_singlefac_posteriorPolicies(self):
        """
        Test for computing posterior over policies (and associated expected free energies)
        in the case of a posterior over hidden states with a single hidden state factor. 
        This version tests using a policy horizon of 3 steps ahead
        """

        n_states = [3]
        n_control = [3]

        qs = Categorical(values = construct_init_qs(n_states))
        B = Categorical(values = construct_generic_B(n_states, n_control))
        pB = Dirichlet(values = construct_pB(n_states,n_control))

        # multiple timestep
        n_step = 3
        policies = core.construct_policies(n_states, n_control, policy_len=n_step)

        # single observation modality
        num_obs = [4]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))
        C = Categorical(values = construct_generic_C(num_obs))

        q_pi, efe = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
        use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

        self.assertEqual(len(q_pi), len(policies))
        self.assertEqual(len(efe), len(policies))

        # multiple observation modalities
        num_obs = [3, 2]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))
        C = Categorical(values = construct_generic_C(num_obs))

        q_pi, efe = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
        use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

        self.assertEqual(len(q_pi), len(policies))
        self.assertEqual(len(efe), len(policies))
    
    def test_onestep_multifac_posteriorPolicies(self):
        """
        Test for computing posterior over policies (and associated expected free energies)
        in the case of a posterior over hidden states with multiple hidden state factors. 
        This version tests using a policy horizon of 1 step ahead
        """
        n_states = [3, 4]
        n_control = [3, 4]

        qs = Categorical(values = construct_init_qs(n_states))
        B = Categorical(values = construct_generic_B(n_states, n_control))
        pB = Dirichlet(values = construct_pB(n_states,n_control))

        # single timestep
        n_step = 1
        policies = core.construct_policies(n_states, n_control, policy_len=n_step)

        # single observation modality
        num_obs = [4]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))
        C = Categorical(values = construct_generic_C(num_obs))

        q_pi, efe = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
        use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

        self.assertEqual(len(q_pi), len(policies))
        self.assertEqual(len(efe), len(policies))

        # multiple observation modalities
        num_obs = [3, 2]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))
        C = Categorical(values = construct_generic_C(num_obs))

        q_pi, efe = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
        use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

        self.assertEqual(len(q_pi), len(policies))
        self.assertEqual(len(efe), len(policies))
    
    def test_multistep_multifac_posteriorPolicies(self):
        """
        Test for computing posterior over policies (and associated expected free energies)
        in the case of a posterior over hidden states with multiple hidden state factors. 
        This version tests using a policy horizon of 3 steps ahead
        """

        n_states = [3, 4]
        n_control = [3, 4]

        qs = Categorical(values = construct_init_qs(n_states))
        B = Categorical(values = construct_generic_B(n_states, n_control))
        pB = Dirichlet(values = construct_pB(n_states,n_control))

        # single timestep
        n_step = 3
        policies = core.construct_policies(n_states, n_control, policy_len=n_step)

        # single observation modality
        num_obs = [4]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))
        C = Categorical(values = construct_generic_C(num_obs))

        q_pi, efe = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
        use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

        self.assertEqual(len(q_pi), len(policies))
        self.assertEqual(len(efe), len(policies))

        # multiple observation modalities
        num_obs = [3, 2]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))
        C = Categorical(values = construct_generic_C(num_obs))

        q_pi, efe = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
        use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

        self.assertEqual(len(q_pi), len(policies))
        self.assertEqual(len(efe), len(policies))
    
    def test_construct_policies_singlefactor(self):
        """
        Test policy constructor function for single factor control states
        """

        n_states = [3]
        n_control = [3]
        control_fac_idx = [0]

        # one step policies
        policy_len = 1

        policies = core.construct_policies(n_states, n_control, policy_len, control_fac_idx)
        self.assertEqual( len(policies), n_control[0])
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len)
        
        # multistep step policies
        policy_len = 3

        policies = core.construct_policies(n_states, n_control, policy_len, control_fac_idx)
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len)

        # now leave out the optional arguments of `construct_policies` such as `n_control` and `control_fac_idx`
        n_states = [3]

        # one step policies
        policy_len = 1

        policies, n_control = core.construct_policies(n_states, None, policy_len, None)
        self.assertEqual( len(policies), n_control[0])
        self.assertEqual( n_states[0], n_control[0])
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len)
        
        # multistep step policies
        policy_len = 3

        policies, n_control = core.construct_policies(n_states, None, policy_len, None)
        self.assertEqual( n_states[0], n_control[0])
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len)
    
    def test_construct_policies_multifactor(self):
        """
        Test policy constructor function for multi factor control states
        """
        
        n_states = [3, 4]
        n_control = [3, 1]
        control_fac_idx = [0]

        # one step policies
        policy_len = 1

        policies = core.construct_policies(n_states, n_control, policy_len, control_fac_idx)
        self.assertEqual( len(policies), n_control[0])
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len)
        
        # multistep step policies
        policy_len = 3

        policies = core.construct_policies(n_states, n_control, policy_len, control_fac_idx)
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len)

        # now leave out the optional arguments of `construct_policies` such as `n_control`

        # one step policies
        policy_len = 1

        policies, n_control = core.construct_policies(n_states, None, policy_len, control_fac_idx)
        self.assertEqual( len(policies), n_control[0])
        self.assertEqual( n_control[1], 1)
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len)
        
        # multistep step policies
        policy_len = 3

        policies, n_control = core.construct_policies(n_states, None, policy_len, control_fac_idx)
        self.assertEqual( n_states[0], n_control[0])
        self.assertEqual( n_control[1], 1)
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len)

        control_fac_idx = [1]
        # one step policies
        policy_len = 1

        policies, n_control = core.construct_policies(n_states, None, policy_len, control_fac_idx)
        self.assertEqual( len(policies), n_control[1])
        self.assertEqual( n_control[0], 1)
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len)
        
        # multistep step policies
        policy_len = 3

        policies, n_control = core.construct_policies(n_states, None, policy_len, control_fac_idx)
        self.assertEqual( n_control[0], 1)
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len)
    

if __name__ == "__main__":
    unittest.main()
