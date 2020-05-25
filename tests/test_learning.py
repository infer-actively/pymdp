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

class TestLearning(unittest.TestCase):

    def test_update_pA_singleFactor_all(self):
        """
        Test for updating prior Dirichlet parameters over sensory likelihood (pA)
        in the case that all observation modalities are updated and the generative model 
        has a single hidden state factor
        """

        n_states = [3]
        qs = Categorical(values = construct_init_qs(n_states))
        learning_rate = 1.0
    
        # single observation modality
        num_obs = [4]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))

        observation = A.dot(qs,return_numpy=False).sample()

        pA_updated = core.update_likelihood_dirichlet(pA, A, observation, qs, lr=learning_rate, modalities="all",return_numpy=True)

        validation_pA = pA + learning_rate * core.spm_cross(np.eye(*num_obs)[observation], qs.values)
        self.assertTrue(np.all(pA_updated==validation_pA.values))

        # multiple observation modalities
        num_obs = [3,4]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))

        observation = A.dot(qs,return_numpy=False).sample()

        pA_updated = core.update_likelihood_dirichlet(pA, A, observation, qs, lr=learning_rate, modalities="all",return_numpy=True)

        for modality, no in enumerate(num_obs):

            validation_pA = pA[modality] + learning_rate * core.spm_cross(np.eye(no)[observation[modality]], qs.values)
            self.assertTrue(np.all(pA_updated[modality]==validation_pA.values))
    
    def test_update_pA_singleFactor_onemodality(self):
        """
        Test for updating prior Dirichlet parameters over sensory likelihood (pA)
        in the case that ONE observation modalities is updated and the generative model 
        has a single hidden state factor
        """

        n_states = [3]
        qs = Categorical(values = construct_init_qs(n_states))
        learning_rate = 1.0
    
        # multiple observation modalities
        num_obs = [3,4]

        modality_to_update = [np.random.randint(len(num_obs))]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))

        observation = A.dot(qs,return_numpy=False).sample()

        pA_updated = core.update_likelihood_dirichlet(pA, A, observation, qs, lr=learning_rate, modalities=modality_to_update,return_numpy=True)

        for modality, no in enumerate(num_obs):
            
            if modality in modality_to_update:
                validation_pA = pA[modality] + learning_rate * core.spm_cross(np.eye(no)[observation[modality]], qs.values)
            else:
                validation_pA = pA[modality]
            self.assertTrue(np.all(pA_updated[modality]==validation_pA.values))

    def test_update_pA_singleFactor_somemodalities(self):
        """
        Test for updating prior Dirichlet parameters over sensory likelihood (pA)
        in the case that SOME observation modalities are updated and the generative model 
        has a single hidden state factor
        """

        n_states = [3]
        qs = Categorical(values = construct_init_qs(n_states))
        learning_rate = 1.0
    
        # multiple observation modalities
        num_obs = [3,4,5]

        modalities_to_update = [0, 2]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))

        observation = A.dot(qs,return_numpy=False).sample()

        pA_updated = core.update_likelihood_dirichlet(pA, A, observation, qs, lr=learning_rate, modalities=modalities_to_update,return_numpy=True)

        for modality, no in enumerate(num_obs):
            
            if modality in modalities_to_update:
                validation_pA = pA[modality] + learning_rate * core.spm_cross(np.eye(no)[observation[modality]], qs.values)
            else:
                validation_pA = pA[modality]
            self.assertTrue(np.all(pA_updated[modality]==validation_pA.values))
    
    def test_update_pA_multiFactor_all(self):
        """
        Test for updating prior Dirichlet parameters over sensory likelihood (pA)
        in the case that all observation modalities are updated and the generative model 
        has multiple hidden state factors
        """

        n_states = [2, 6]
        qs = Categorical(values = construct_init_qs(n_states))
        learning_rate = 1.0
    
        # single observation modality
        num_obs = [4]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))

        observation = A.dot(qs,return_numpy=False).sample()

        pA_updated = core.update_likelihood_dirichlet(pA, A, observation, qs, lr=learning_rate, modalities="all",return_numpy=True)

        validation_pA = pA + learning_rate * core.spm_cross(np.eye(*num_obs)[observation], qs.values)
        self.assertTrue(np.all(pA_updated==validation_pA.values))

        # multiple observation modalities
        num_obs = [3,4]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))

        observation = A.dot(qs,return_numpy=False).sample()

        pA_updated = core.update_likelihood_dirichlet(pA, A, observation, qs, lr=learning_rate, modalities="all",return_numpy=True)

        for modality, no in enumerate(num_obs):

            validation_pA = pA[modality] + learning_rate * core.spm_cross(np.eye(no)[observation[modality]], qs.values)
            self.assertTrue(np.all(pA_updated[modality]==validation_pA.values))
    
    def test_update_pA_multiFactor_onemodality(self):
        """
        Test for updating prior Dirichlet parameters over sensory likelihood (pA)
        in the case that ONE observation modalities is updated and the generative model 
        has multiple hidden state factors
        """

        n_states = [2, 6]
        qs = Categorical(values = construct_init_qs(n_states))
        learning_rate = 1.0
    
        # multiple observation modalities
        num_obs = [3,4]

        modality_to_update = [np.random.randint(len(num_obs))]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))

        observation = A.dot(qs,return_numpy=False).sample()

        pA_updated = core.update_likelihood_dirichlet(pA, A, observation, qs, lr=learning_rate, modalities=modality_to_update,return_numpy=True)

        for modality, no in enumerate(num_obs):
            
            if modality in modality_to_update:
                validation_pA = pA[modality] + learning_rate * core.spm_cross(np.eye(no)[observation[modality]], qs.values)
            else:
                validation_pA = pA[modality]
            self.assertTrue(np.all(pA_updated[modality]==validation_pA.values))

    def test_update_pA_multiFactor_somemodalities(self):
        """
        Test for updating prior Dirichlet parameters over sensory likelihood (pA)
        in the case that SOME observation modalities are updated and the generative model 
        has multiple hidden state factors
        """

        n_states = [2, 6]
        qs = Categorical(values = construct_init_qs(n_states))
        learning_rate = 1.0
    
        # multiple observation modalities
        num_obs = [3,4,5]

        modalities_to_update = [0, 2]

        A = Categorical(values = construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values = construct_pA(num_obs,n_states))

        observation = A.dot(qs,return_numpy=False).sample()

        pA_updated = core.update_likelihood_dirichlet(pA, A, observation, qs, lr=learning_rate, modalities=modalities_to_update,return_numpy=True)

        for modality, no in enumerate(num_obs):
            
            if modality in modalities_to_update:
                validation_pA = pA[modality] + learning_rate * core.spm_cross(np.eye(no)[observation[modality]], qs.values)
            else:
                validation_pA = pA[modality]
            self.assertTrue(np.all(pA_updated[modality]==validation_pA.values))

    def test_update_pB_singleFactor_noActions(self):
        """
        Test for updating prior Dirichlet parameters over transition likelihood (pB)
        in the case that the one and only hidden state factor is updated, and there 
        are no actions.
        """

        n_states = [3]
        n_control = [1] # this is how we encode the fact that there aren't any actions
        qs_prev = Categorical(values = construct_init_qs(n_states))
        qs = Categorical(values = construct_init_qs(n_states))
        learning_rate = 1.0

        B = Categorical(values = np.random.rand(n_states[0],n_states[0],n_control[0]))
        B.normalize()
        pB = Dirichlet(values = np.ones_like(B.values))

        action = np.array([np.random.randint(nc) for nc in n_control])

        pB_updated = core.update_transition_dirichlet(pB,B,action,qs,qs_prev,lr=learning_rate,factors="all",return_numpy=True)

        validation_pB = pB.copy()
        validation_pB[:,:,0] += learning_rate * core.spm_cross(qs.values, qs_prev.values)
        self.assertTrue(np.all(pB_updated==validation_pB.values))


if __name__ == "__main__":
    unittest.main()
