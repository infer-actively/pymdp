# %%
import sys
import pathlib
import numpy as np
import copy

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from pymdp.distributions import Categorical, Dirichlet
from pymdp import core


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

