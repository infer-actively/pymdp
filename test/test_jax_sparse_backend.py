#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Conor Heins, Toon van der Maele, Ozan Catal
"""

import os
import unittest
from functools import partial

import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import vmap, nn
from jax import random as jr

from pymdp.jax.inference import smoothing_ovf
from pymdp import utils, maths

from typing import Any, List, Dict

def make_model_configs(source_seed=0, num_models=4) -> Dict:
    """
    This creates a bunch of model configurations (random amounts of num states, num obs, num controls, etc.)
    that will be looped over and used as inputs for each unit test. This is intended to test each function on a variety of 
    differently-dimensioned generative models 
    """
    ""
    rng_keys = jr.split(jr.PRNGKey(source_seed), num_models)
    num_factors_list = [ jr.randint(key, (1,), 1, 7)[0].item() for key in rng_keys ] # list of total numbers of hidden state factors per model
    num_states_list = [ jr.randint(key, (nf,), 2, 5).tolist() for nf, key in zip(num_factors_list, rng_keys) ]
    num_controls_list = [ jr.randint(key, (nf,), 1, 3).tolist() for nf, key in zip(num_factors_list, rng_keys) ]

    rng_keys = jr.split(rng_keys[-1], num_models)
    num_modalities_list = [ jr.randint(key, (1,), 1, 10)[0].item() for key in rng_keys ]
    num_obs_list = [ jr.randint(key, (nm,), 1, 5).tolist() for nm, key in zip(num_modalities_list, rng_keys) ]

    rng_keys = jr.split(rng_keys[-1], num_models)
    A_deps_list, B_deps_list = [], []
    for nf, nm, model_key in zip(num_factors_list, num_modalities_list, rng_keys):
        modality_keys_model_i = jr.split(model_key, nm)
        num_f_per_modality = [jr.randint(key, shape=(), minval=1, maxval=nf+1).item() for key in modality_keys_model_i] # this is the number of factors that each modality depends on
        A_deps_model_i = [sorted(jr.choice(key, a=nf, shape=(num_f_m,), replace=False).tolist()) for key, num_f_m in zip(modality_keys_model_i, num_f_per_modality)]
        A_deps_list.append(A_deps_model_i)

        factor_keys_model_i = jr.split(modality_keys_model_i[-1], nf)
        num_f_per_factor = [jr.randint(key, shape=(), minval=1, maxval=nf+1).item() for key in factor_keys_model_i] # this is the number of factors that each factor depends on
        B_deps_model_i = [sorted(jr.choice(key, a=nf, shape=(num_f_f,), replace=False).tolist()) for key, num_f_f in zip(factor_keys_model_i, num_f_per_factor)]
        B_deps_list.append(B_deps_model_i)

    return {'nf_list': num_factors_list, 
            'ns_list': num_states_list, 
            'nc_list': num_controls_list,
            'nm_list': num_modalities_list, 
            'no_list': num_obs_list, 
            'A_deps_list': A_deps_list,
            'B_deps_list': B_deps_list
        }

def make_A_full(A_reduced: List[np.ndarray], A_dependencies: List[List[int]], num_obs: List[int], num_states: List[int]) -> np.ndarray:
    """ 
    Given a reduced A matrix, `A_reduced`, and a list of dependencies between hidden state factors and observation modalities, `A_dependencies`,
    return a full A matrix, `A_full`, where `A_full[m]` is the full A matrix for modality `m`. This means all redundant conditional independencies
    between observation modalities `m` and all hidden state factors (i.e. `range(len(num_states))`) are represented as lagging dimensions in `A_full`.
    """
    A_full = utils.initialize_empty_A(num_obs, num_states) # initialize the full likelihood tensor (ALL modalities might depend on ALL factors)
    all_factors = range(len(num_states)) # indices of all hidden state factors
    for m, A_m in enumerate(A_full):

        # Step 1. Extract the list of the factors that modality `m` does NOT depend on
        non_dependent_factors = list(set(all_factors) - set(A_dependencies[m])) 

        # Step 2. broadcast or tile the reduced A matrix (`A_reduced`) along the dimensions of corresponding to `non_dependent_factors`, to give it the full shape of `(num_obs[m], *num_states)`
        expanded_dims = [num_obs[m]] + [1 if f in non_dependent_factors else ns for (f, ns) in enumerate(num_states)]
        tile_dims = [1] + [ns if f in non_dependent_factors else 1 for (f, ns) in enumerate(num_states)]
        A_full[m] = np.tile(A_reduced[m].reshape(expanded_dims), tile_dims)
    
    return A_full
                
class TestJaxSparseOperations(unittest.TestCase):

    def test_sparse_smoothing(self):
        cfg = {'source_seed': 0,
                'num_models': 4
            }
        gm_params = make_model_configs(**cfg)
        num_states_list, num_obs_list = gm_params['ns_list'], gm_params['no_list']

        for (num_states, num_obs) in zip(num_states_list, num_obs_list):

            # Make numpy versions of each generative model component and observatiosn
            prior = utils.random_single_categorical(num_states)
            A = utils.random_A_matrix(num_obs, num_states)

            obs = utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

            # extract B's, D',s etc.

            # dense jax version
            prior = [jnp.array(prior_f) for prior_f in prior]
            A = [jnp.array(a_m) for a_m in A]
            obs = [jnp.array(o_m) for o_m in obs]
            # ... finish making generative model
            # .... put the dense version of smoothing_ovf here


            # sparse jax version
            prior = [jnp.array(prior_f) for prior_f in prior]
            A = [jnp.array(a_m) for a_m in A]
            obs = [jnp.array(o_m) for o_m in obs]
            # ... finish making generative model
            # .... put the sparse version of smoothing_ovf here

            # for example, something like this
            for f, (dense_out, sparse_out) in enumerate(zip(smoothed_beliefs_dense, smoothed_beliefs_sparse)):
                self.assertTrue(np.allclose(dense_out[f], sparse_out[f]))



if __name__ == "__main__":
    unittest.main()






    
