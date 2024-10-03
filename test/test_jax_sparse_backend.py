#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Conor Heins, Toon Van de Maele, Ozan Catal
"""

import os
import unittest
from functools import partial

import copy

import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import vmap, nn
from jax import random as jr

from pymdp.inference import smoothing_ovf
from pymdp.legacy import utils, maths
from pymdp.legacy.control import construct_policies

from jax.experimental import sparse

from typing import Any, List, Dict

def make_model_configs(source_seed=0, num_models=4) -> Dict:
    """
    This creates a bunch of model configurations (random amounts of num states, num obs, num controls, etc.)
    that will be looped over and used as inputs for each unit test. This is intended to test each function on a variety of
    differently-dimensioned generative models
    """
    ""
    rng_keys = jr.split(jr.PRNGKey(source_seed), num_models)
    num_factors_list = [
        jr.randint(key, (1,), 1, 7)[0].item() for key in rng_keys
    ]  # list of total numbers of hidden state factors per model
    num_states_list = [jr.randint(key, (nf,), 2, 5).tolist() for nf, key in zip(num_factors_list, rng_keys)]
    num_controls_list = [jr.randint(key, (nf,), 1, 3).tolist() for nf, key in zip(num_factors_list, rng_keys)]

    rng_keys = jr.split(rng_keys[-1], num_models)
    num_modalities_list = [jr.randint(key, (1,), 1, 10)[0].item() for key in rng_keys]
    num_obs_list = [jr.randint(key, (nm,), 1, 5).tolist() for nm, key in zip(num_modalities_list, rng_keys)]

    rng_keys = jr.split(rng_keys[-1], num_models)
    A_deps_list, B_deps_list = [], []
    for nf, nm, model_key in zip(num_factors_list, num_modalities_list, rng_keys):
        modality_keys_model_i = jr.split(model_key, nm)
        num_f_per_modality = [
            jr.randint(key, shape=(), minval=1, maxval=nf + 1).item() for key in modality_keys_model_i
        ]  # this is the number of factors that each modality depends on
        A_deps_model_i = [
            sorted(jr.choice(key, a=nf, shape=(num_f_m,), replace=False).tolist())
            for key, num_f_m in zip(modality_keys_model_i, num_f_per_modality)
        ]
        A_deps_list.append(A_deps_model_i)

        factor_keys_model_i = jr.split(modality_keys_model_i[-1], nf)
        num_f_per_factor = [
            jr.randint(key, shape=(), minval=1, maxval=nf + 1).item() for key in factor_keys_model_i
        ]  # this is the number of factors that each factor depends on
        B_deps_model_i = [
            sorted(jr.choice(key, a=nf, shape=(num_f_f,), replace=False).tolist())
            for key, num_f_f in zip(factor_keys_model_i, num_f_per_factor)
        ]
        B_deps_list.append(B_deps_model_i)

    return {
        "nf_list": num_factors_list,
        "ns_list": num_states_list,
        "nc_list": num_controls_list,
        "nm_list": num_modalities_list,
        "no_list": num_obs_list,
        "A_deps_list": A_deps_list,
        "B_deps_list": B_deps_list,
    }


def make_A_full(
    A_reduced: List[np.ndarray],
    A_dependencies: List[List[int]],
    num_obs: List[int],
    num_states: List[int],
) -> np.ndarray:
    """
    Given a reduced A matrix, `A_reduced`, and a list of dependencies between hidden state factors and observation modalities, `A_dependencies`,
    return a full A matrix, `A_full`, where `A_full[m]` is the full A matrix for modality `m`. This means all redundant conditional independencies
    between observation modalities `m` and all hidden state factors (i.e. `range(len(num_states))`) are represented as lagging dimensions in `A_full`.
    """
    A_full = utils.initialize_empty_A(
        num_obs, num_states
    )  # initialize the full likelihood tensor (ALL modalities might depend on ALL factors)
    all_factors = range(len(num_states))  # indices of all hidden state factors
    for m, A_m in enumerate(A_full):

        # Step 1. Extract the list of the factors that modality `m` does NOT depend on
        non_dependent_factors = list(set(all_factors) - set(A_dependencies[m]))

        # Step 2. broadcast or tile the reduced A matrix (`A_reduced`) along the dimensions of corresponding to
        # `non_dependent_factors`, to give it the full shape of `(num_obs[m], *num_states)`
        expanded_dims = [num_obs[m]] + [1 if f in non_dependent_factors else ns for (f, ns) in enumerate(num_states)]
        tile_dims = [1] + [ns if f in non_dependent_factors else 1 for (f, ns) in enumerate(num_states)]
        A_full[m] = np.tile(A_reduced[m].reshape(expanded_dims), tile_dims)

    return A_full


class TestJaxSparseOperations(unittest.TestCase):

    def test_sparse_smoothing(self):
        cfg = {"source_seed": 1, "num_models": 4}
        gm_params = make_model_configs(**cfg)
        num_states_list, num_obs_list = (
            gm_params["ns_list"],
            gm_params["no_list"],
        )
        num_controls_list, B_deps_list = (
            gm_params["nc_list"],
            gm_params["B_deps_list"],
        )

        num_states_list = num_states_list

        n_time = 8
        n_batch = 1

        for num_states, num_obs, num_controls in zip(num_states_list, num_obs_list, num_controls_list):

            # Randomly create a B matrix that contains a lot of zeros
            B = utils.random_B_matrix(num_states, num_controls)
            B = [jnp.array(x.astype(np.float32)) for x in B]
            # Map all values below the mean to 0 to create a B tensor with zeros
            B = jtu.tree_map(
                lambda x: jnp.array(utils.norm_dist(jnp.clip((x - x.mean()), 0, 1))),
                B,
            )

            # Create a sparse array B
            sparse_B = jtu.tree_map(lambda b: sparse.BCOO.fromdense(b), B)

            # Construct a random list of actions
            policies = construct_policies(num_states, num_controls, policy_len=1)
            acs = [None for _ in range(n_time - 1)]
            for t in range(n_time - 1):
                pol = policies[np.random.randint(len(policies))]
                # Get rid of the policy length index, and insert batch dim
                pol = jnp.expand_dims(pol[0], 0)
                # Broadcast to add in the batch dim
                pol = jnp.broadcast_to(pol, (n_batch, 1, len(num_controls)))
                acs[t] = pol
            action_hist = jnp.concatenate(acs, axis=1)

            # Construct a random list of beliefs
            beliefs = [None for _ in range(len(num_states))]
            for m, ns in enumerate(num_states):
                beliefs[m] = np.random.uniform(0, 1, size=(n_batch, n_time, ns))
                beliefs[m] /= beliefs[m].sum(axis=-1, keepdims=True)
                beliefs[m] = jnp.array(beliefs[m])

            # Take the ith element from the pytree (not testing batched here)
            take_i = lambda pytree, i: jtu.tree_map(lambda leaf: leaf[i], pytree)

            for i in range(n_batch):
                smoothed_beliefs_dense = smoothing_ovf(take_i(beliefs, i), B, action_hist[i])
                dense_marginals, dense_joints = smoothed_beliefs_dense

                # sparse jax version
                smoothed_beliefs_sparse = smoothing_ovf(take_i(beliefs, i), sparse_B, action_hist[i])
                sparse_marginals, sparse_joints = smoothed_beliefs_sparse

                # test equality of marginal distributions from dense and sparse versions of smoothing
                for f, (dense_out, sparse_out) in enumerate(zip(dense_marginals, sparse_marginals)):
                    
                    self.assertTrue(np.allclose(dense_out, sparse_out))

                # test equality of joint distributions from dense and sparse versions of smoothing
                for f, (dense_out, sparse_out) in enumerate(zip(dense_joints, sparse_joints)):

                    # Densify
                    qs_joint_sparse = jnp.array([i.todense() for i in sparse_out])

                    self.assertTrue(np.allclose(dense_out, qs_joint_sparse))


if __name__ == "__main__":
    unittest.main()
