#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Conor Heins, Toon Van de Maele, Ozan Catal
"""

import unittest


import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random as jr

from pymdp.inference import smoothing_ovf
from pymdp import utils
from pymdp.legacy.control import construct_policies
from jax.experimental.sparse._base import JAXSparse

from jax.experimental import sparse

from typing import List, Dict

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


class TestJaxSparseOperations(unittest.TestCase):

    def test_sparse_smoothing(self):
        cfg = {"source_seed": 1, "num_models": 4}
        gm_params = make_model_configs(**cfg)
        num_states_list = gm_params["ns_list"]
        
        num_controls_list = gm_params["nc_list"]

        n_time = 8
        n_batch = 1

        b_keys = jr.split(jr.PRNGKey(42), cfg["num_models"])
        for b_key, num_states, num_controls in zip(b_keys, num_states_list, num_controls_list):

            # Randomly create a B matrix that contains a lot of zeros
            B = utils.random_B_array(b_key, num_states, num_controls)
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
                    qs_joint_sparse = jnp.array(
                        [j.todense() if isinstance(j, JAXSparse) else j for j in sparse_out]
                    )

                    self.assertTrue(np.allclose(dense_out, qs_joint_sparse))

    def test_sparse_smoothing_with_invalid_actions(self):
        cfg = {"source_seed": 2, "num_models": 3}
        gm_params = make_model_configs(**cfg)
        num_states_list = gm_params["ns_list"]
        num_controls_list = gm_params["nc_list"]

        n_time = 8
        n_batch = 1
        n_invalid = 3

        b_keys = jr.split(jr.PRNGKey(24), cfg["num_models"])
        for b_key, num_states, num_controls in zip(b_keys, num_states_list, num_controls_list):

            # Randomly create a B matrix with zeros so sparse conversion is meaningful.
            B = utils.random_B_array(b_key, num_states, num_controls)
            B = jtu.tree_map(
                lambda x: jnp.array(utils.norm_dist(jnp.clip((x - x.mean()), 0, 1))),
                B,
            )
            sparse_B = jtu.tree_map(lambda b: sparse.BCOO.fromdense(b), B)

            # Construct valid random actions, then pad the first `n_invalid` steps
            # with the window sentinel used by rollout (-1). These should map to
            # identity/no-transition dynamics, not to the last action slice.
            policies = construct_policies(num_states, num_controls, policy_len=1)
            acs = [None for _ in range(n_time - 1)]
            for t in range(n_time - 1):
                pol = policies[np.random.randint(len(policies))]
                pol = jnp.expand_dims(pol[0], 0)
                pol = jnp.broadcast_to(pol, (n_batch, 1, len(num_controls)))
                acs[t] = pol
            action_hist = jnp.concatenate(acs, axis=1)
            if n_invalid > 0:
                action_hist = action_hist.at[:, :n_invalid, :].set(-1)

            beliefs = [None for _ in range(len(num_states))]
            for m, ns in enumerate(num_states):
                leaf = np.random.uniform(0, 1, size=(n_batch, n_time, ns))
                leaf /= leaf.sum(axis=-1, keepdims=True)
                beliefs[m] = jnp.array(leaf)

            take_i = lambda pytree, i: jtu.tree_map(lambda leaf: leaf[i], pytree)

            for i in range(n_batch):
                smoothed_beliefs_dense = smoothing_ovf(take_i(beliefs, i), B, action_hist[i])
                dense_marginals, dense_joints = smoothed_beliefs_dense

                smoothed_beliefs_sparse = smoothing_ovf(take_i(beliefs, i), sparse_B, action_hist[i])
                sparse_marginals, sparse_joints = smoothed_beliefs_sparse

                for f, (dense_out, sparse_out) in enumerate(zip(dense_marginals, sparse_marginals)):
                    self.assertTrue(np.allclose(dense_out, sparse_out))

                for f, (dense_out, sparse_out) in enumerate(zip(dense_joints, sparse_joints)):
                    qs_joint_sparse = jnp.array(
                        [j.todense() if isinstance(j, JAXSparse) else j for j in sparse_out]
                    )
                    self.assertTrue(np.allclose(dense_out, qs_joint_sparse))

    def test_sparse_b_agent_fpi(self):
        """Test constructing Agent with sparse B (BCOO) and running FPI inference."""
        from pymdp.agent import Agent
        from pymdp.maths import spm_dot_sparse

        V = 64
        # Create sparse identity transition
        data = jnp.ones(V, dtype=jnp.float32)
        idx_3d = jnp.stack([jnp.arange(V), jnp.arange(V), jnp.zeros(V, dtype=int)], axis=1)
        sparse_B0 = sparse.BCOO((data, idx_3d), shape=(V, V, 1))

        dense_B0 = np.eye(V, dtype=np.float32)[:, :, np.newaxis]
        A = [np.eye(V, dtype=np.float32)]
        D = [np.ones(V, dtype=np.float32) / V]

        dense_agent = Agent(A=A, B=[dense_B0], D=D, batch_size=1)
        sparse_agent = Agent(A=A, B=[sparse_B0], D=D, batch_size=1)

        prior_d = [jnp.array(dense_agent.D[0])]
        prior_s = [jnp.array(sparse_agent.D[0])]

        # Run inference steps and compare posteriors
        for obs_val in [0, 5, 12, 30]:
            qs_d = dense_agent.infer_states([jnp.array([obs_val])], prior_d)
            qs_s = sparse_agent.infer_states([jnp.array([obs_val])], prior_s)
            self.assertTrue(np.allclose(qs_d[0], qs_s[0], atol=1e-6))

            prior_d = dense_agent.update_empirical_prior(jnp.array([[0]]), qs_d)
            prior_s = sparse_agent.update_empirical_prior(jnp.array([[0]]), qs_s)
            self.assertTrue(np.allclose(prior_d[0], prior_s[0], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
