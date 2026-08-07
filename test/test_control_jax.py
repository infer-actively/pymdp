#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Dimitrije Markovic, Conor Heins
"""

import itertools
import subprocess
import sys
import unittest

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

import pymdp.control as ctl_jax
import pymdp.legacy.control as ctl_np

from pymdp.utils import (
    list_array_zeros,
    random_A_array,
    random_B_array,
    random_factorized_categorical,
)
from pymdp.legacy import utils

cfg = {"source_key": 0, "num_models": 4}

def generate_model_params():
    """
    Generate random model dimensions
    """
    rng_keys = jr.split(jr.PRNGKey(cfg["source_key"]), cfg["num_models"])
    num_factors_list = [ jr.randint(key, (1,), 1, 10)[0].item() for key in rng_keys ]
    num_states_list = [ jr.randint(key, (nf,), 1, 5).tolist() for nf, key in zip(num_factors_list, rng_keys) ]

    rng_keys = jr.split(rng_keys[-1], cfg["num_models"])
    num_modalities_list = [ jr.randint(key, (1,), 1, 10)[0].item() for key in rng_keys ]
    num_obs_list = [ jr.randint(key, (nm,), 1, 5).tolist() for nm, key in zip(num_modalities_list, rng_keys) ]

    rng_keys = jr.split(rng_keys[-1], cfg["num_models"])
    A_deps_list = []
    for nf, nm, model_key in zip(num_factors_list, num_modalities_list, rng_keys):
        keys_model_i = jr.split(model_key, nm)
        A_deps_model_i = [jr.randint(key, (nm,), 0, nf).tolist() for key in keys_model_i]
        A_deps_list.append(A_deps_model_i)
    
    return {'nf_list': num_factors_list, 
            'ns_list': num_states_list, 
            'nm_list': num_modalities_list, 
            'no_list': num_obs_list, 
            'A_deps_list': A_deps_list}

class TestControlJax(unittest.TestCase):

    def test_get_expected_obs_factorized(self):
        """
        Tests the jax-ified version of computations of expected observations under some hidden states and policy
        """
        gm_params = generate_model_params()
        _num_factors_list, num_states_list, _num_modalities_list, num_obs_list, A_deps_list = gm_params['nf_list'], gm_params['ns_list'], gm_params['nm_list'], gm_params['no_list'], gm_params['A_deps_list']
        keys = jr.split(jr.PRNGKey(42), len(num_states_list)*2).reshape((len(num_states_list), 2, 2))
        for (keys_per_element, num_states, num_obs, A_deps) in zip(keys, num_states_list, num_obs_list, A_deps_list):
            
            qs_jax = random_factorized_categorical(keys_per_element[0], num_states)
            qs_numpy = utils.obj_array_from_list(qs_jax)

            A_jax = random_A_array(keys_per_element[1], num_obs, num_states, A_dependencies=A_deps)
            A_np = [np.array(A_m) for A_m in A_jax]

            qo_test = ctl_jax.compute_expected_obs(qs_jax, A_jax, A_deps) 
            qo_validation = ctl_np.get_expected_obs_factorized([qs_numpy], A_np, A_deps) # need to wrap `qs` in list because `get_expected_obs_factorized` expects a list of `qs` (representing multiple timesteps)

            for qo_m, qo_val_m in zip(qo_test, qo_validation[0]): # need to extract first index of `qo_validation` because `get_expected_obs_factorized` returns a list of `qo` (representing multiple timesteps)
                self.assertTrue(np.allclose(qo_m, qo_val_m))

    def test_info_gain_factorized(self):
        """ 
        Unit test the `calc_states_info_gain_factorized` function by qualitatively checking that in the T-Maze (contextual bandit)
        example, the state info gain is higher for the policy that leads to visiting the cue, which is higher than state info gain
        for visiting the bandit arm, which in turn is higher than the state info gain for the policy that leads to staying in the start state.
        """

        num_states = [2, 3]  
        num_obs = [3, 3, 3]

        A_dependencies = [[0, 1], [0, 1], [1]] 
        A = []
        for m, obs in enumerate(num_obs):
            lagging_dimensions = [ns for i, ns in enumerate(num_states) if i in A_dependencies[m]]
            modality_shape = [obs] + lagging_dimensions
            A.append(np.zeros(modality_shape))
            if m == 0:
                A[m][:, :, 0] = np.ones( (num_obs[m], num_states[0]) ) / num_obs[m]
                A[m][:, :, 1] = np.ones( (num_obs[m], num_states[0]) ) / num_obs[m]
                A[m][:, :, 2] = np.array([[0.9, 0.1], [0.0, 0.0], [0.1, 0.9]]) # cue statistics
            if m == 1:
                A[m][2, :, 0] = np.ones(num_states[0])
                A[m][0:2, :, 1] = np.array([[0.6, 0.4], [0.6, 0.4]]) # bandit statistics (mapping between reward-state (first hidden state factor) and rewards (Good vs Bad))
                A[m][2, :, 2] = np.ones(num_states[0])
            if m == 2:
                A[m] = np.eye(obs)

        qs_start = list(utils.obj_array_uniform(num_states))
        qs_start[1] = np.array([1., 0., 0.]) # agent believes it's in the start state

        A = [jnp.array(A_m) for A_m in A]
        qs_start = [jnp.array(qs) for qs in qs_start]
        qo_start = ctl_jax.compute_expected_obs(qs_start, A, A_dependencies)
        
        start_info_gain = ctl_jax.compute_info_gain(qs_start, qo_start, A, A_dependencies)

        qs_arm = list(utils.obj_array_uniform(num_states))
        qs_arm[1] = np.array([0., 1., 0.]) # agent believes it's in the arm-visiting state
        qs_arm = [jnp.array(qs) for qs in qs_arm]
        qo_arm = ctl_jax.compute_expected_obs(qs_arm, A, A_dependencies)
        
        arm_info_gain = ctl_jax.compute_info_gain(qs_arm, qo_arm, A, A_dependencies)
        
        qs_cue = utils.obj_array_uniform(num_states)
        qs_cue[1] = np.array([0., 0., 1.]) # agent believes it's in the cue-visiting state
        qs_cue = [jnp.array(qs) for qs in qs_cue]
        
        qo_cue = ctl_jax.compute_expected_obs(qs_cue, A, A_dependencies)
        cue_info_gain = ctl_jax.compute_info_gain(qs_cue, qo_cue, A, A_dependencies)
        
        self.assertGreater(arm_info_gain, start_info_gain)
        self.assertGreater(cue_info_gain, arm_info_gain)

        gm_params = generate_model_params()
        _num_factors_list, num_states_list, _num_modalities_list, num_obs_list, A_deps_list = gm_params['nf_list'], gm_params['ns_list'], gm_params['nm_list'], gm_params['no_list'], gm_params['A_deps_list']
        keys = jr.split(jr.PRNGKey(1234), len(num_states_list)*2).reshape((len(num_states_list), 2, 2))
        for (keys_per_element, num_states, num_obs, A_deps) in zip(keys, num_states_list, num_obs_list, A_deps_list):

            qs_jax = random_factorized_categorical(keys_per_element[0], num_states)
            qs_numpy = utils.obj_array_from_list(qs_jax)

            A_jax = random_A_array(keys_per_element[1], num_obs, num_states, A_dependencies=A_deps)
            A_np = [np.array(A_m) for A_m in A_jax]

            qo = ctl_jax.compute_expected_obs(qs_jax, A_jax, A_deps)

            info_gain = ctl_jax.compute_info_gain(qs_jax, qo, A_jax, A_deps)
            info_gain_validation = ctl_np.calc_states_info_gain_factorized(A_np, [qs_numpy],  A_deps)

            self.assertTrue(np.allclose(info_gain, info_gain_validation, atol=1e-4))
    def test_update_posterior_policies_accepts_partial_param_posteriors(self):
        """Parameter epistemic value should work with only pA or only pB provided."""

        num_states = [3, 2]
        num_obs = [4, 3]
        num_controls = [2, 1]
        A_dependencies = [[0, 1], [1]]
        B_dependencies = [[0], [1]]

        key = jr.PRNGKey(7)
        key_A, key_B, key_qs, key_pA, key_pB = jr.split(key, 5)

        A = random_A_array(key_A, num_obs, num_states, A_dependencies=A_dependencies)
        B = random_B_array(key_B, num_states, num_controls, B_dependencies=B_dependencies)
        qs = random_factorized_categorical(key_qs, num_states)

        pA = [
            jr.uniform(k, a_m.shape, minval=0.5, maxval=2.0)
            for a_m, k in zip(A, jr.split(key_pA, len(A)))
        ]
        pB = [
            jr.uniform(k, b_f.shape, minval=0.5, maxval=2.0)
            for b_f, k in zip(B, jr.split(key_pB, len(B)))
        ]

        policy_matrix = ctl_jax.construct_policies(num_states, num_controls, policy_len=2)
        C = list_array_zeros(num_obs)
        E = jnp.ones(policy_matrix.shape[0]) / policy_matrix.shape[0]

        for pA_arg, pB_arg in ((pA, None), (None, pB)):
            q_pi, neg_efe = ctl_jax.update_posterior_policies(
                policy_matrix,
                qs,
                A,
                B,
                C,
                E,
                pA_arg,
                pB_arg,
                A_dependencies,
                B_dependencies,
                use_utility=False,
                use_states_info_gain=False,
                use_param_info_gain=True,
            )

            self.assertEqual(q_pi.shape, (policy_matrix.shape[0],))
            self.assertEqual(neg_efe.shape, (policy_matrix.shape[0],))
            self.assertTrue(np.isclose(np.array(q_pi).sum(), 1.0))
            self.assertTrue(np.all(np.isfinite(np.array(neg_efe))))

    def test_update_posterior_policies_requires_param_posterior_when_enabled(self):
        """Enabling parameter epistemic value without pA or pB should fail fast."""

        num_states = [3, 2]
        num_obs = [4, 3]
        num_controls = [2, 1]
        A_dependencies = [[0, 1], [1]]
        B_dependencies = [[0], [1]]

        key = jr.PRNGKey(11)
        key_A, key_B, key_qs = jr.split(key, 3)

        A = random_A_array(key_A, num_obs, num_states, A_dependencies=A_dependencies)
        B = random_B_array(key_B, num_states, num_controls, B_dependencies=B_dependencies)
        qs = random_factorized_categorical(key_qs, num_states)

        policy_matrix = ctl_jax.construct_policies(num_states, num_controls, policy_len=2)
        C = list_array_zeros(num_obs)
        E = jnp.ones(policy_matrix.shape[0]) / policy_matrix.shape[0]
        depth = 1
        I = [jnp.zeros((depth, ns)) for ns in num_states]
        error_msg = "use_param_info_gain=True requires at least one of pA or pB."

        with self.assertRaisesRegex(ValueError, error_msg):
            ctl_jax.update_posterior_policies(
                policy_matrix,
                qs,
                A,
                B,
                C,
                E,
                None,
                None,
                A_dependencies,
                B_dependencies,
                use_utility=False,
                use_states_info_gain=False,
                use_param_info_gain=True,
            )

        with self.assertRaisesRegex(ValueError, error_msg):
            ctl_jax.update_posterior_policies_inductive(
                policy_matrix,
                qs,
                A,
                B,
                C,
                E,
                None,
                None,
                A_dependencies,
                B_dependencies,
                I,
                use_utility=False,
                use_states_info_gain=False,
                use_param_info_gain=True,
                use_inductive=False,
            )

        with self.assertRaisesRegex(ValueError, error_msg):
            ctl_jax.compute_neg_efe_policy(
                qs,
                A,
                B,
                C,
                None,
                None,
                A_dependencies,
                B_dependencies,
                policy_matrix[0],
                use_utility=False,
                use_states_info_gain=False,
                use_param_info_gain=True,
            )

        with self.assertRaisesRegex(ValueError, error_msg):
            ctl_jax.compute_neg_efe_policy_inductive(
                qs,
                A,
                B,
                C,
                None,
                None,
                A_dependencies,
                B_dependencies,
                I,
                policy_matrix[0],
                use_utility=False,
                use_states_info_gain=False,
                use_param_info_gain=True,
                use_inductive=False,
            )

def _reference_construct_policies_array(num_states, num_controls=None, policy_len=1, control_fac_idx=None):
    """
    Independent array-based reference for policy construction, deliberately not sharing
    any code with `control._construct_policies_tuple`/`control.construct_policies`
    (which now both delegate to the same tuple-building logic internally) -- exists so
    the equivalence test below has real ground truth rather than comparing the code
    under test against itself.
    """
    num_factors = len(num_states)
    if control_fac_idx is None:
        if num_controls is not None:
            control_fac_idx = [f for f, n_c in enumerate(num_controls) if n_c > 1]
        else:
            control_fac_idx = list(range(num_factors))
    if num_controls is None:
        num_controls = [num_states[c_idx] if c_idx in control_fac_idx else 1 for c_idx in range(num_factors)]

    x = num_controls * policy_len
    policies = list(itertools.product(*[list(range(i)) for i in x]))
    for pol_i in range(len(policies)):
        policies[pol_i] = jnp.array(policies[pol_i]).reshape(policy_len, num_factors)
    return jnp.stack(policies)


class TestPoliciesTupleEquivalence(unittest.TestCase):
    """
    Regression coverage for the hashable-tuple `Policies`/`_construct_policies_tuple`
    rework (pymdp#346): verifies the pure-Python tuple builders against independent
    references, and that hashability holds across construction paths.
    """

    cases = [
        dict(num_states=[3, 3], num_controls=[3, 2], policy_len=2),
        dict(num_states=[2, 2, 1], num_controls=[2, 2, 2], policy_len=1),
        dict(num_states=[4], num_controls=[4], policy_len=3),
        dict(num_states=[2, 3, 4], num_controls=None, policy_len=1),
        dict(num_states=[2, 3, 4], num_controls=None, policy_len=1, control_fac_idx=[0, 2]),
        dict(num_states=[5], num_controls=[5], policy_len=1),
        dict(num_states=[2, 2], num_controls=[1, 2], policy_len=4),
        dict(num_states=[4, 5, 2], num_controls=[2, 3, 2], policy_len=1),
        dict(num_states=[4, 5, 2], num_controls=[2, 3, 2], policy_len=3),
    ]

    def test_construct_policies_tuple_matches_reference(self):
        for kwargs in self.cases:
            with self.subTest(**kwargs):
                reference = _reference_construct_policies_array(**kwargs)
                tup = ctl_jax._construct_policies_tuple(**kwargs)
                got = jnp.array(tup, dtype=jnp.int32)
                self.assertEqual(reference.shape, got.shape)
                self.assertTrue(jnp.array_equal(reference, got))

    def test_construct_policies_matches_tuple_builder(self):
        # construct_policies() itself now delegates to _construct_policies_tuple(), so
        # this only checks the array-wrapping is faithful, not correctness of the
        # underlying combinatorics (that's covered against the independent reference above)
        for kwargs in self.cases:
            with self.subTest(**kwargs):
                tup = ctl_jax._construct_policies_tuple(**kwargs)
                array_version = ctl_jax.construct_policies(**kwargs)
                self.assertTrue(jnp.array_equal(array_version, jnp.array(tup, dtype=jnp.int32)))

    def test_policies_array_and_tuple_construction_are_hash_and_eq_consistent(self):
        for kwargs in self.cases:
            with self.subTest(**kwargs):
                array_version = ctl_jax.construct_policies(**kwargs)
                p_from_array = ctl_jax.Policies(array_version)
                p_from_tuple = ctl_jax.Policies(p_from_array._policy_tup)

                self.assertEqual(p_from_array, p_from_tuple)
                self.assertEqual(hash(p_from_array), hash(p_from_tuple))
                self.assertEqual(p_from_array._dtype, p_from_tuple._dtype)
                self.assertEqual(hash(p_from_array._dtype), hash(p_from_tuple._dtype))
                self.assertTrue(jnp.array_equal(p_from_array.policy_arr, p_from_tuple.policy_arr))

    def test_policies_hash_differs_for_different_policy_tables(self):
        p1 = ctl_jax.Policies(ctl_jax.construct_policies(num_states=[3], num_controls=[3], policy_len=1))
        p2 = ctl_jax.Policies(ctl_jax.construct_policies(num_states=[4], num_controls=[4], policy_len=1))
        self.assertNotEqual(p1, p2)
        self.assertNotEqual(hash(p1), hash(p2))

    def test_policy_arr_cache_does_not_leak_tracers_across_jit_boundary(self):
        """
        Regression test for a tracer-leak bug in `_materialize_policy_arr`'s cache: a
        naive `functools.lru_cache` keyed on `(policy_tup, dtype)` could cache a JAX
        tracer produced by a first access inside a `jax.jit`/`lax.scan` trace, then
        hand that stale tracer to a later, unrelated eager access, raising
        `UnexpectedTracerError`. The fix only caches concrete results.

        Order matters here: the cold cache access must happen inside the jit trace
        first, otherwise this test never exercises the bug path.
        """
        ctl_jax._policy_arr_cache.clear()

        arr = ctl_jax.construct_policies(num_states=[3, 2], num_controls=[3, 2], policy_len=2)
        policies = ctl_jax.Policies(arr)
        key = (policies._policy_tup, policies._dtype)
        self.assertNotIn(key, ctl_jax._policy_arr_cache, "test setup requires a cold cache")

        # FIRST access: cold cache, happens INSIDE an active jax.jit trace -- this is
        # the moment a naive cache would store a tracer.
        @jax.jit
        def access_inside_jit(policies_obj):
            return policies_obj.policy_arr.sum()

        result = access_inside_jit(policies)
        self.assertTrue(jnp.array_equal(result, arr.sum()))

        if key in ctl_jax._policy_arr_cache:
            self.assertFalse(isinstance(ctl_jax._policy_arr_cache[key], jax.core.Tracer))

        # SECOND access: eager, same key, and actually USE the returned array (merely
        # holding a reference to a stale tracer doesn't raise -- feeding it into another
        # op outside its original trace is what raises `UnexpectedTracerError`, mirroring
        # `control.get_marginals`'s indexing/comparison on `agent.policies.policy_arr`).
        eager_result = policies.policy_arr
        self.assertFalse(isinstance(eager_result, jax.core.Tracer))
        sliced = eager_result[:, 0, 0]
        self.assertTrue(jnp.array_equal(sliced, arr[:, 0, 0]))

    def test_construct_policies_and_tuple_policies_respect_x64_config(self):
        """
        Regression test for a silent x64-downgrade bug: both `construct_policies()`
        and `Policies.__init__`'s tuple branch used to hardcode `dtype=jnp.int32`,
        discarding int64 precision under `jax_enable_x64=True`.

        Runs in a fresh subprocess since x64 is process-global config and can't be
        toggled inline without leaking into other tests in the same pytest worker.
        """
        script = (
            "import jax; jax.config.update('jax_enable_x64', True)\n"
            "import jax.numpy as jnp\n"
            "from pymdp.control import Policies, construct_policies\n"
            "arr = construct_policies(num_states=[3], num_controls=[3], policy_len=1)\n"
            "assert arr.dtype == jnp.dtype('int64'), f'construct_policies: expected int64, got {arr.dtype}'\n"
            "p = Policies(((0,), (1,), (2,)))\n"
            "assert p._dtype == jnp.dtype('int64'), f'Policies tuple branch: expected int64, got {p._dtype}'\n"
            "assert p.policy_arr.dtype == jnp.dtype('int64')\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, f"stdout: {result.stdout}\nstderr: {result.stderr}")
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
