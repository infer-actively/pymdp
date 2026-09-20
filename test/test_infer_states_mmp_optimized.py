"""
Unit tests to compare outputs of different sequence-based (MMP) state inference methods.

This module tests that the following methods produce identical results:
- Original MMP method
- Hybrid method
- Clustered Hybrid method
- Hybrid Block method
- Clustered Hybrid Block method
- End2End padded method
- Clustered End2End method

All optimized MMP methods assume trivial B_dependencies (each factor depends
only on its own state).

"""

import unittest
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn, jit, vmap
from jax.experimental import sparse as jsparse
from functools import partial

# Import helpers from pymdp
from pymdp.utils import init_A_and_D_from_spec, generate_agent_specs_from_parameter_sets

# Import MMP setup helpers (sequence-based inference with trivial B_dependencies)
from pymdp.utils import (
    init_B_from_spec,
    get_sample_action_seq,
    get_sample_state,
    get_obs_seq_from_actions,
    pad_individual_Bs
)

# Import utils functions for different methods
from pymdp.utils import (
    apply_padding_batched,
    get_A_dep_clusters,
    apply_padding_per_cluster,
    preprocess_A_for_block_diag,
    concatenate_observations_block_diag,
    prep_clustered_block_data,
    apply_A_end2end_padding_batched,
    apply_obs_end2end_padding_batched,
    apply_A_end2end_padding_per_cluster,
    apply_obs_end2end_padding_per_cluster
)

# Import math functions
from pymdp.maths import (
    compute_log_likelihoods_padded,
    deconstruct_lls,
    compute_log_likelihoods_per_cluster,
    deconstruct_log_likelihoods_per_cluster,
    compute_log_likelihoods_block_diag,
    compute_log_likelihoods_block_diag_clustered,
    compute_log_likelihood_per_modality_end2end_padded,
    compute_log_likelihoods_end2end_per_cluster
)

# Import algorithms
from pymdp.algos import (
    run_mmp_hybrid,
    run_mmp_end2end_padded,
    run_mmp_clustered_end2end
)

# Import original inference method
from pymdp.inference import update_posterior_states

class TestInferStatesMmpComparison(unittest.TestCase):
    """Test that all MMP inference methods produce identical results."""

    # Default test parameters
    NUM_ITER = 8
    T = 8
    TAU = 1.0
    ATOL = 1e-5

    # Agent specs will be generated from parameter sets
    AGENT_SPECS = None

    @classmethod
    def setUpClass(cls):
        """Generate agent specs from parameter sets once before all tests."""

        # Use explicit seed for reproducible tests
        # This keeps memory complexity low: (see issue #335)
        test_seed = 17

        # Define coordinated parameter sets
        # (num_factors, num_modalities, state_dim_upper_limit, obs_dim_upper_limit, dim_sampling_type, label)
        parameter_sets = [
            (5, 5, 5, 5, 'uniform', 'low'),
            (10, 10, 10, 10, 'uniform', 'medium'),
            (18, 18, 18, 18, 'uniform', 'high'),
            # (125, 125, 125, 125, 'uniform', 'extreme'),  # Uncomment to include extreme cases
        ]

        # Generate agent specs without dumping to file
        spec_data = generate_agent_specs_from_parameter_sets(
            parameter_sets,
            num_agents_per_set=1,
            max_A_dependency_list_size=3,
            output_file=None,  # Don't save to file
            seed=test_seed
        )

        # Load 'arbitrary dependencies' category specs
        cls.AGENT_SPECS = []
        category = 'arbitrary dependencies'
        if category in spec_data:
            specs = spec_data[category]
            for i, spec in enumerate(specs):
                # Add a name field for easier identification
                spec['name'] = f"{category}_{i}"
                spec['category'] = category
                cls.AGENT_SPECS.append(spec)
            print(f"Generated {len(cls.AGENT_SPECS)} agent specs from parameter sets", flush=True)
        else:
            raise ValueError(f"Category '{category}' not found in generated specs")

    @classmethod
    def should_skip_spec(cls, spec):
        """
        Determine if a spec should be skipped based on filtering criteria.

        Args:
            spec: Agent specification object containing metadata

        Returns:
            bool: True if spec should be skipped
        """

        metadata = spec.get('metadata', {})

        # Filter for extreme dimensions combination
        if (metadata.get('state_dim_upper_limit') == "extreme" and
            metadata.get('obs_dim_upper_limit') == "extreme"):
            return True

        # Filter for high modalities with high/extreme state dimensions
        if (metadata.get('num_modalities') == "high" and
            metadata.get('state_dim_upper_limit') in ["high", "extreme"]):
            return True

        return False

    @classmethod
    def get_specs_subset(cls, max_specs=None, filter_fn=None):
        """Get a subset of agent specs for testing.

        Args:
            max_specs: Maximum number of specs to return (None for all)
            filter_fn: Optional function to filter specs (takes spec dict, returns bool)

        Returns:
            List of agent specs
        """
        specs = cls.AGENT_SPECS

        if filter_fn:
            specs = [s for s in specs if filter_fn(s)]

        if max_specs is not None:
            specs = specs[:max_specs]

        return specs

    def _compare_results(self, r1, r2, m1, m2, spec):
        """Compare posterior results (qs) from two methods with flexible batch shapes."""
        print(f"\n=== Comparing Results: {m1} vs {m2} ===")
        print(f"Spec: {spec}")
        print(f"Number of factors - {m1}: {len(r1)}, {m2}: {len(r2)}")

        self.assertEqual(len(r1), len(r2),
                        f"[{spec}] {m1} vs {m2}: different #factors")

        ATOL = getattr(self, "ATOL", 1e-6)
        RTOL = getattr(self, "RTOL", 1e-6)
        print(f"Tolerance: ATOL={ATOL}, RTOL={RTOL}")

        for i, (a, b) in enumerate(zip(r1, r2)):
            print(f"\n--- Factor {i} ---")

            a = jnp.asarray(a)
            b = jnp.asarray(b)

            print(f"{m1} - Type: {type(a)}, Shape: {a.shape}, Dtype: {a.dtype}")
            print(f"{m2} - Type: {type(b)}, Shape: {b.shape}, Dtype: {b.dtype}")
            print(f"{m1} - Min: {float(jnp.min(a)):.6f}, Max: {float(jnp.max(a)):.6f}, Mean: {float(jnp.mean(a)):.6f}")
            print(f"{m2} - Min: {float(jnp.min(b)):.6f}, Max: {float(jnp.max(b)):.6f}, Mean: {float(jnp.mean(b)):.6f}")

            a_orig, b_orig = a.shape, b.shape

            self.assertEqual(
                a.shape, b.shape,
                f"[{spec}] {m1} vs {m2}: factor {i} shape {a.shape} != {b.shape} "
                f"(orig {a_orig} vs {b_orig})"
            )

            if not bool(jnp.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)):
                diff = jnp.abs(a - b)
                idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
                print(f"\n[{spec}] {m1} vs {m2}: factor {i}")
                print(f"Shapes: {a_orig} vs {b_orig} -> reshaped {a.shape}")
                print(f"Max diff: {float(diff[idx])}, mean diff: {float(jnp.mean(diff))}")
                self.fail(f"[{spec}] {m1} vs {m2}: factor {i} values differ (atol={ATOL}, rtol={RTOL})")
            else:
                diff = jnp.abs(a - b)
                print(f"✓ Factor {i} matches! Max diff: {float(jnp.max(diff)):.2e}, Mean diff: {float(jnp.mean(diff)):.2e}")

        print(f"\n✓ All factors match between {m1} and {m2}!\n")

    def _test_single_spec_with_batch(self, spec, batch_size=4, A_sparsity_level=None, use_sparsity=False):
        """Test a single agent spec with batch size.

        Args:
            spec: Agent specification
            batch_size: Batch size for inference
            A_sparsity_level: Sparsity level for A matrix generation (0.0-1.0)
            use_sparsity: Whether to convert A matrices to sparse BCOO format
        """
        spec_name = spec['name']
        A_dependencies = spec["A_dependencies"]
        num_states = spec["num_states"]
        num_obs = spec["num_obs"]
        num_controls = [2 for i in range(spec["num_factors"])]

        # The optimized MMP routines assume trivial B_dependencies
        # (each factor depends on its own state only)
        B_dependencies = [[f] for f in range(spec["num_factors"])]

        # Initialize agent from spec
        A, D = init_A_and_D_from_spec(
            num_obs,
            num_states,
            A_dependencies,
            A_sparsity_level=A_sparsity_level,
            batch_size=batch_size
        )
        B = init_B_from_spec(num_states, num_controls, batch_size=batch_size)

        # Roll out an action/observation sequence and prepare inputs
        past_actions = get_sample_action_seq(num_controls, self.T, batch_size=batch_size)
        start_state = get_sample_state(num_states, batch_size=batch_size)
        obs_seq = get_obs_seq_from_actions(past_actions, start_state, A, A_dependencies, B, B_dependencies)
        o_vec = [nn.one_hot(o, num_obs[m]) for m, o in enumerate(obs_seq)]

        B_padded = pad_individual_Bs(B)

        # The optimized MMP methods operate on dense log-likelihoods, so the
        # end2end likelihood computation always uses sparsity='ll_only'
        # (identical to a plain log-stable for dense A, densifying for sparse A).
        sparsity = 'll_only'

        # === Original MMP method (from pymdp.inference) ===
        infer_states_orig_pymdp = vmap(
            partial(
                update_posterior_states,
                A_dependencies=A_dependencies,
                B_dependencies=B_dependencies,
                num_iter=self.NUM_ITER,
                method='mmp'
            )
        )
        qs_original = infer_states_orig_pymdp(A, B, o_vec, past_actions, D)

        # === Hybrid method ===
        def infer_states_mmp_hybrid(obs_padded, A_padded, D, past_actions, B_padded, A_shapes, num_states, A_dependencies, B_dependencies, num_iter, tau=1.):
            lls_padded = vmap(compute_log_likelihoods_padded, in_axes=(1, None), out_axes=1)(obs_padded, A_padded)
            log_likelihoods = deconstruct_lls(lls_padded, A_shapes, has_time_axis=True)
            return vmap(
                partial(run_mmp_hybrid, num_states=num_states, A_dependencies=A_dependencies, B_dependencies=B_dependencies, num_iter=num_iter, tau=tau)
            )(log_likelihoods, D, past_actions, B_padded)

        A_padded_hybrid = apply_padding_batched(A)
        A_shapes = [a.shape for a in A]
        if use_sparsity:
            A_padded_hybrid = jsparse.BCOO.fromdense(A_padded_hybrid, n_batch=1)
        obs_padded_hybrid = apply_padding_batched(o_vec)

        infer_states_mmp_hybrid_jit = jit(partial(infer_states_mmp_hybrid, A_shapes=A_shapes, num_states=num_states, A_dependencies=A_dependencies, B_dependencies=B_dependencies, num_iter=self.NUM_ITER, tau=self.TAU))
        qs_hybrid = infer_states_mmp_hybrid_jit(obs_padded_hybrid, A_padded_hybrid, D, past_actions, B_padded)

        # === Clustered Hybrid method ===
        def infer_states_mmp_clustered_hybrid(obs_clusters, A_clusters, D, past_actions, B_padded, c2o_mapping, A_shapes, num_states, A_dependencies, B_dependencies, num_iter, tau=1.):
            ll_clusters = vmap(compute_log_likelihoods_per_cluster, in_axes=(1, None), out_axes=1)(obs_clusters, A_clusters)
            log_likelihoods = deconstruct_log_likelihoods_per_cluster(ll_clusters, A_shapes, c2o_mapping, has_time_axis=True)
            return vmap(
                partial(run_mmp_hybrid, num_states=num_states, A_dependencies=A_dependencies, B_dependencies=B_dependencies, num_iter=num_iter, tau=tau)
            )(log_likelihoods, D, past_actions, B_padded)

        c2o_mapping = get_A_dep_clusters(A_dependencies)
        A_clusters_hybrid = apply_padding_per_cluster(A, c2o_mapping)
        obs_clusters_hybrid = apply_padding_per_cluster(o_vec, c2o_mapping)
        if use_sparsity:
            A_clusters_hybrid = [jsparse.BCOO.fromdense(a) for a in A_clusters_hybrid]

        infer_states_mmp_clustered_hybrid_jit = jit(partial(infer_states_mmp_clustered_hybrid, c2o_mapping=c2o_mapping, A_shapes=A_shapes, num_states=num_states, A_dependencies=A_dependencies, B_dependencies=B_dependencies, num_iter=self.NUM_ITER, tau=self.TAU))
        qs_clustered_hybrid = infer_states_mmp_clustered_hybrid_jit(obs_clusters_hybrid, A_clusters_hybrid, D, past_actions, B_padded)

        # === Hybrid Block method ===
        def infer_states_mmp_hybrid_block(A_big, obs_big, D, past_actions, B_padded, state_shapes, cuts, num_states, A_dependencies, B_dependencies, num_iter, tau=1., use_einsum=False):
            log_likelihoods = vmap(
                partial(compute_log_likelihoods_block_diag, use_einsum=use_einsum), in_axes=(None, 1, None, None), out_axes=1
            )(A_big, obs_big, state_shapes, cuts)
            return vmap(
                partial(run_mmp_hybrid, num_states=num_states, A_dependencies=A_dependencies, B_dependencies=B_dependencies, num_iter=num_iter, tau=tau)
            )(log_likelihoods, D, past_actions, B_padded)

        A_moveaxis = [jnp.moveaxis(a, 1, -1) for a in A]
        A_big, state_shapes, cuts = preprocess_A_for_block_diag(A_moveaxis)
        if use_sparsity:
            A_big = jsparse.BCOO.fromdense(A_big, n_batch=1)
        obs_concat = concatenate_observations_block_diag(o_vec)

        infer_states_mmp_hybrid_block_jit = jit(partial(infer_states_mmp_hybrid_block, state_shapes=state_shapes, cuts=cuts, num_states=num_states, A_dependencies=A_dependencies, B_dependencies=B_dependencies, num_iter=self.NUM_ITER, tau=self.TAU, use_einsum=False))
        qs_hybrid_block = infer_states_mmp_hybrid_block_jit(A_big, obs_concat, D, past_actions, B_padded)

        # === Clustered Hybrid Block method ===
        def infer_states_mmp_clustered_hybrid_block(A_groups, obs_groups, D, past_actions, B_padded, shape_groups, cut_groups, group_mapping, num_states, A_dependencies, B_dependencies, num_iter, tau=1.):
            num_modalities = len(A_dependencies)
            log_likelihoods = vmap(
                compute_log_likelihoods_block_diag_clustered, in_axes=(None, 1, None, None, None, None), out_axes=1
            )(A_groups, obs_groups, shape_groups, cut_groups, group_mapping, num_modalities)
            return vmap(
                partial(run_mmp_hybrid, num_states=num_states, A_dependencies=A_dependencies, B_dependencies=B_dependencies, num_iter=num_iter, tau=tau)
            )(log_likelihoods, D, past_actions, B_padded)

        # Cluster modalities (host-side preprocessing) and build one block-diagonal system per group
        A_block, obs_block, state_shapes_block, cuts_block, group_mapping = prep_clustered_block_data(A, o_vec)
        if use_sparsity:
            A_block = [jsparse.BCOO.fromdense(a, n_batch=1) for a in A_block]

        infer_states_mmp_clustered_hybrid_block_jit = jit(partial(infer_states_mmp_clustered_hybrid_block, shape_groups=state_shapes_block, cut_groups=cuts_block, group_mapping=group_mapping, num_states=num_states, A_dependencies=A_dependencies, B_dependencies=B_dependencies, num_iter=self.NUM_ITER, tau=self.TAU))
        qs_clustered_hybrid_block = infer_states_mmp_clustered_hybrid_block_jit(A_block, obs_block, D, past_actions, B_padded)

        # === End2End padded method ===
        def infer_states_mmp_end2end_padded(obs_padded, A_padded, D, past_actions, B_padded, num_states, A_dependencies, B_dependencies, num_iter, tau=1., sparsity='ll_only'):
            lls_padded = vmap(
                partial(compute_log_likelihood_per_modality_end2end_padded, sparsity=sparsity), in_axes=(2, None)
            )(obs_padded, A_padded)
            return run_mmp_end2end_padded(lls_padded, D, past_actions, B_padded, num_states, A_dependencies, B_dependencies, num_iter=num_iter, tau=tau)

        A_padded_e2e = apply_A_end2end_padding_batched(A)
        if use_sparsity:
            A_padded_e2e = jsparse.BCOO.fromdense(A_padded_e2e)
        max_obs_dim = A_padded_e2e.shape[2]
        obs_padded_e2e = apply_obs_end2end_padding_batched(o_vec, max_obs_dim)

        infer_states_mmp_e2e_jit = jit(partial(infer_states_mmp_end2end_padded, num_states=num_states, A_dependencies=A_dependencies, B_dependencies=B_dependencies, num_iter=self.NUM_ITER, tau=self.TAU, sparsity=sparsity))
        qs_end2end = infer_states_mmp_e2e_jit(obs_padded_e2e, A_padded_e2e, D, past_actions, B_padded)

        # === Clustered End2End method ===
        def infer_states_mmp_clustered_end2end(obs_clusters, A_clusters, D, past_actions, B_padded, c2o_mapping, c2s_mapping, max_state_dims, num_states, A_dependencies, B_dependencies, num_iter, tau=1., sparsity='ll_only'):
            ll_clusters = vmap(
                partial(compute_log_likelihoods_end2end_per_cluster, sparsity=sparsity), in_axes=(2, None)
            )(obs_clusters, A_clusters)
            return run_mmp_clustered_end2end(ll_clusters, D, past_actions, B_padded, c2o_mapping, c2s_mapping, max_state_dims, num_states, A_dependencies, B_dependencies, num_iter=num_iter, tau=tau)

        A_clusters_e2e = apply_A_end2end_padding_per_cluster(A, c2o_mapping)
        max_obs_dims = [a.shape[2] for a in A_clusters_e2e]
        max_state_dims = [a.shape[-1] for a in A_clusters_e2e]
        obs_clusters_e2e = apply_obs_end2end_padding_per_cluster(o_vec, c2o_mapping, max_obs_dims)
        c2s_mapping = [[s for o in o_list for s in A_dependencies[o]] for o_list in c2o_mapping]
        if use_sparsity:
            A_clusters_e2e = [jsparse.BCOO.fromdense(a) for a in A_clusters_e2e]

        infer_states_mmp_clustered_e2e_jit = jit(partial(infer_states_mmp_clustered_end2end, c2o_mapping=c2o_mapping, c2s_mapping=c2s_mapping, max_state_dims=max_state_dims, num_states=num_states, A_dependencies=A_dependencies, B_dependencies=B_dependencies, num_iter=self.NUM_ITER, tau=self.TAU, sparsity=sparsity))
        qs_clustered_end2end = infer_states_mmp_clustered_e2e_jit(obs_clusters_e2e, A_clusters_e2e, D, past_actions, B_padded)

        # Compare all methods
        self._compare_results(qs_original, qs_hybrid,
                             "Original MMP", "Hybrid", spec_name)
        self._compare_results(qs_original, qs_clustered_hybrid,
                             "Original MMP", "Clustered Hybrid", spec_name)
        self._compare_results(qs_original, qs_hybrid_block,
                             "Original MMP", "Hybrid Block", spec_name)
        self._compare_results(qs_original, qs_clustered_hybrid_block,
                             "Original MMP", "Clustered Hybrid Block", spec_name)
        self._compare_results(qs_original, qs_end2end,
                             "Original MMP", "End2End Padded", spec_name)
        self._compare_results(qs_original, qs_clustered_end2end,
                             "Original MMP", "Clustered End2End", spec_name)

    # Test methods for different subsets of specs
    def test_first_spec_with_batch(self):
        """Test first agent spec with batch size."""
        self._test_single_spec_with_batch(self.AGENT_SPECS[0], batch_size=4)

    def test_small_subset_with_batch(self):
        """Test first 5 agent specs with batch size."""
        specs = self.get_specs_subset(max_specs=5)
        batch_size = 4
        skipped_count = 0
        tested_count = 0
        for spec in specs:
            print(f"Testing spec '{spec['name']}' [BS={batch_size}]")

            should_skip_extreme = self.should_skip_spec(spec)

            if should_skip_extreme:
                skipped_count += 1
                print("  ⏭️  Skipped due to extreme dimensions")
                continue

            with self.subTest(spec=spec['name']):
                tested_count += 1
                self._test_single_spec_with_batch(spec, batch_size=batch_size)

        print(f"\n✓ Tested {tested_count} specs, skipped {skipped_count} specs")

    def test_different_batch_sizes(self):
        """Test that batch size method works with different batch sizes."""
        spec = self.AGENT_SPECS[0]  # Use first spec

        for batch_size in [1, 4]:
            with self.subTest(batch_size=batch_size):
                self._test_single_spec_with_batch(spec, batch_size=batch_size)

    def test_low_complexity_specs_with_batch(self):
        """Test only low complexity specs (low num_factors and low num_modalities) with batch."""
        def is_low_complexity(spec):
            return (spec.get('num_factors', 0) == 5 and
                   spec.get('num_modalities', 0) == 5)

        specs = self.get_specs_subset(filter_fn=is_low_complexity, max_specs=25)
        print(f"\nTesting {len(specs)} low complexity specs with batch size")

        batch_size = 4
        skipped_count = 0
        tested_count = 0
        for spec in specs:
            print(f"Testing spec '{spec['name']}' [BS={batch_size}]")

            should_skip_extreme = self.should_skip_spec(spec)

            if should_skip_extreme:
                skipped_count += 1
                print("  ⏭️  Skipped due to extreme dimensions")
                continue

            with self.subTest(spec=spec['name']):
                tested_count += 1
                self._test_single_spec_with_batch(spec, batch_size=batch_size)

        print(f"\n✓ Tested {tested_count} specs, skipped {skipped_count} specs")

    def test_sparsity_with_batch(self):
        """Test sparse matrix support with batch size."""
        print("\nTesting sparsity support (with batch)")

        # Use first spec for sparsity test
        spec = self.AGENT_SPECS[0]
        A_sparsity_level = 0.95  # 95% sparse
        batch_size = 4

        print(f"Testing spec '{spec['name']}' [BS={batch_size}] with sparsity={A_sparsity_level}")

        # Test with sparsity using the helper function
        self._test_single_spec_with_batch(spec, batch_size=batch_size, A_sparsity_level=A_sparsity_level, use_sparsity=True)

        print("✓ Sparse matrix operations completed successfully")

    def test_all_agents_with_batch(self):
        """Test all agent specs with batch size."""
        specs = self.AGENT_SPECS
        batch_size = 4
        skipped_count = 0
        tested_count = 0

        print(f"\nTesting all {len(specs)} agent specs with batch size {batch_size}")

        for spec in specs:
            print(f"Testing spec '{spec['name']}' [BS={batch_size}]")

            should_skip_extreme = self.should_skip_spec(spec)

            if should_skip_extreme:
                skipped_count += 1
                print("  ⏭️  Skipped due to extreme dimensions")
                continue

            with self.subTest(spec=spec['name']):
                tested_count += 1
                self._test_single_spec_with_batch(spec, batch_size=batch_size)

        print(f"\n✓ Tested {tested_count} specs, skipped {skipped_count} specs")

if __name__ == '__main__':
    unittest.main()
