"""
Unit tests to compare outputs of different state inference methods.

This module tests that the following methods produce identical results:
- Original method
- End2End padded method
- Hybrid method
- Hybrid block method

"""

import unittest
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn, jit, vmap
from jax.experimental import sparse as jsparse
from functools import partial

# Import helpers from pymdp
from pymdp.utils import init_A_and_D_from_spec, get_sample_obs, generate_agent_specs_from_parameter_sets

# Import utils functions for different methods
from pymdp.utils import (
    apply_padding_batched,
    preprocess_A_for_block_diag,
    concatenate_observations_block_diag,
    apply_A_end2end_padding_batched,
    apply_obs_end2end_padding_batched
)

# Import math functions
from pymdp.maths import (
    compute_log_likelihoods_padded,
    deconstruct_lls,
    compute_log_likelihoods_block_diag,
    compute_log_likelihood_per_modality_end2end2_padded
)

# Import algorithms
from pymdp.algos import (
    run_factorized_fpi_hybrid,
    run_factorized_fpi_end2end_padded
)

# Import original inference method
from pymdp.inference import update_posterior_states

class TestInferStatesComparison(unittest.TestCase):
    """Test that all inference methods produce identical results."""

    # Default test parameters
    NUM_ITER = 8
    ATOL = 1e-5

    # Agent specs will be generated from parameter sets
    AGENT_SPECS = None

    @classmethod
    def setUpClass(cls):
        """Generate agent specs from parameter sets once before all tests."""

        # Define coordinated parameter sets
        # (num_factors, num_modalities, state_dim_upper_limit, obs_dim_upper_limit, dim_sampling_type, label)
        parameter_sets = [
            (5, 5, 5, 5, 'uniform', 'low'),
            (10, 10, 10, 10, 'uniform', 'medium'),
            (25, 25, 25, 25, 'uniform', 'high'),
            # (125, 125, 125, 125, 'uniform', 'extreme'),  # Uncomment to include extreme cases
        ]

        # Generate agent specs without dumping to file
        spec_data = generate_agent_specs_from_parameter_sets(
            parameter_sets,
            num_agents_per_set=1,
            output_file=None  # Don't save to file
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

        # Initialize agent from spec
        A, D = init_A_and_D_from_spec(
            spec["num_obs"],
            spec["num_states"],
            spec["A_dependencies"],
            A_sparsity_level=A_sparsity_level,
            batch_size=batch_size
        )
        obs = get_sample_obs(spec["num_obs"], batch_size=batch_size)

        # Prepare observations
        o_vec = [nn.one_hot(o, spec["num_obs"][m]) for m, o in enumerate(obs)]
        obs_tmp = jtu.tree_map(lambda x: jnp.squeeze(x, 1), o_vec)

        # === Original method (from pymdp.inference) ===
        infer_states_orig_pymdp = vmap(
            partial(
                update_posterior_states,
                A_dependencies=A_dependencies,
                num_iter=self.NUM_ITER,
                method='fpi'
            )
        )
        qs_original = infer_states_orig_pymdp(A, None, o_vec, None, D)

        # === Hybrid method ===
        def infer_states_hybrid(obs_padded, A_padded, D, A_shapes, A_dependencies, num_iter):
            lls_padded = compute_log_likelihoods_padded(obs_padded, A_padded)
            log_likelihoods = deconstruct_lls(lls_padded, A_shapes)
            return vmap(partial(run_factorized_fpi_hybrid, A_dependencies=A_dependencies, num_iter=num_iter))(log_likelihoods, D)

        A_padded_hybrid = apply_padding_batched(A)
        A_shapes = [a.shape for a in A]
        if use_sparsity:
            A_padded_hybrid = jsparse.BCOO.fromdense(A_padded_hybrid, n_batch=1)
        obs_padded_hybrid = apply_padding_batched(obs_tmp)

        infer_states_hybrid_jit = jit(partial(infer_states_hybrid, A_shapes=A_shapes, A_dependencies=A_dependencies, num_iter=self.NUM_ITER))
        qs_hybrid = infer_states_hybrid_jit(obs_padded_hybrid, A_padded_hybrid, D)

        # === Hybrid Block method ===
        def infer_states_hybrid_block(obs, A_big, D, state_shapes, cuts, A_dependencies, num_iter, use_einsum=False):
            """Hybrid inference using block diagonal approach for log-likelihood computation."""
            log_likelihoods = compute_log_likelihoods_block_diag(A_big, obs, state_shapes, cuts, use_einsum=use_einsum)
            return vmap(partial(run_factorized_fpi_hybrid, A_dependencies=A_dependencies, num_iter=num_iter))(log_likelihoods, D)

        A_moveaxis = [jnp.moveaxis(a, 1, -1) for a in A]
        A_big, state_shapes, cuts = preprocess_A_for_block_diag(A_moveaxis)
        if use_sparsity:
            A_big = jsparse.BCOO.fromdense(A_big, n_batch=1)
        obs_concat = concatenate_observations_block_diag(obs_tmp)

        infer_states_hybrid_block_jit = jit(partial(infer_states_hybrid_block, state_shapes=state_shapes, cuts=cuts, A_dependencies=A_dependencies, num_iter=self.NUM_ITER, use_einsum=False))
        qs_hybrid_block = infer_states_hybrid_block_jit(obs_concat, A_big, D)

        # === End2End padded method ===
        def infer_states_end2end_padded(A_padded, obs_padded, D, A_dependencies, max_obs_dim, max_state_dim, num_iter, sparsity=None):
            lls_padded = compute_log_likelihood_per_modality_end2end2_padded(obs_padded, A_padded, sparsity=sparsity)
            return run_factorized_fpi_end2end_padded(lls_padded, D, A_dependencies, max_obs_dim, max_state_dim, num_iter)

        A_padded_e2e = apply_A_end2end_padding_batched(A)
        if use_sparsity:
            A_padded_e2e = jsparse.BCOO.fromdense(A_padded_e2e)
        max_obs_dim = A_padded_e2e.shape[2]
        max_state_dim = max(A_padded_e2e.shape[3:])
        obs_padded_e2e = apply_obs_end2end_padding_batched(obs_tmp, max_obs_dim)

        infer_states_e2e_jit = jit(partial(infer_states_end2end_padded, A_dependencies=A_dependencies, max_obs_dim=max_obs_dim, max_state_dim=max_state_dim, num_iter=self.NUM_ITER, sparsity='ll_only' if use_sparsity else None))
        qs_end2end = infer_states_e2e_jit(A_padded_e2e, obs_padded_e2e, D)

        # Compare all methods
        self._compare_results(qs_original, qs_hybrid,
                             "Original", "Hybrid", spec_name)
        self._compare_results(qs_original, qs_hybrid_block,
                             "Original", "Hybrid Block", spec_name)
        self._compare_results(qs_original, qs_end2end,
                             "Original", "End2End Padded", spec_name)

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

        for batch_size in [1, 2, 4, 8]:
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