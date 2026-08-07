"""Deterministic unit tests for RGM learning-path primitives.

These tests validate the interleaved-EM solver against a hand-ported
reference of SPM's within-level update (spm_VBX / spm_MDP_VB_XXX). We have
no SPM MATLAB runtime here, so rather than golden fixtures they assert
internal consistency with the reference equations described in the review
of PR #414.

Run from this directory (the example modules import each other by bare name):

    uv run pytest examples/renormalising_generative_models/test_learning_primitives.py
"""

import os
import sys

import numpy as np
import jax.numpy as jnp
import jax.nn
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from training import _interleaved_em  # noqa: E402
from mutual_information import _mi_per_mapping_from_pA  # noqa: E402
from strategies import make_strategy, SPM_PARITY, BENCHMARK  # noqa: E402
from hierarchical import (  # noqa: E402
    _top_down_D_from_cls,
    _top_down_D_hierarchical,
    hierarchical_obs_valid_mask,
)


def _manual_single_pass(pA_list, obs_list, D, valid_mask):
    """One explicit M-step -> E-step under the prior A, matching OPTIONS.B=0.

    Mirrors spm_VBX's single non-iterative belief-propagation pass followed by
    one Dirichlet accumulation: Q is initialised to D, A is formed from the
    prior counts, Q is updated once, then the counts are accumulated at that Q.
    """
    Q = D
    # A from the *prior* counts (M-step with Q = D)
    qa0 = [
        (pa + jnp.einsum("po,ps->pos", o, Q)) * (pa > 0)
        for pa, o in zip(pA_list, obs_list)
    ]
    A0 = [q / jnp.clip(q.sum(axis=1, keepdims=True), 1e-16) for q in qa0]
    # E-step: one belief update
    log_q = jnp.log(jnp.clip(D, 1e-16)) + sum(
        jnp.einsum("pos,po->ps", jnp.log(jnp.clip(A_m, 1e-16)), o)
        for A_m, o in zip(A0, obs_list)
    )
    Q1 = jax.nn.softmax(log_q, axis=-1)
    Q1 = jnp.where(valid_mask, Q1, 0.0)
    Q1 = Q1 / jnp.clip(Q1.sum(axis=-1, keepdims=True), 1e-16)
    # Final accumulation at the converged Q
    qa1 = [
        (pa + jnp.einsum("po,ps->pos", o, Q1)) * (pa > 0)
        for pa, o in zip(pA_list, obs_list)
    ]
    return qa1, Q1


@pytest.fixture
def toy_problem():
    pA = [
        jnp.stack(
            [
                jnp.array([[1.0, 1.0], [0.5, 0.5], [0.1, 2.0]]),
                jnp.array([[1.3, 1.3], [0.8, 0.8], [0.4, 2.3]]),
            ]
        )
    ]  # (2 patches, 3 obs, 2 states)
    obs = [jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])]  # (2, 3)
    D = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    mask = jnp.ones((2, 2), dtype=bool)
    return pA, obs, D, mask


def test_num_iter_default_is_one():
    """SPM's MNIST demo runs OPTIONS.B=0 -> a single non-iterative pass."""
    assert _interleaved_em.__defaults__[-1] == 1


def test_num_iter_1_matches_manual_single_pass(toy_problem):
    pA, obs, D, mask = toy_problem
    qa, Q = _interleaved_em(pA, obs, D, mask, num_iter=1)
    qa_ref, Q_ref = _manual_single_pass(pA, obs, D, mask)
    assert jnp.allclose(Q, Q_ref, atol=1e-6)
    for a, b in zip(qa, qa_ref):
        assert jnp.allclose(a, b, atol=1e-6)


def test_iteration_sharpens_posterior(toy_problem):
    """Iterating Q against an A refit to the same obs is self-reinforcing:
    the max belief mass should be no smaller after 16 iters than after 1."""
    pA, obs, D, mask = toy_problem
    _, Q1 = _interleaved_em(pA, obs, D, mask, num_iter=1)
    _, Q16 = _interleaved_em(pA, obs, D, mask, num_iter=16)
    assert jnp.all(Q16.max(axis=-1) >= Q1.max(axis=-1) - 1e-6)


def test_padded_states_stay_zero(toy_problem):
    """Invalid (padded) states must never accrue belief mass."""
    pA, obs, D, mask = toy_problem
    mask = mask.at[:, 1].set(False)  # kill the second state everywhere
    D = jnp.where(mask, D, 0.0)
    _, Q = _interleaved_em(pA, obs, D, mask, num_iter=4)
    assert jnp.allclose(Q[:, 1], 0.0)


# ---------------------------------------------------------------------------
# Top-down D messages (the learning prior for issues #2 / #3)
# ---------------------------------------------------------------------------


def test_top_down_D_from_cls_marginalises_over_classes():
    """D_top = sum_d P(top_state | digit=d) * P(digit=d | data)."""
    # cls_A[0]: (1, n_top_states=2, n_classes=3)
    cls_A = [
        jnp.array(
            [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]]
        )
    ]
    q_cls = jnp.array([[0.2, 0.5, 0.3]])  # (1, n_classes)
    D_top = _top_down_D_from_cls(cls_A, q_cls)
    expected = cls_A[0][0] @ q_cls[0]  # (n_top_states,)
    assert D_top.shape == (1, 2)
    assert jnp.allclose(D_top[0], expected, atol=1e-6)


def test_top_down_D_hierarchical_places_children_by_offset():
    """Each 2x2 offset routes its prediction to the correct child patch."""
    # 1 parent patch (parent_grid=1) -> child_grid=2 -> 4 child patches.
    max_parent, max_child = 2, 3
    # 4 modality mappings, each (n_parent=1, max_child, max_parent)
    parent_A = [
        jnp.zeros((1, max_child, max_parent)).at[0, m, 0].set(1.0)
        for m in range(4)  # modality m predicts child-state m (m<3) deterministically
    ]
    parent_A[3] = jnp.zeros((1, max_child, max_parent)).at[0, 0, 0].set(1.0)
    q_parent = jnp.array([[1.0, 0.0]])  # (1, max_parent)
    D_child = _top_down_D_hierarchical(parent_A, q_parent, child_grid=2, max_child_states=max_child)
    assert D_child.shape == (4, max_child)
    # child patch order is row-major: idx 0=(0,0)=TL, 1=(0,1)=TR, 2=(1,0)=BL, 3=(1,1)=BR
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for m, (di, dj) in enumerate(offsets):
        child_idx = (2 * 0 + di) * 2 + (2 * 0 + dj)
        pred = jnp.einsum("pcs,ps->pc", parent_A[m], q_parent)[0]
        assert jnp.allclose(D_child[child_idx], pred, atol=1e-6)


# ---------------------------------------------------------------------------
# Phase 3: per-mapping MI gate and padded-outcome masking
# ---------------------------------------------------------------------------


def test_mi_per_mapping_is_unaggregated_and_ordered():
    """MI is returned per (patch, modality), not summed. A peaked mapping has
    higher MI than a uniform one, and a uniform mapping has ~0 MI."""
    A_peaked = jnp.array([[0.99, 0.01], [0.01, 0.99]])  # (n_obs, n_states)
    A_uniform = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    pA = [jnp.stack([A_peaked, A_uniform])]  # (2 patches, 2 obs, 2 states)
    mi = _mi_per_mapping_from_pA(pA)
    assert len(mi) == 1 and mi[0].shape == (2,)
    assert float(mi[0][0]) > float(mi[0][1])       # peaked > uniform
    assert abs(float(mi[0][1])) < 1e-6             # uniform ~ 0


def test_mi_gate_is_independent_across_patches():
    """A patch's gate must depend only on its own MI change: raising one patch's
    posterior MI must not change the MI of a different patch."""
    A = jnp.array([[0.6, 0.4], [0.4, 0.6]])
    pA = [jnp.stack([A, A])]                        # two identical patches
    mi_before = _mi_per_mapping_from_pA(pA)[0]
    # Sharpen only patch 0's posterior.
    A_sharp = jnp.array([[0.95, 0.05], [0.05, 0.95]])
    pA2 = [jnp.stack([A_sharp, A])]
    mi_after = _mi_per_mapping_from_pA(pA2)[0]
    assert float(mi_after[0]) > float(mi_before[0])           # patch 0 changed
    assert abs(float(mi_after[1]) - float(mi_before[1])) < 1e-6  # patch 1 unchanged


def test_hierarchical_obs_valid_mask_marks_real_outcomes():
    """Outcome o is valid at patch p iff o < child_num_states[p][m]; padded rows
    above each mapping's real vocabulary are masked out."""

    class FakeStats:
        num_states = np.zeros((2, 2))  # 2x2 grid -> 4 patches
        # child_num_states[i][j][m]; patches p = i*2 + j
        child_num_states = [
            [[3, 3, 3, 3], [2, 2, 2, 2]],  # (0,0)->3, (0,1)->2
            [[4, 4, 4, 4], [1, 1, 1, 1]],  # (1,0)->4, (1,1)->1
        ]

    masks = hierarchical_obs_valid_mask(FakeStats(), max_child_states=4)
    assert len(masks) == 4
    m0 = masks[0]
    assert m0.shape == (4, 4)
    # per-patch valid-outcome counts
    assert [int(m0[p].sum()) for p in range(4)] == [3, 2, 4, 1]
    # padded rows are False (e.g. patch (1,1) keeps only outcome 0)
    assert bool(m0[3, 0]) and not bool(m0[3, 1])


# ---------------------------------------------------------------------------
# Phase 4: evaluation strategies
# ---------------------------------------------------------------------------


def _fake_mnist(n_train=40, n_test=15):
    # tiny stand-in raw images; make_strategy only slices + preprocesses them.
    rng = np.arange  # deterministic, no RNG
    x_train = np.tile(rng(28 * 28).reshape(1, 28, 28), (n_train, 1, 1)).astype(np.float32)
    y_train = (np.arange(n_train) % 10).astype(np.int64)
    x_test = np.tile(rng(28 * 28).reshape(1, 28, 28), (n_test, 1, 1)).astype(np.float32)
    y_test = (np.arange(n_test) % 10).astype(np.int64)
    return x_train, y_train, x_test, y_test


def test_benchmark_strategy_uses_full_test_and_no_basis_leak():
    x_train, y_train, x_test, y_test = _fake_mnist(n_train=30, n_test=15)
    p = make_strategy(BENCHMARK, x_train, y_train, x_test, y_test, n_train=20, n_eval=10)
    assert p.name == BENCHMARK
    assert p.x_basis is None                      # front-end fit on exemplars only
    assert p.x_train.shape[0] == 20               # active-learning set
    assert p.x_eval.shape[0] == 15                # the *full* official test set
    assert np.array_equal(np.asarray(p.y_eval), y_test)


def test_spm_parity_strategy_fits_11k_and_evals_in_distribution():
    x_train, y_train, x_test, y_test = _fake_mnist(n_train=40, n_test=15)
    n_train, n_eval = 20, 10
    p = make_strategy(SPM_PARITY, x_train, y_train, x_test, y_test,
                      n_train=n_train, n_eval=n_eval)
    assert p.name == SPM_PARITY
    assert p.x_basis is not None
    assert p.x_basis.shape[0] == n_train + n_eval  # basis fit on the 11k-equivalent
    assert p.x_eval.shape[0] == n_eval             # in-distribution slice
    # eval slice is drawn from the training split, right after the train set
    assert np.array_equal(np.asarray(p.y_eval), y_train[n_train:n_train + n_eval])


def test_unknown_strategy_raises():
    x_train, y_train, x_test, y_test = _fake_mnist()
    with pytest.raises(ValueError):
        make_strategy("nonsense", x_train, y_train, x_test, y_test)


def test_top_down_D_is_not_the_refined_posterior():
    """Regression for issue #2: the learning prior (top-down message) must not
    equal the refined posterior — otherwise this level's observations would be
    counted twice. Here the top-down message is uniform over classes while the
    data pulls the posterior toward one class; they must differ."""
    cls_A = [jnp.array([[[1.0, 0.0], [0.0, 1.0]]])]  # identity, 2 top-states/2 classes
    q_cls = jnp.array([[0.9, 0.1]])
    D_top = _top_down_D_from_cls(cls_A, q_cls)          # top-down message
    # A refined posterior that also folded in obs would sharpen further; assert
    # the message itself is exactly the class-marginal, unmodified by any obs.
    assert jnp.allclose(D_top[0], jnp.array([0.9, 0.1]), atol=1e-6)
