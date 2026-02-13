#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for associative-scan HMM filtering/smoothing."""

import unittest
from jax import numpy as jnp, random as jr, nn
from jax import vmap, grad
from jax import tree_util as jtu

from pymdp.algos import (
    hmm_filter_scan_rowstoch,
    hmm_smoother_scan_rowstoch,
    hmm_filter_scan_colstoch,
    hmm_smoother_scan_colstoch,
)
from pymdp import inference, utils
from pymdp.agent import Agent
from pymdp.maths import log_stable, MINVAL


def _normalize_rows(x, axis=-1):
    s = jnp.sum(x, axis=axis, keepdims=True)
    return x / jnp.clip(s, min=MINVAL)


def _random_simplex(key, shape):
    x = jr.uniform(key, shape) + 1e-3
    return _normalize_rows(x, axis=-1)


def _normalize_cols(x, axis=0):
    s = jnp.sum(x, axis=axis, keepdims=True)
    return x / jnp.clip(s, min=MINVAL)


def _random_col_stochastic(key, shape):
    x = jr.uniform(key, shape) + 1e-3
    axis = 0 if len(shape) == 2 else 1
    return _normalize_cols(x, axis=axis)


def _sparse_absorbing_rowstoch(K):
    """
    Build a row-stochastic transition matrix with many strict zeros and one absorbing state.
    """
    A = jnp.zeros((K, K))
    idx = jnp.arange(K - 1)
    A = A.at[idx, idx].set(0.9)
    A = A.at[idx, idx + 1].set(0.1)
    A = A.at[K - 1, K - 1].set(1.0)  # absorbing state
    return A


def _near_zero_rowstoch(K, tiny=1e-20):
    """
    Build a row-stochastic matrix where off-diagonal terms are near zero.
    """
    A = jnp.full((K, K), tiny)
    diag_val = 1.0 - (K - 1) * tiny
    A = A.at[jnp.arange(K), jnp.arange(K)].set(diag_val)
    return A


def _reference_filter(initial_probs, transition_mats, log_likelihoods):
    T, K = log_likelihoods.shape
    if transition_mats.ndim == 2:
        transition_mats = jnp.broadcast_to(transition_mats, (T - 1, K, K))

    filtered = []
    predicted = []
    log_c = []

    # t = 0
    pred0 = initial_probs
    ll0 = log_likelihoods[0]
    ll0_max = jnp.max(ll0)
    w0 = jnp.exp(ll0 - ll0_max)
    weights0 = pred0 * w0
    c0 = jnp.sum(weights0)
    filt0 = weights0 / c0
    log_c0 = log_stable(c0) + ll0_max

    predicted.append(pred0)
    filtered.append(filt0)
    log_c.append(log_c0)

    # t >= 1
    for t in range(1, T):
        pred_t = filtered[t - 1] @ transition_mats[t - 1]
        ll_t = log_likelihoods[t]
        ll_t_max = jnp.max(ll_t)
        w_t = jnp.exp(ll_t - ll_t_max)
        weights_t = pred_t * w_t
        c_t = jnp.sum(weights_t)
        filt_t = weights_t / c_t
        log_c_t = log_stable(c_t) + ll_t_max

        predicted.append(pred_t)
        filtered.append(filt_t)
        log_c.append(log_c_t)

    predicted = jnp.stack(predicted, axis=0)
    filtered = jnp.stack(filtered, axis=0)
    log_c = jnp.stack(log_c, axis=0)
    marginal_loglik = jnp.sum(log_c)
    return marginal_loglik, filtered, predicted, log_c


def _reference_smoother(initial_probs, transition_mats, log_likelihoods):
    T, K = log_likelihoods.shape
    if transition_mats.ndim == 2:
        transition_mats = jnp.broadcast_to(transition_mats, (T - 1, K, K))

    marginal_loglik, filtered, predicted, log_c = _reference_filter(
        initial_probs, transition_mats, log_likelihoods
    )

    c = jnp.exp(log_c)
    beta = [None] * T
    beta[-1] = jnp.ones((K,))

    for t in range(T - 2, -1, -1):
        ll_next = log_likelihoods[t + 1]
        scale = jnp.exp(ll_next)
        beta[t] = (transition_mats[t] * scale[None, :]) @ beta[t + 1] / c[t + 1]

    beta = jnp.stack(beta, axis=0)

    smoothed = filtered * beta
    smoothed = smoothed / jnp.sum(smoothed, axis=-1, keepdims=True)

    if T == 1:
        trans_probs = jnp.zeros((0, K, K))
        cond_probs = jnp.zeros((0, K, K))
    else:
        xi_list = []
        for t in range(T - 1):
            ll_next = log_likelihoods[t + 1]
            numer = (
                filtered[t][:, None]
                * transition_mats[t]
                * jnp.exp(ll_next)[None, :]
                * beta[t + 1][None, :]
                / c[t + 1]
            )
            numer = numer / jnp.sum(numer)
            xi_list.append(numer)

        trans_probs = jnp.stack(xi_list, axis=0)
        cond_probs = trans_probs.transpose(0, 2, 1) / smoothed[1:, :, None]

    return marginal_loglik, filtered, predicted, smoothed, trans_probs, cond_probs


def _reference_filter_col(initial_probs, B_mats, log_likelihoods):
    T, K = log_likelihoods.shape
    if B_mats.ndim == 2:
        B_mats = jnp.broadcast_to(B_mats, (T - 1, K, K))

    filtered = []
    predicted = []
    log_c = []

    pred0 = initial_probs
    ll0 = log_likelihoods[0]
    ll0_max = jnp.max(ll0)
    w0 = jnp.exp(ll0 - ll0_max)
    weights0 = pred0 * w0
    c0 = jnp.sum(weights0)
    filt0 = weights0 / c0
    log_c0 = log_stable(c0) + ll0_max

    predicted.append(pred0)
    filtered.append(filt0)
    log_c.append(log_c0)

    for t in range(1, T):
        pred_t = B_mats[t - 1] @ filtered[t - 1]
        ll_t = log_likelihoods[t]
        ll_t_max = jnp.max(ll_t)
        w_t = jnp.exp(ll_t - ll_t_max)
        weights_t = pred_t * w_t
        c_t = jnp.sum(weights_t)
        filt_t = weights_t / c_t
        log_c_t = log_stable(c_t) + ll_t_max

        predicted.append(pred_t)
        filtered.append(filt_t)
        log_c.append(log_c_t)

    predicted = jnp.stack(predicted, axis=0)
    filtered = jnp.stack(filtered, axis=0)
    log_c = jnp.stack(log_c, axis=0)
    marginal_loglik = jnp.sum(log_c)
    return marginal_loglik, filtered, predicted, log_c


def _reference_smoother_col(initial_probs, B_mats, log_likelihoods):
    T, K = log_likelihoods.shape
    if B_mats.ndim == 2:
        B_mats = jnp.broadcast_to(B_mats, (T - 1, K, K))

    marginal_loglik, filtered, predicted, log_c = _reference_filter_col(
        initial_probs, B_mats, log_likelihoods
    )

    row_scale = jnp.exp(log_likelihoods[1:] - log_c[1:, None])
    beta = [None] * T
    beta[-1] = jnp.ones((K,))

    for t in range(T - 2, -1, -1):
        beta[t] = B_mats[t].T @ (row_scale[t] * beta[t + 1])

    beta = jnp.stack(beta, axis=0)

    smoothed = filtered * beta
    smoothed = smoothed / jnp.clip(jnp.sum(smoothed, axis=-1, keepdims=True), min=MINVAL)

    if T == 1:
        trans_probs = jnp.zeros((0, K, K))
        cond_probs = jnp.zeros((0, K, K))
    else:
        xi_list = []
        for t in range(T - 1):
            numer = (
                B_mats[t]
                * filtered[t][None, :]
                * row_scale[t][:, None]
                * beta[t + 1][:, None]
            )
            numer = numer / jnp.clip(jnp.sum(numer), min=MINVAL)
            xi_list.append(numer)

        xi_nc = jnp.stack(xi_list, axis=0)  # (T-1, next, curr)
        cond_probs = xi_nc / jnp.clip(smoothed[1:, :, None], min=MINVAL)
        trans_probs = xi_nc.transpose(0, 2, 1)

    return marginal_loglik, filtered, predicted, smoothed, trans_probs, cond_probs


class TestAssociativeScanHMM(unittest.TestCase):
    def _make_case(self, key, T, K, time_varying=False):
        k1, k2, k3 = jr.split(key, 3)
        initial_probs = _random_simplex(k1, (K,))
        if time_varying:
            transition_mats = _random_simplex(k2, (T - 1, K, K))
        else:
            transition_mats = _random_simplex(k2, (K, K))
        log_likelihoods = nn.log_softmax(jr.normal(k3, (T, K)), axis=-1)
        return initial_probs, transition_mats, log_likelihoods

    def _make_case_col(self, key, T, K, time_varying=False):
        k1, k2, k3 = jr.split(key, 3)
        initial_probs = _random_simplex(k1, (K,))
        if time_varying:
            B_mats = _random_col_stochastic(k2, (T - 1, K, K))
        else:
            B_mats = _random_col_stochastic(k2, (K, K))
        log_likelihoods = nn.log_softmax(jr.normal(k3, (T, K)), axis=-1)
        return initial_probs, B_mats, log_likelihoods

    def _make_batch_row(self, key, batch_size, T, K, time_varying=False):
        cases = [self._make_case(jr.fold_in(key, i), T, K, time_varying) for i in range(batch_size)]
        init = jnp.stack([c[0] for c in cases], axis=0)
        trans = jnp.stack([c[1] for c in cases], axis=0)
        ll = jnp.stack([c[2] for c in cases], axis=0)
        return init, trans, ll

    def _make_batch_col(self, key, batch_size, T, K, time_varying=False):
        cases = [self._make_case_col(jr.fold_in(key, i), T, K, time_varying) for i in range(batch_size)]
        init = jnp.stack([c[0] for c in cases], axis=0)
        trans = jnp.stack([c[1] for c in cases], axis=0)
        ll = jnp.stack([c[2] for c in cases], axis=0)
        return init, trans, ll

    def _assert_close(self, a, b, atol=1e-5, rtol=1e-5):
        self.assertTrue(bool(jnp.allclose(a, b, atol=atol, rtol=rtol)))

    def _assert_all_finite_tree(self, tree):
        leaves = jtu.tree_leaves(tree)
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(x))) for x in leaves))

    def test_filter_scan_matches_reference_stationary(self):
        key = jr.PRNGKey(0)
        initial_probs, transition_mats, log_likelihoods = self._make_case(
            key, T=6, K=4, time_varying=False
        )

        mll_ref, filt_ref, pred_ref, _ = _reference_filter(
            initial_probs, transition_mats, log_likelihoods
        )
        mll_scan, filt_scan, pred_scan = hmm_filter_scan_rowstoch(
            initial_probs, transition_mats, log_likelihoods
        )

        self._assert_close(mll_ref, mll_scan)
        self._assert_close(filt_ref, filt_scan)
        self._assert_close(pred_ref, pred_scan)

    def test_smoother_scan_matches_reference_time_varying(self):
        key = jr.PRNGKey(1)
        initial_probs, transition_mats, log_likelihoods = self._make_case(
            key, T=5, K=3, time_varying=True
        )

        (mll_ref, filt_ref, pred_ref, smooth_ref, xi_ref, cond_ref) = _reference_smoother(
            initial_probs, transition_mats, log_likelihoods
        )
        (mll_scan, filt_scan, pred_scan, smooth_scan, xi_scan, cond_scan) = hmm_smoother_scan_rowstoch(
            initial_probs, transition_mats, log_likelihoods
        )

        self._assert_close(mll_ref, mll_scan)
        self._assert_close(filt_ref, filt_scan)
        self._assert_close(pred_ref, pred_scan)
        self._assert_close(smooth_ref, smooth_scan)
        self._assert_close(xi_ref, xi_scan)
        self._assert_close(cond_ref, cond_scan)

    def test_smoother_scan_T1_edge(self):
        key = jr.PRNGKey(2)
        initial_probs, transition_mats, log_likelihoods = self._make_case(
            key, T=1, K=3, time_varying=False
        )

        (mll_ref, filt_ref, pred_ref, smooth_ref, xi_ref, cond_ref) = _reference_smoother(
            initial_probs, transition_mats, log_likelihoods
        )
        (mll_scan, filt_scan, pred_scan, smooth_scan, xi_scan, cond_scan) = hmm_smoother_scan_rowstoch(
            initial_probs, transition_mats, log_likelihoods
        )

        self._assert_close(mll_ref, mll_scan)
        self._assert_close(filt_ref, filt_scan)
        self._assert_close(pred_ref, pred_scan)
        self._assert_close(smooth_ref, smooth_scan)
        self._assert_close(xi_ref, xi_scan)
        self._assert_close(cond_ref, cond_scan)

    def test_filter_scan_colstoch_matches_reference_stationary(self):
        key = jr.PRNGKey(3)
        initial_probs, B_mats, log_likelihoods = self._make_case_col(
            key, T=6, K=4, time_varying=False
        )

        mll_ref, filt_ref, pred_ref, _ = _reference_filter_col(
            initial_probs, B_mats, log_likelihoods
        )
        mll_scan, filt_scan, pred_scan = hmm_filter_scan_colstoch(
            initial_probs, B_mats, log_likelihoods
        )

        self._assert_close(mll_ref, mll_scan)
        self._assert_close(filt_ref, filt_scan)
        self._assert_close(pred_ref, pred_scan)

    def test_smoother_scan_colstoch_matches_reference_time_varying(self):
        key = jr.PRNGKey(4)
        initial_probs, B_mats, log_likelihoods = self._make_case_col(
            key, T=5, K=3, time_varying=True
        )

        (mll_ref, filt_ref, pred_ref, smooth_ref, xi_ref, cond_ref) = _reference_smoother_col(
            initial_probs, B_mats, log_likelihoods
        )
        (mll_scan, filt_scan, pred_scan, smooth_scan, xi_scan, cond_scan) = hmm_smoother_scan_colstoch(
            initial_probs, B_mats, log_likelihoods, return_trans_probs=True
        )

        self._assert_close(mll_ref, mll_scan)
        self._assert_close(filt_ref, filt_scan)
        self._assert_close(pred_ref, pred_scan)
        self._assert_close(smooth_ref, smooth_scan)
        self._assert_close(xi_ref, xi_scan)
        self._assert_close(cond_ref, cond_scan)

    def test_row_col_equivalence(self):
        key = jr.PRNGKey(5)
        initial_probs, transition_mats, log_likelihoods = self._make_case(
            key, T=5, K=4, time_varying=False
        )

        B_mats = transition_mats.T

        (mll_row, filt_row, pred_row, smooth_row, xi_row, cond_row) = hmm_smoother_scan_rowstoch(
            initial_probs, transition_mats, log_likelihoods
        )
        (mll_col, filt_col, pred_col, smooth_col, xi_col, cond_col) = hmm_smoother_scan_colstoch(
            initial_probs, B_mats, log_likelihoods, return_trans_probs=True
        )

        self._assert_close(mll_row, mll_col)
        self._assert_close(filt_row, filt_col)
        self._assert_close(pred_row, pred_col)
        self._assert_close(smooth_row, smooth_col)
        self._assert_close(xi_row, xi_col)
        self._assert_close(cond_row, cond_col)

    def test_smoother_scan_colstoch_stability_shifted_ll(self):
        key = jr.PRNGKey(6)
        initial_probs, B_mats, log_likelihoods = self._make_case_col(
            key, T=100, K=100, time_varying=True
        )
        log_likelihoods = log_likelihoods - 100.0

        (mll_ref, filt_ref, pred_ref, smooth_ref, xi_ref, cond_ref) = _reference_smoother_col(
            initial_probs, B_mats, log_likelihoods
        )
        (mll_scan, filt_scan, pred_scan, smooth_scan, xi_scan, cond_scan) = hmm_smoother_scan_colstoch(
            initial_probs, B_mats, log_likelihoods, return_trans_probs=True
        )

        self._assert_close(mll_ref, mll_scan)
        self._assert_close(filt_ref, filt_scan)
        self._assert_close(pred_ref, pred_scan)
        self._assert_close(smooth_ref, smooth_scan)
        self._assert_close(xi_ref, xi_scan)
        self._assert_close(cond_ref, cond_scan)

    def test_vmap_filter_scan_row(self):
        key = jr.PRNGKey(7)
        batch_size = 3
        init, trans, ll = self._make_batch_row(key, batch_size, T=5, K=4, time_varying=True)

        vmap_out = vmap(hmm_filter_scan_rowstoch, in_axes=(0, 0, 0))(init, trans, ll)
        seq_out = [hmm_filter_scan_rowstoch(init[i], trans[i], ll[i]) for i in range(batch_size)]
        stacked = tuple(jnp.stack([o[j] for o in seq_out], axis=0) for j in range(len(vmap_out)))

        for a, b in zip(vmap_out, stacked):
            self._assert_close(a, b)

    def test_vmap_smoother_scan_row(self):
        key = jr.PRNGKey(8)
        batch_size = 3
        init, trans, ll = self._make_batch_row(key, batch_size, T=5, K=4, time_varying=True)

        vmap_out = vmap(hmm_smoother_scan_rowstoch, in_axes=(0, 0, 0))(init, trans, ll)
        seq_out = [hmm_smoother_scan_rowstoch(init[i], trans[i], ll[i]) for i in range(batch_size)]
        stacked = tuple(jnp.stack([o[j] for o in seq_out], axis=0) for j in range(len(vmap_out)))

        for a, b in zip(vmap_out, stacked):
            self._assert_close(a, b)

    def test_vmap_filter_scan_col(self):
        key = jr.PRNGKey(9)
        batch_size = 3
        init, trans, ll = self._make_batch_col(key, batch_size, T=5, K=4, time_varying=True)

        vmap_out = vmap(hmm_filter_scan_colstoch, in_axes=(0, 0, 0))(init, trans, ll)
        seq_out = [hmm_filter_scan_colstoch(init[i], trans[i], ll[i]) for i in range(batch_size)]
        stacked = tuple(jnp.stack([o[j] for o in seq_out], axis=0) for j in range(len(vmap_out)))

        for a, b in zip(vmap_out, stacked):
            self._assert_close(a, b)

    def test_vmap_smoother_scan_col(self):
        key = jr.PRNGKey(10)
        batch_size = 3
        init, trans, ll = self._make_batch_col(key, batch_size, T=5, K=4, time_varying=True)

        smoother_fn = lambda pi, b, lls: hmm_smoother_scan_colstoch(pi, b, lls, return_trans_probs=True)
        vmap_out = vmap(smoother_fn, in_axes=(0, 0, 0))(init, trans, ll)
        seq_out = [smoother_fn(init[i], trans[i], ll[i]) for i in range(batch_size)]
        stacked = tuple(jnp.stack([o[j] for o in seq_out], axis=0) for j in range(len(vmap_out)))

        for a, b in zip(vmap_out, stacked):
            self._assert_close(a, b)

    def test_gradients_finite_rowstoch_smoother(self):
        key = jr.PRNGKey(11)
        T, K = 24, 24
        k1, k2, k3 = jr.split(key, 3)
        pi_logits = jr.normal(k1, (K,))
        trans_logits = jr.normal(k2, (K, K))
        ll_logits = jr.normal(k3, (T, K))

        def loss_fn(pi_l, trans_l, ll_l):
            pi = nn.softmax(pi_l, axis=-1)
            trans = nn.softmax(trans_l, axis=-1)  # row-stochastic
            ll = nn.log_softmax(ll_l, axis=-1)
            mll, filt, pred, smooth, trans_probs, cond_probs = hmm_smoother_scan_rowstoch(pi, trans, ll)
            return (
                mll
                + 1e-2 * (filt**2).sum()
                + 1e-2 * (pred**2).sum()
                + 1e-2 * (smooth**2).sum()
                + 1e-2 * (trans_probs**2).sum()
                + 1e-2 * (cond_probs**2).sum()
            )

        pi = nn.softmax(pi_logits, axis=-1)
        trans = nn.softmax(trans_logits, axis=-1)
        ll = nn.log_softmax(ll_logits, axis=-1)
        outputs = hmm_smoother_scan_rowstoch(pi, trans, ll)
        self._assert_all_finite_tree(outputs)

        loss = loss_fn(pi_logits, trans_logits, ll_logits)
        grads = grad(loss_fn, argnums=(0, 1, 2))(pi_logits, trans_logits, ll_logits)

        self.assertTrue(bool(jnp.isfinite(loss)))
        self._assert_all_finite_tree(grads)

    def test_gradients_finite_colstoch_smoother(self):
        key = jr.PRNGKey(12)
        T, K = 24, 24
        k1, k2, k3 = jr.split(key, 3)
        pi_logits = jr.normal(k1, (K,))
        B_logits = jr.normal(k2, (K, K))
        ll_logits = jr.normal(k3, (T, K))

        def loss_fn(pi_l, B_l, ll_l):
            pi = nn.softmax(pi_l, axis=-1)
            B = nn.softmax(B_l, axis=0)  # col-stochastic
            ll = nn.log_softmax(ll_l, axis=-1)
            mll, filt, pred, smooth, trans_probs, cond_probs = hmm_smoother_scan_colstoch(
                pi, B, ll, return_trans_probs=True
            )
            return (
                mll
                + 1e-2 * (filt**2).sum()
                + 1e-2 * (pred**2).sum()
                + 1e-2 * (smooth**2).sum()
                + 1e-2 * (trans_probs**2).sum()
                + 1e-2 * (cond_probs**2).sum()
            )

        pi = nn.softmax(pi_logits, axis=-1)
        B = nn.softmax(B_logits, axis=0)
        ll = nn.log_softmax(ll_logits, axis=-1)
        outputs = hmm_smoother_scan_colstoch(pi, B, ll, return_trans_probs=True)
        self._assert_all_finite_tree(outputs)

        loss = loss_fn(pi_logits, B_logits, ll_logits)
        grads = grad(loss_fn, argnums=(0, 1, 2))(pi_logits, B_logits, ll_logits)

        self.assertTrue(bool(jnp.isfinite(loss)))
        self._assert_all_finite_tree(grads)

    def test_strict_zero_absorbing_transitions_finite_outputs_and_grads(self):
        key = jr.PRNGKey(14)
        T, K = 40, 14
        ll_logits = jr.normal(key, (T, K)) * 3.0 - 80.0
        ll = nn.log_softmax(ll_logits, axis=-1)

        # Deterministic initial state includes strict zeros.
        pi = jnp.zeros((K,)).at[0].set(1.0)
        A = _sparse_absorbing_rowstoch(K)
        B = A.T

        self.assertTrue(bool(jnp.any(A == 0.0)))
        self.assertTrue(bool(jnp.isclose(A[-1, -1], 1.0)))

        row_outputs = hmm_smoother_scan_rowstoch(pi, A, ll)
        col_outputs = hmm_smoother_scan_colstoch(pi, B, ll, return_trans_probs=True)
        self._assert_all_finite_tree(row_outputs)
        self._assert_all_finite_tree(col_outputs)

        def row_loss(ll_l):
            ll_local = nn.log_softmax(ll_l, axis=-1)
            mll, filt, pred, smooth, trans_probs, cond_probs = hmm_smoother_scan_rowstoch(pi, A, ll_local)
            return (
                mll
                + 1e-2 * (filt**2).sum()
                + 1e-2 * (pred**2).sum()
                + 1e-2 * (smooth**2).sum()
                + 1e-2 * (trans_probs**2).sum()
                + 1e-2 * (cond_probs**2).sum()
            )

        def col_loss(ll_l):
            ll_local = nn.log_softmax(ll_l, axis=-1)
            mll, filt, pred, smooth, trans_probs, cond_probs = hmm_smoother_scan_colstoch(
                pi, B, ll_local, return_trans_probs=True
            )
            return (
                mll
                + 1e-2 * (filt**2).sum()
                + 1e-2 * (pred**2).sum()
                + 1e-2 * (smooth**2).sum()
                + 1e-2 * (trans_probs**2).sum()
                + 1e-2 * (cond_probs**2).sum()
            )

        row_grad = grad(row_loss)(ll_logits)
        col_grad = grad(col_loss)(ll_logits)
        self.assertTrue(bool(jnp.all(jnp.isfinite(row_grad))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(col_grad))))

    def test_near_zero_transitions_finite_outputs_and_grads(self):
        key = jr.PRNGKey(15)
        T, K = 24, 20
        k1, k2 = jr.split(key, 2)
        ll_logits = jr.normal(k1, (T, K)) * 4.0 - 120.0
        ll = nn.log_softmax(ll_logits, axis=-1)
        pi = nn.softmax(jr.normal(k2, (K,)), axis=-1)
        A = _near_zero_rowstoch(K, tiny=1e-20)
        B = A.T

        # Near-zero (but nonzero) off-diagonals
        self.assertTrue(bool(jnp.all(A > 0.0)))
        self.assertTrue(bool(jnp.min(A) < 1e-10))

        row_outputs = hmm_smoother_scan_rowstoch(pi, A, ll)
        col_outputs = hmm_smoother_scan_colstoch(pi, B, ll, return_trans_probs=True)
        self._assert_all_finite_tree(row_outputs)
        self._assert_all_finite_tree(col_outputs)

        def row_loss(ll_l):
            ll_local = nn.log_softmax(ll_l, axis=-1)
            mll, filt, pred, smooth, trans_probs, cond_probs = hmm_smoother_scan_rowstoch(pi, A, ll_local)
            return (
                mll
                + 1e-2 * (filt**2).sum()
                + 1e-2 * (pred**2).sum()
                + 1e-2 * (smooth**2).sum()
                + 1e-2 * (trans_probs**2).sum()
                + 1e-2 * (cond_probs**2).sum()
            )

        def col_loss(ll_l):
            ll_local = nn.log_softmax(ll_l, axis=-1)
            mll, filt, pred, smooth, trans_probs, cond_probs = hmm_smoother_scan_colstoch(
                pi, B, ll_local, return_trans_probs=True
            )
            return (
                mll
                + 1e-2 * (filt**2).sum()
                + 1e-2 * (pred**2).sum()
                + 1e-2 * (smooth**2).sum()
                + 1e-2 * (trans_probs**2).sum()
                + 1e-2 * (cond_probs**2).sum()
            )

        row_grad = grad(row_loss)(ll_logits)
        col_grad = grad(col_loss)(ll_logits)
        self.assertTrue(bool(jnp.all(jnp.isfinite(row_grad))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(col_grad))))

    def test_directional_derivative_rowstoch_filter_mll(self):
        key = jr.PRNGKey(13)
        T, K = 12, 8
        n_pi = K
        n_trans = K * K
        n_ll = T * K
        n_total = n_pi + n_trans + n_ll
        theta0 = jr.normal(key, (n_total,))

        def scalar_mll(theta):
            pi_l = theta[:n_pi]
            trans_l = theta[n_pi:n_pi + n_trans].reshape(K, K)
            ll_l = theta[n_pi + n_trans:].reshape(T, K)

            pi = nn.softmax(pi_l, axis=-1)
            trans = nn.softmax(trans_l, axis=-1)  # row-stochastic
            ll = nn.log_softmax(ll_l, axis=-1)

            mll, _, _ = hmm_filter_scan_rowstoch(pi, trans, ll)
            return mll

        g = grad(scalar_mll)(theta0)
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

        # Check sampled directional derivatives against central finite differences.
        eps = 1e-3
        n_dirs = 3
        dir_keys = jr.split(jr.PRNGKey(101), n_dirs)
        for dkey in dir_keys:
            v = jr.normal(dkey, theta0.shape)
            v = v / (jnp.linalg.norm(v) + 1e-12)
            ad_dir = jnp.vdot(g, v)
            fd_dir = (scalar_mll(theta0 + eps * v) - scalar_mll(theta0 - eps * v)) / (2.0 * eps)
            self.assertTrue(bool(jnp.isfinite(ad_dir)))
            self.assertTrue(bool(jnp.isfinite(fd_dir)))
            self.assertTrue(bool(jnp.allclose(ad_dir, fd_dir, atol=5e-3, rtol=5e-2)))

    def test_update_posterior_states_exact_inference_matches_wrapper(self):
        key = jr.PRNGKey(202)
        k1, k2, k3, k4 = jr.split(key, 4)
        T, K, no, nu = 7, 6, 5, 3

        A = utils.random_A_array(k1, [no], [K], A_dependencies=[[0]])
        B = utils.random_B_array(k2, [K], [nu], B_dependencies=[[0]])
        prior = [jr.dirichlet(k3, jnp.ones((K,)))]

        obs_idx = jr.randint(k4, (T,), 0, no)
        obs = [nn.one_hot(obs_idx, no)]
        actions = jr.randint(jr.fold_in(k4, 1), (T - 1,), 0, nu)
        ll = vmap(lambda obs_t: jnp.log(A[0][obs_t]))(obs_idx)
        B_seq = vmap(lambda u: B[0][..., u])(actions)
        _, filtered_ref, _ = hmm_filter_scan_colstoch(prior[0], B_seq, ll)

        prior_t = prior
        for t in range(T):
            obs_t = [obs[0][t : t + 1]]

            qs_exact = inference.update_posterior_states(
                A=A,
                B=B,
                obs=obs_t,
                past_actions=None,
                prior=prior_t,
                A_dependencies=[[0]],
                B_dependencies=[[0]],
                method="exact",
                distr_obs=True,
            )
            self._assert_close(qs_exact[0][-1], filtered_ref[t])

            if t < T - 1:
                prior_t = [B[0][..., actions[t]] @ qs_exact[0][-1]]

    def test_exact_inference_respects_inference_horizon(self):
        key = jr.PRNGKey(203)
        k1, k2, k3, k4 = jr.split(key, 4)
        T, K, no, nu = 8, 5, 4, 2
        horizon = 4

        A = utils.random_A_array(k1, [no], [K], A_dependencies=[[0]])
        B = utils.random_B_array(k2, [K], [nu], B_dependencies=[[0]])
        prior = [jr.dirichlet(k3, jnp.ones((K,)))]

        obs_idx = jr.randint(k4, (T,), 0, no)
        obs = [nn.one_hot(obs_idx, no)]
        actions = jr.randint(jr.fold_in(k4, 1), (T - 1,), 0, nu)

        ll = vmap(lambda obs_t: jnp.log(A[0][obs_t]))(obs_idx)
        B_seq = vmap(lambda u: B[0][..., u])(actions)
        _, filtered_ref, _ = hmm_filter_scan_colstoch(prior[0], B_seq, ll)

        prior_t = prior
        qs_hist = None
        for t in range(T):
            obs_t = [obs[0][t : t + 1]]
            qs_hist = inference.update_posterior_states(
                A=A,
                B=B,
                obs=obs_t,
                past_actions=None,
                prior=prior_t,
                qs_hist=qs_hist,
                A_dependencies=[[0]],
                B_dependencies=[[0]],
                method="exact",
                distr_obs=True,
                inference_horizon=horizon,
            )
            expected = filtered_ref[max(0, t - horizon + 1) : t + 1]
            self.assertEqual(qs_hist[0].shape[0], expected.shape[0])
            self._assert_close(qs_hist[0], expected)

            if t < T - 1:
                prior_t = [B[0][..., actions[t]] @ qs_hist[0][-1]]

    def test_exact_inference_loop_with_policy_inference_and_learning(self):
        key = jr.PRNGKey(204)
        k1, k2, k3, k4 = jr.split(key, 4)
        T, K, no, nu = 6, 5, 4, 3
        horizon = 4

        A = utils.random_A_array(k1, [no], [K], A_dependencies=[[0]])
        B = utils.random_B_array(k2, [K], [nu], B_dependencies=[[0]])
        pA = [jnp.ones_like(A[0])]
        pB = [jnp.ones_like(B[0])]
        D = [jr.dirichlet(k3, jnp.ones((K,)))]

        agent = Agent(
            A=A,
            B=B,
            D=D,
            pA=pA,
            pB=pB,
            A_dependencies=[[0]],
            B_dependencies=[[0]],
            num_controls=[nu],
            inference_algo="exact",
            inference_horizon=horizon,
            policy_len=1,
            learn_A=True,
            learn_B=True,
            batch_size=1,
        )

        obs_idx = jr.randint(k4, (T,), 0, no)
        action_hist = jr.randint(jr.fold_in(k4, 1), (1, T - 1, 1), 0, nu)

        empirical_prior = agent.D
        qs_hist = None
        for t in range(T):
            obs_t = [jnp.array([obs_idx[t]])]
            qs_hist = agent.infer_states(
                observations=obs_t,
                empirical_prior=empirical_prior,
                qs_hist=qs_hist,
            )

            expected_len = min(t + 1, horizon)
            self.assertEqual(qs_hist[0].shape, (1, expected_len, K))

            if t < T - 1:
                action_t = action_hist[:, t, :]
                empirical_prior = agent.update_empirical_prior(action_t, qs_hist)

        q_pi, G = agent.infer_policies(qs_hist)
        self.assertTrue(bool(jnp.allclose(q_pi.sum(-1), 1.0, atol=1e-5)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(G))))

        prior_next = agent.update_empirical_prior(action_hist[:, -1, :], qs_hist)
        self.assertEqual(prior_next[0].shape, (1, K))
        self.assertTrue(bool(jnp.allclose(prior_next[0].sum(-1), 1.0, atol=1e-5)))

        outcomes_window = [jnp.expand_dims(obs_idx[-horizon:], 0)]
        actions_window = action_hist[:, -(horizon - 1):, :]
        pA_before = agent.pA[0]
        pB_before = agent.pB[0]
        agent_updated = agent.infer_parameters(qs_hist, outcomes_window, actions_window)

        self.assertFalse(bool(jnp.allclose(agent_updated.pA[0], pA_before)))
        self.assertFalse(bool(jnp.allclose(agent_updated.pB[0], pB_before)))

    def test_exact_inference_accepts_missing_past_actions(self):
        key = jr.PRNGKey(205)
        k1, k2, k3, k4 = jr.split(key, 4)
        T, K, no, nu = 5, 4, 3, 2

        A = utils.random_A_array(k1, [no], [K], A_dependencies=[[0]])
        B = utils.random_B_array(k2, [K], [nu], B_dependencies=[[0]])
        D = [jr.dirichlet(k3, jnp.ones((K,)))]

        agent = Agent(
            A=A,
            B=B,
            D=D,
            A_dependencies=[[0]],
            B_dependencies=[[0]],
            num_controls=[nu],
            inference_algo="exact",
            inference_horizon=T,
            batch_size=1,
        )

        obs_idx = jr.randint(k4, (T,), 0, no)
        observations = [jnp.expand_dims(obs_idx, 0)]  # (batch, T)

        qs = agent.infer_states(
            observations=observations,
            empirical_prior=agent.D,
            past_actions=None,
        )
        self.assertEqual(qs[0].shape, (1, 1, K))
        self.assertTrue(bool(jnp.allclose(qs[0].sum(-1), 1.0, atol=1e-5)))

    def test_exact_inference_rejects_multi_factor_models(self):
        key = jr.PRNGKey(206)
        k1, k2, k3 = jr.split(key, 3)
        no = 4
        num_states = [3, 2]
        num_controls = [2, 2]

        A = utils.random_A_array(k1, [no], num_states, A_dependencies=[[0, 1]])
        B = utils.random_B_array(k2, num_states, num_controls, B_dependencies=[[0], [1]])
        D = [
            jr.dirichlet(jr.fold_in(k3, 0), jnp.ones((num_states[0],))),
            jr.dirichlet(jr.fold_in(k3, 1), jnp.ones((num_states[1],))),
        ]

        with self.assertRaisesRegex(ValueError, "single hidden-state factor"):
            _ = Agent(
                A=A,
                B=B,
                D=D,
                A_dependencies=[[0, 1]],
                B_dependencies=[[0], [1]],
                num_controls=num_controls,
                inference_algo="exact",
                batch_size=1,
            )


if __name__ == "__main__":
    unittest.main()
