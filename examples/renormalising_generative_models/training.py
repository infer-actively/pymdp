"""MI-gated Dirichlet training loop for RGM hierarchies.

Implements the sequential interleaved-EM training matching SPM's
spm_VBX / spm_MDP_VB_XXX within-level solver and the DEM_MNIST_RGM.m
outer loop.  All functions here are JAX-JIT-safe and designed to be
used as lax.scan bodies.
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
from typing import NamedTuple

import equinox as eqx

from pymdp.agent import Agent

from mutual_information import _mi_per_mapping_from_pA
from hierarchical import (
    _bottom_up_pass,
    _top_down_refinement_pass,
    _extract_soft_obs,
)


# ---------------------------------------------------------------------------
# MI-gated Dirichlet update and interleaved EM
# ---------------------------------------------------------------------------


def _mi_gated_update(
    agent_prior: Agent,
    agent_posterior: Agent,
    beta: float = 512.0,
    eta: float = 512.0,
    valid_mask: jnp.ndarray | None = None,
) -> Agent:
    """Apply MI-gated asymptotic Dirichlet update, following SPM's spm_MDP_VB_XXX.

    Gates the update per likelihood mapping — each (patch, modality) slice is
    gated on its *own* MI change, not once across the whole level (SPM applies
    the gate to each mapping independently). For every mapping:

        Pa = softmax(beta * [MI(pa), MI(qa)])            # per (patch, modality)
        pA_new = (Pa[0]*pa + Pa[1]*qa) * eta / (eta + Pa[1])

    With beta=512 the gate is near-binary: a mapping's posterior is accepted only
    if MI(qa) > MI(pa) for that mapping, i.e., the new example measurably
    increases its mutual information. The eta term bounds accumulation
    (asymptotic forgetting). Gating per mapping matters at beta=512 because a
    single scalar level-wide gate lets one high-MI patch veto learning at all the
    others (or vice-versa).

    Args:
        agent_prior: Agent before Dirichlet update (holds pa = prior pA)
        agent_posterior: Agent returned by infer_parameters (holds qa = posterior pA)
        beta: softmax sharpness (SPM uses 512)
        eta: asymptote / forgetting parameter (SPM uses 512)
        valid_mask: optional (n_patches, max_states) bool mask marking real
            (non-padded) states. The batched pA pads every mapping's state
            axis to the level's max state count with a nonzero (uniform)
            likelihood; without this mask those phantom states would inflate
            the MI used to gate learning.

    Returns:
        Agent with gated pA and recomputed A
    """
    pa = agent_prior.pA
    qa = agent_posterior.pA

    mi_pa_list = _mi_per_mapping_from_pA(pa, valid_mask)  # list of (n_patches,)
    mi_qa_list = _mi_per_mapping_from_pA(qa, valid_mask)  # list of (n_patches,)

    pA_new = []
    for pa_m, qa_m, mi_pa_m, mi_qa_m in zip(pa, qa, mi_pa_list, mi_qa_list):
        # Per-patch softmax gate for this modality — (n_patches, 2), traced so
        # _mi_gated_update stays JIT-safe.
        Pa = jax.nn.softmax(
            beta * jnp.stack([mi_pa_m, mi_qa_m], axis=-1), axis=-1
        )  # (n_patches, 2)
        Pa0 = Pa[:, 0][:, None, None]  # (n_patches, 1, 1)
        Pa1 = Pa[:, 1][:, None, None]
        scale = eta / (eta + Pa1)      # (n_patches, 1, 1) asymptotic forgetting
        pA_new.append((Pa0 * pa_m + Pa1 * qa_m) * scale)

    # Recompute A: normalize along obs axis (axis 1)
    A_new = [pa_m / pa_m.sum(axis=1, keepdims=True) for pa_m in pA_new]

    return eqx.tree_at(lambda x: (x.A, x.pA), agent_posterior, (A_new, pA_new))


def _interleaved_em(
    pA_list: list[jnp.ndarray],
    obs_soft_list: list[jnp.ndarray],
    D: jnp.ndarray,
    valid_mask: jnp.ndarray,
    num_iter: int = 1,
) -> tuple[list[jnp.ndarray], jnp.ndarray]:
    """Interleaved Q-A EM matching MATLAB's spm_VBX within-level solver.

    Each iteration derives A from the *previous* qa, then resets qa to the
    prior and adds the current sufficient statistic, mirroring MATLAB's
    ``qa = pa; qa += cross(O, Q)`` pattern:

        A_m    = normalize(qa_m)             (spm_norm; qa_m == pa_m on iter 1)
        E-step: Q = softmax(sum_m log(A_m @ obs_m) + log(D))
        M-step: qa_m = (pa_m + einsum('po,ps->pos', obs_m, Q)) * (pa_m > 0)

    On iteration 1, qa == pa, so A is the fixed prior-normalised likelihood —
    this sample has not yet been folded in. ``num_iter`` defaults to 1: SPM's
    DEM_MNIST_RGM demo runs ``spm_VBX`` with ``OPTIONS.B = 0``, i.e. a single
    non-iterative belief-propagation pass under that prior A, followed by one
    Dirichlet accumulation. The 16-iteration scheme (deriving A afresh from
    each iteration's own qa, which is self-reinforcing) lives in
    ``spm_backwards`` and is only entered when ``OPTIONS.B = 1`` (not used by
    the MNIST demo); larger ``num_iter`` values reproduce that scheme for
    experimentation.

    The E-step uses SPM's soft-likelihood form — the log outside the
    expectation, ``log(sum_o A(o|s) q(o))`` — rather than
    ``sum_o q(o) log A(o|s)``. The two agree only when q(o) is one-hot; the
    upper hierarchical levels pass soft (non-one-hot) beliefs as observations,
    so the placement of the log is a substantive difference, not notation.

    Uses lax.scan over iterations (carry = qa_list) to keep XLA graph size
    O(1) regardless of num_iter — avoiding command-buffer OOM from loop
    unrolling.

    Args:
        pA_list:       list of (n_patches, n_obs_m, max_states) Dirichlet priors
        obs_soft_list: list of (n_patches, n_obs_m) soft observation vectors
        D:             (n_patches, max_states) state prior
        valid_mask:    (n_patches, max_states) bool — zero out invalid patches
        num_iter:      number of EM iterations (SPM MNIST demo uses 1; the
                       16-iteration spm_backwards path requires OPTIONS.B=1)

    Returns:
        (qa_list, Q): final Dirichlet accumulators and state posterior
    """
    log_D = jnp.log(jnp.clip(D, 1e-16))

    def em_step(qa_list, _):
        A_list = [
            qa_m / jnp.clip(qa_m.sum(axis=1, keepdims=True), 1e-16)
            for qa_m in qa_list
        ]
        # E-step: Q = softmax(sum_m log(A_m @ obs_m) + log(D)) — log outside
        # the expectation, matching spm_VBX's soft-likelihood form.
        log_q = log_D + sum(
            jnp.log(jnp.clip(jnp.einsum('pos,po->ps', A_m, obs_m), 1e-16))
            for A_m, obs_m in zip(A_list, obs_soft_list)
        )
        Q_new = jax.nn.softmax(log_q, axis=-1)
        Q_new = jnp.where(valid_mask, Q_new, 0.0)
        Q_new = Q_new / jnp.clip(Q_new.sum(axis=-1, keepdims=True), 1e-16)
        # M-step: reset to pa, add this sample's cross term once. Mask zeros
        # so unobserved (obs, state) pairs are never activated — matches
        # spm_backwards: qa = qa .* (pa > 0)
        qa_new = [
            (pa_m + jnp.einsum('po,ps->pos', obs_m, Q_new)) * (pa_m > 0)
            for pa_m, obs_m in zip(pA_list, obs_soft_list)
        ]
        return qa_new, Q_new

    qa_list, Q_seq = lax.scan(em_step, pA_list, None, length=num_iter)
    Q = Q_seq[-1]
    return qa_list, Q


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class _TrainCarry(NamedTuple):
    """Immutable carry for the lax.scan training loop.

    Holds only the JAX-mutable state: the per-level agents (as pytrees) and
    the classification agent.  valid_masks, stats, and other structural data
    are captured as static closures and never change during training.
    """
    levels: tuple   # tuple[Agent, ...] — hierarchical levels only (no cls)
    cls_agent: Agent


def _train_step(
    carry: _TrainCarry,
    x: tuple,
    valid_masks: tuple,
    n_classes: int,
    beta: float,
    eta: float,
    num_iter: int = 1,
) -> tuple[_TrainCarry, dict]:
    """Pure per-image training step — suitable as a lax.scan body.

    Performs one full bottom-up / supervised-classification / top-down /
    interleaved-EM-update cycle for a single pre-encoded image.

    Args:
        carry: current training state (hierarchical + cls agents)
        x: (obs_image, label) where obs_image is (n_i, n_j, n_mod) and
           label is a scalar int32 JAX array
        valid_masks: tuple of (n_patches, max_states) bool masks, one per
                     hierarchical level (static, closed over)
        n_classes: number of digit classes (static)
        beta: MI-gate softmax sharpness
        eta: asymptotic forgetting parameter
        num_iter: interleaved-EM iterations per level (SPM parity = 1)

    Returns:
        (new_carry, metrics) where metrics = {"correct_inc": int32 scalar}
    """
    obs_image, label = x
    levels = carry.levels
    cls_agent = carry.cls_agent
    n_hier = len(levels)  # static at trace time

    # --- Bottom-up soft pass ---
    level_soft_beliefs, level_obs_lists = _bottom_up_pass(levels, valid_masks, obs_image)
    child_beliefs = level_soft_beliefs[-1]

    # --- Unsupervised prediction (for accuracy tracking) ---
    qs_cls_pred = cls_agent.infer_states([child_beliefs], cls_agent.D)
    pred = jnp.argmax(qs_cls_pred[0][:, 0, :][0])
    correct_inc = (pred == label).astype(jnp.int32)

    # --- Supervised classification with one-hot D prior ---
    supervised_D = [jax.nn.one_hot(label, n_classes)[None, :]]
    qs_cls = cls_agent.infer_states([child_beliefs], supervised_D)
    q_cls = qs_cls[0][:, 0, :]  # (1, n_classes)

    # --- Top-down pass: refine beliefs using cls posterior ---
    # td_D holds the label-conditioned top-down message that entered each level
    # *before* that level's observations were folded in (spm_dot(A, Q_parent)).
    refined_soft, td_D = _top_down_refinement_pass(
        levels, valid_masks, cls_agent, q_cls, level_soft_beliefs, level_obs_lists
    )

    # --- Interleaved EM updates at every hierarchical level (matching spm_VBX) ---
    new_levels = []
    for lv_idx in range(n_hier):
        level = levels[lv_idx]
        # Learning prior = top-down message alone, so this level's observations
        # enter the M-step exactly once (using refined_soft here would double
        # count them — it has already absorbed those observations).
        D_em = td_D[lv_idx]
        if lv_idx == 0:
            # L1 outcomes are the actual SVD bin observations (data, not beliefs),
            # so they are label-independent and used as-is.
            n_obs_l1 = level.pA[0].shape[1]  # n_levels (SVD bin outcomes)
            obs_soft = [jax.nn.one_hot(o, n_obs_l1) for o in level_obs_lists[0]]
        else:
            # Outcome side = refined (label-conditioned) child beliefs. In SPM the
            # supervised one-hot D conditions the whole downward sweep, so the
            # child posteriors that come back up as outcomes O are label-conditioned
            # by the time qa += cross(O, Q) runs.
            child_refined = refined_soft[lv_idx - 1]
            child_grid = int(round(child_refined.shape[0] ** 0.5))
            obs_soft = _extract_soft_obs(child_refined, child_grid)
        qa_final, _ = _interleaved_em(
            level.pA, obs_soft, D_em, valid_masks[lv_idx], num_iter=num_iter
        )
        A_new_em = [qa_m / jnp.clip(qa_m.sum(axis=1, keepdims=True), 1e-16) for qa_m in qa_final]
        agent_posterior = eqx.tree_at(lambda x: (x.pA, x.A), level, (qa_final, A_new_em))
        new_levels.append(
            _mi_gated_update(level, agent_posterior, beta, eta, valid_masks[lv_idx])
        )

    # --- Classification level: interleaved EM ---
    # Outcome side = refined (label-conditioned) top belief; learning prior =
    # supervised one-hot label (the cls level's top-down message), so the top
    # belief evidence enters the M-step once rather than also seeding the prior.
    obs_soft_cls = [refined_soft[n_hier - 1]]   # (1, n_top_states)
    D_cls = supervised_D[0]                      # (1, n_classes) one-hot label
    qa_cls_final, _ = _interleaved_em(
        cls_agent.pA, obs_soft_cls, D_cls,
        jnp.ones((1, n_classes), dtype=jnp.bool_),
        num_iter=num_iter,
    )
    A_cls_new = [qa_m / jnp.clip(qa_m.sum(axis=1, keepdims=True), 1e-16) for qa_m in qa_cls_final]
    cls_posterior = eqx.tree_at(lambda x: (x.pA, x.A), cls_agent, (qa_cls_final, A_cls_new))
    new_cls_agent = _mi_gated_update(cls_agent, cls_posterior, beta, eta)

    new_carry = _TrainCarry(levels=tuple(new_levels), cls_agent=new_cls_agent)
    return new_carry, {"correct_inc": correct_inc}
