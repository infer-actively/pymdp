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

from mutual_information import _mi_from_pA
from hierarchical import _bottom_up_pass, _top_down_refinement_pass


# ---------------------------------------------------------------------------
# MI-gated Dirichlet update and interleaved EM
# ---------------------------------------------------------------------------


def _mi_gated_update(
    agent_prior: Agent,
    agent_posterior: Agent,
    beta: float = 512.0,
    eta: float = 512.0,
) -> Agent:
    """Apply MI-gated asymptotic Dirichlet update, following SPM's spm_MDP_VB_XXX.

    Computes MI for the prior and posterior Dirichlet parameters and gates the
    update with a softmax over the two MI values:

        Pa = softmax(beta * [MI(pa), MI(qa)])
        pA_new = (Pa[0]*pa + Pa[1]*qa) * eta / (eta + Pa[1])

    With beta=512 the gate is near-binary: the posterior is accepted only if
    MI(qa) > MI(pa), i.e., the new example measurably increases the mutual
    information of the learned A matrix. The eta term prevents unbounded
    accumulation (asymptotic forgetting).

    Args:
        agent_prior: Agent before Dirichlet update (holds pa = prior pA)
        agent_posterior: Agent returned by infer_parameters (holds qa = posterior pA)
        beta: softmax sharpness (SPM uses 512)
        eta: asymptote / forgetting parameter (SPM uses 512)

    Returns:
        Agent with gated pA and recomputed A
    """
    pa = agent_prior.pA
    qa = agent_posterior.pA

    mi_pa = _mi_from_pA(pa)
    mi_qa = _mi_from_pA(qa)

    # Softmax gate — keep Pa as a traced array so _mi_gated_update is JIT-safe
    Pa = jax.nn.softmax(beta * jnp.stack([mi_pa, mi_qa]))

    # Gated asymptotic blend
    scale = eta / (eta + Pa[1])
    pA_new = [(Pa[0] * pa_m + Pa[1] * qa_m) * scale for pa_m, qa_m in zip(pa, qa)]

    # Recompute A: normalize along obs axis (axis 1)
    A_new = [pa_m / pa_m.sum(axis=1, keepdims=True) for pa_m in pA_new]

    return eqx.tree_at(lambda x: (x.A, x.pA), agent_posterior, (A_new, pA_new))


def _interleaved_em(
    pA_list: list[jnp.ndarray],
    obs_soft_list: list[jnp.ndarray],
    D: jnp.ndarray,
    valid_mask: jnp.ndarray,
    num_iter: int = 16,
) -> tuple[list[jnp.ndarray], jnp.ndarray]:
    """Interleaved Q-A EM matching MATLAB's spm_VBX within-level solver.

    Each iteration resets qa to the prior then adds the current sufficient
    statistic, mirroring MATLAB's `qa = pa; qa += cross(O, Q)` pattern:

        M-step: qa_m = pa_m + einsum('po,ps->pos', obs_m, Q)
        A_m    = normalize(qa_m)                               (spm_norm)
        E-step: Q = softmax(sum_m log(A_m) @ obs_m + log(D))

    Only Q evolves across iterations; pa is the fixed Dirichlet prior.
    Q is initialised to D (the state prior) before the first M-step.

    Uses lax.scan over iterations (carry = Q only) to keep XLA graph size O(1)
    regardless of num_iter — avoiding command-buffer OOM from loop unrolling.

    Args:
        pA_list:       list of (n_patches, n_obs_m, max_states) Dirichlet priors
        obs_soft_list: list of (n_patches, n_obs_m) soft observation vectors
        D:             (n_patches, max_states) state prior
        valid_mask:    (n_patches, max_states) bool — zero out invalid patches
        num_iter:      number of EM iterations (MATLAB uses 16)

    Returns:
        (qa_list, Q): final Dirichlet accumulators and state posterior
    """
    log_D = jnp.log(jnp.clip(D, 1e-16))

    def em_step(Q, _):
        # M-step: qa_m = (pa_m + outer(obs_m, Q)) * (pa_m > 0)
        # Reset to pa each iteration; mask zeros so unobserved (obs, state) pairs
        # are never activated — matches spm_backwards: qa = qa .* (pa > 0)
        qa_list_inner = [
            (pa_m + jnp.einsum('po,ps->pos', obs_m, Q)) * (pa_m > 0)
            for pa_m, obs_m in zip(pA_list, obs_soft_list)
        ]
        A_list_inner = [
            qa_m / jnp.clip(qa_m.sum(axis=1, keepdims=True), 1e-16)
            for qa_m in qa_list_inner
        ]
        # E-step: Q = softmax(sum_m log(A_m) @ obs_m + log(D))
        log_q = log_D + sum(
            jnp.einsum('pos,po->ps', jnp.log(jnp.clip(A_m, 1e-16)), obs_m)
            for A_m, obs_m in zip(A_list_inner, obs_soft_list)
        )
        Q_new = jax.nn.softmax(log_q, axis=-1)
        Q_new = jnp.where(valid_mask, Q_new, 0.0)
        Q_new = Q_new / jnp.clip(Q_new.sum(axis=-1, keepdims=True), 1e-16)
        return Q_new, None

    Q, _ = lax.scan(em_step, D, None, length=num_iter)

    # Final M-step to recover qa_list at the converged Q (with mask)
    qa_list = [
        (pa_m + jnp.einsum('po,ps->pos', obs_m, Q)) * (pa_m > 0)
        for pa_m, obs_m in zip(pA_list, obs_soft_list)
    ]
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
    refined_soft = _top_down_refinement_pass(
        levels, valid_masks, cls_agent, q_cls, level_soft_beliefs, level_obs_lists
    )

    # --- Interleaved EM updates at every hierarchical level (matching spm_VBX) ---
    new_levels = []
    for lv_idx in range(n_hier):
        level = levels[lv_idx]
        D_em = refined_soft[lv_idx]   # top-down refined prior over states
        if lv_idx == 0:
            n_obs_l1 = level.pA[0].shape[1]  # n_levels (SVD bin outcomes)
            obs_soft = [jax.nn.one_hot(o, n_obs_l1) for o in level_obs_lists[0]]
        else:
            obs_soft = level_obs_lists[lv_idx]  # already (n_patches, max_child_states)
        qa_final, _ = _interleaved_em(level.pA, obs_soft, D_em, valid_masks[lv_idx])
        A_new_em = [qa_m / jnp.clip(qa_m.sum(axis=1, keepdims=True), 1e-16) for qa_m in qa_final]
        agent_posterior = eqx.tree_at(lambda x: (x.pA, x.A), level, (qa_final, A_new_em))
        new_levels.append(_mi_gated_update(level, agent_posterior, beta, eta))

    # --- Classification level: interleaved EM ---
    obs_soft_cls = [child_beliefs]   # (1, n_top_states)
    D_cls = q_cls                    # (1, n_classes) supervised posterior
    qa_cls_final, _ = _interleaved_em(
        cls_agent.pA, obs_soft_cls, D_cls,
        jnp.ones((1, n_classes), dtype=jnp.bool_),
    )
    A_cls_new = [qa_m / jnp.clip(qa_m.sum(axis=1, keepdims=True), 1e-16) for qa_m in qa_cls_final]
    cls_posterior = eqx.tree_at(lambda x: (x.pA, x.A), cls_agent, (qa_cls_final, A_cls_new))
    new_cls_agent = _mi_gated_update(cls_agent, cls_posterior, beta, eta)

    new_carry = _TrainCarry(levels=tuple(new_levels), cls_agent=new_cls_agent)
    return new_carry, {"correct_inc": correct_inc}
