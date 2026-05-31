"""Hierarchical agents for Renormalised Generative Models.

Level 1: batched pymdp Agent (one per 2×2 group location) mapping joint
4-tile raw SVD observations to learned group states, matching MATLAB's
spm_MB_structure_learning. No argmax bottleneck between raw observations
and the first pooling level.

Levels 2+: generic hierarchical agents that pool 2x2 blocks of child
posterior beliefs into parent states. Inference passes soft probability
distributions between levels instead of MAP (argmax) states, preserving
uncertainty and avoiding the argmax bottleneck.

Classification level: maps top-level states to digit classes via a soft
likelihood learned from exemplar label counts.

The number of levels is derived automatically from the config:
    n_groups = image_size // group_size
    n_levels = log2(n_groups)
"""

import functools
import math

import jax
import jax.lax as lax
import jax.random as jr
import numpy as np
from jax import numpy as jnp
from typing import NamedTuple

import equinox as eqx

from pymdp.agent import Agent
from pymdp.control import Policies

from discretise import (
    DiscretiseConfig,
    OverlappingSVDBasis,
    compute_svd_basis_overlapping,
    encode_images_overlapping,
    decode_observations_overlapping,
)


# ---------------------------------------------------------------------------
# State statistics: unique state vocabularies from discrete observations
# ---------------------------------------------------------------------------


class StateStats(NamedTuple):
    num_states: jnp.ndarray    # (n_grid, n_grid) int — unique state count per location
    state_patterns: list       # [i][j] -> (num_states_ij, n_obs) unique observation tuples
    state_to_images: list      # [i][j][s] -> list of image indices mapping to state s
    child_num_states: list | None  # [i][j] -> (4,) num_states of each child (None for L1)


def compute_group_states(
    observations: jnp.ndarray,
    config: DiscretiseConfig,
) -> StateStats:
    """Find unique joint 4-tile observation patterns per 2×2 group.

    Matches MATLAB's spm_MB_structure_learning L1: groups the ng×ng SVD tiles
    into (ng//2)×(ng//2) parent locations, concatenates the 4 tile observation
    vectors per group into a single (4k,) integer tuple, and finds unique rows
    across N exemplars.  No argmax is taken — the vocabulary is built directly
    over raw observations.

    Args:
        observations: (N, ng, ng, k) integer bin indices from encode_images_overlapping
        config: discretisation config

    Returns:
        StateStats for the 2×2-grouped level:
          num_states: (ng//2, ng//2) unique state count per group location
          state_patterns: [i][j] → (n_states, 4k) unique joint obs tuples
          state_to_images: [i][j][s] → list of exemplar indices
          child_num_states: [i][j] → (4k,) n_levels per (tile, component) modality
    """
    obs_np = np.asarray(observations)
    N, ng, _, k = obs_np.shape
    ng2 = ng // 2
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]

    num_states = np.zeros((ng2, ng2), dtype=np.int32)
    state_patterns = [[None for _ in range(ng2)] for _ in range(ng2)]
    state_to_images = [[None for _ in range(ng2)] for _ in range(ng2)]
    child_num_states = [[None for _ in range(ng2)] for _ in range(ng2)]

    for i in range(ng2):
        for j in range(ng2):
            # Gather 4 child tiles into joint (N, 4k) observation tuples
            tiles = np.stack([
                obs_np[:, 2 * i + di, 2 * j + dj, :]
                for di, dj in offsets
            ], axis=1)  # (N, 4, k)
            joint = tiles.reshape(N, 4 * k)  # (N, 4k)
            unique, inverse = np.unique(joint, axis=0, return_inverse=True)
            num_states[i, j] = len(unique)
            state_patterns[i][j] = jnp.array(unique)
            state_to_images[i][j] = [
                list(np.where(inverse == s)[0]) for s in range(len(unique))
            ]
            child_num_states[i][j] = np.full(4 * k, config.n_levels)

    return StateStats(
        num_states=jnp.array(num_states),
        state_patterns=state_patterns,
        state_to_images=state_to_images,
        child_num_states=child_num_states,
    )


def compute_hierarchical_state_stats(
    child_map_states: np.ndarray,
    child_stats: StateStats,
) -> StateStats:
    """Find unique child-state tuples per 2x2 block of children.

    Each parent location pools a 2x2 block of child MAP states. The unique
    4-tuples across all structure images become the parent's state vocabulary.

    Args:
        child_map_states: (N, child_grid, child_grid) MAP state indices at child level
        child_stats: stats from the child level (for reading child num_states)

    Returns:
        StateStats for the parent level
    """
    child_map_np = np.asarray(child_map_states)
    _, child_grid, _ = child_map_np.shape
    n_parent = child_grid // 2

    child_ns = np.asarray(child_stats.num_states)

    num_states = np.zeros((n_parent, n_parent), dtype=np.int32)
    state_patterns = [[None for _ in range(n_parent)] for _ in range(n_parent)]
    state_to_images = [[None for _ in range(n_parent)] for _ in range(n_parent)]
    child_num_states = [[None for _ in range(n_parent)] for _ in range(n_parent)]

    for i in range(n_parent):
        for j in range(n_parent):
            # Gather 2x2 block of child states: top-left, top-right, bottom-left, bottom-right
            ci, cj = 2 * i, 2 * j
            block = np.stack([
                child_map_np[:, ci, cj],
                child_map_np[:, ci, cj + 1],
                child_map_np[:, ci + 1, cj],
                child_map_np[:, ci + 1, cj + 1],
            ], axis=1)  # (N, 4)

            unique, inverse = np.unique(block, axis=0, return_inverse=True)
            num_states[i, j] = len(unique)
            state_patterns[i][j] = jnp.array(unique)
            state_to_images[i][j] = [
                list(np.where(inverse == s)[0]) for s in range(len(unique))
            ]
            child_num_states[i][j] = np.array([
                child_ns[ci, cj],
                child_ns[ci, cj + 1],
                child_ns[ci + 1, cj],
                child_ns[ci + 1, cj + 1],
            ])

    return StateStats(
        num_states=jnp.array(num_states),
        state_patterns=state_patterns,
        state_to_images=state_to_images,
        child_num_states=child_num_states,
    )


class RGMLevel(NamedTuple):
    """One level of the RGM hierarchy."""

    agent: Agent
    valid_mask: jnp.ndarray
    stats: StateStats


# ---------------------------------------------------------------------------
# Shared agent construction and inference
# ---------------------------------------------------------------------------


def _build_agent(
    A: list[jnp.ndarray],
    n_patches: int,
    max_states: int,
    n_mod: int,
    valid_mask: jnp.ndarray,
    learn_A: bool = False,
    pA: list[jnp.ndarray] | None = None,
    categorical_obs: bool = False,
) -> Agent:
    """Build a batched pymdp Agent with identity transitions and uniform prior.

    Args:
        A: list of n_mod observation likelihood arrays, each (n_patches, n_obs, max_states)
        n_patches: batch size
        max_states: hidden state dimension
        n_mod: number of observation modalities
        valid_mask: (n_patches, max_states) boolean mask
        learn_A: if True, enable Dirichlet learning for A matrices
        pA: Dirichlet concentration parameters (required when learn_A=True)
        categorical_obs: if True, observations are probability vectors (n_obs,) rather
            than integer indices; required for passing soft beliefs between levels

    Returns:
        Configured Agent
    """
    B_eye = jnp.broadcast_to(
        jnp.eye(max_states)[None, :, :, None],
        (n_patches, max_states, max_states, 1),
    )

    D_vals = jnp.where(valid_mask, 1.0, 0.0)
    D_vals = D_vals / D_vals.sum(axis=1, keepdims=True)

    return Agent(
        A=A,
        B=[B_eye],
        D=[D_vals],
        pA=pA,
        A_dependencies=[[0]] * n_mod,
        B_dependencies=[[0]],
        policies=Policies(jnp.zeros((1, 1, 1), dtype=jnp.int32)),
        num_controls=[1],
        batch_size=n_patches,
        inference_algo="fpi",
        num_iter=16,
        learn_A=learn_A,
        categorical_obs=categorical_obs,
    )


def _infer_map_states(
    agent: Agent,
    valid_mask: jnp.ndarray,
    obs_list: list[jnp.ndarray],
    output_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Run inference and return MAP state indices.

    Args:
        agent: Batched Agent
        valid_mask: (n_patches, max_states) boolean mask
        obs_list: list of n_mod arrays, each (n_patches,) integer observations
        output_shape: shape to reshape the result into

    Returns:
        MAP state indices reshaped to output_shape
    """
    qs = agent.infer_states(obs_list, agent.D)
    beliefs = qs[0][:, 0, :]
    masked = jnp.where(valid_mask, beliefs, -jnp.inf)
    return jnp.argmax(masked, axis=1).reshape(output_shape)


def _infer_beliefs(
    agent: Agent,
    valid_mask: jnp.ndarray,
    obs_list: list[jnp.ndarray],
) -> jnp.ndarray:
    """Run inference and return the full posterior (masked).

    Args:
        agent: Batched Agent
        valid_mask: (n_patches, max_states) boolean mask
        obs_list: list of n_mod arrays, each (n_patches,) integer observations

    Returns:
        (n_patches, max_states) posterior beliefs with padded states zeroed
    """
    qs = agent.infer_states(obs_list, agent.D)
    beliefs = qs[0][:, 0, :]
    return jnp.where(valid_mask, beliefs, 0.0)


# ---------------------------------------------------------------------------
# Mutual information and bidirectional message passing helpers
# ---------------------------------------------------------------------------


def _mi_from_pA(pA_list: list[jnp.ndarray]) -> jnp.ndarray:
    """Compute MI I(obs; states) from a list of Dirichlet parameter arrays.

    For each modality array (n_patches, n_obs, n_states):
      - Normalize each patch independently to a joint distribution
      - Compute MI = H(o) + H(s) - H(o,s) per patch
      - Sum MI across patches
    Then sum across modalities.
    Matches spm_MI from SPM's DEM_MNIST_RGM.m.

    Returns a scalar jnp.ndarray (JIT-safe; no device→host sync).
    """
    per_mod = []
    for A_m in pA_list:
        # A_m shape: (n_patches, n_obs, n_states)
        totals = A_m.sum(axis=(1, 2), keepdims=True)  # (n_patches, 1, 1)
        A_norm = A_m / jnp.clip(totals, 1e-16, None)  # (n_patches, n_obs, n_states)
        log_A = jnp.log(jnp.clip(A_norm, 1e-16, None))
        joint = jnp.sum(A_norm * log_A, axis=(1, 2))  # (n_patches,)
        p_s = A_norm.sum(axis=1)  # (n_patches, n_states)
        p_o = A_norm.sum(axis=2)  # (n_patches, n_obs)
        h_s = jnp.sum(p_s * jnp.log(jnp.clip(p_s, 1e-16, None)), axis=1)
        h_o = jnp.sum(p_o * jnp.log(jnp.clip(p_o, 1e-16, None)), axis=1)
        per_mod.append(jnp.sum(joint - h_s - h_o))
    return jnp.stack(per_mod).sum()


def compute_level_mi(agent: Agent) -> float:
    """Compute MI I(observations; states) for an agent level.

    Uses pA (Dirichlet parameters) if available, otherwise falls back to A.

    Args:
        agent: Agent whose pA (or A) matrices define the joint distribution

    Returns:
        Mutual information in nats (Python float)
    """
    src = agent.pA if agent.pA is not None else agent.A
    return float(_mi_from_pA(src))


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
    l1_obs_list = _raw_obs_to_group_obs_list(obs_image)
    qs_l1 = levels[0].infer_states(l1_obs_list, levels[0].D)
    child_beliefs = jnp.where(valid_masks[0], qs_l1[0][:, 0, :], 0.0)

    level_soft_beliefs = [child_beliefs]
    level_obs_lists = [l1_obs_list]

    for lv_idx in range(1, n_hier):
        level = levels[lv_idx]
        child_grid = int(round(child_beliefs.shape[0] ** 0.5))  # static
        obs_list = _extract_soft_obs(child_beliefs, child_grid)
        level_obs_lists.append(obs_list)
        qs = level.infer_states(obs_list, level.D)
        child_beliefs = jnp.where(valid_masks[lv_idx], qs[0][:, 0, :], 0.0)
        level_soft_beliefs.append(child_beliefs)

    # child_beliefs is now the top hierarchical level's soft beliefs

    # --- Unsupervised prediction (for accuracy tracking) ---
    qs_cls_pred = cls_agent.infer_states([child_beliefs], cls_agent.D)
    pred = jnp.argmax(qs_cls_pred[0][:, 0, :][0])
    correct_inc = (pred == label).astype(jnp.int32)

    # --- Supervised classification with one-hot D prior ---
    supervised_D = [jax.nn.one_hot(label, n_classes)[None, :]]
    qs_cls = cls_agent.infer_states([child_beliefs], supervised_D)
    q_cls = qs_cls[0][:, 0, :]  # (1, n_classes)

    # --- Top-down pass: refine beliefs using cls posterior ---
    refined_soft = [None] * n_hier

    D_top = _top_down_D_from_cls(cls_agent.A, q_cls)
    child_input = level_soft_beliefs[n_hier - 2] if n_hier > 1 else level_soft_beliefs[0]
    child_grid_top = int(round(child_input.shape[0] ** 0.5))  # static
    obs_top = _extract_soft_obs(child_input, child_grid_top)
    qs_top_refined = levels[n_hier - 1].infer_states(obs_top, [D_top])
    refined_soft[n_hier - 1] = jnp.where(
        valid_masks[n_hier - 1], qs_top_refined[0][:, 0, :], 0.0
    )

    for lv_idx in range(n_hier - 2, -1, -1):
        parent_lv = lv_idx + 1
        q_parent = refined_soft[parent_lv]
        child_level = levels[lv_idx]
        n_child_patches = child_level.batch_size  # static
        child_grid = int(round(n_child_patches ** 0.5))  # static
        max_child_states = child_level.num_states[0]  # static
        D_child = _top_down_D_hierarchical(
            levels[parent_lv].A, q_parent, child_grid, max_child_states
        )
        if lv_idx == 0:
            qs_l1_refined = child_level.infer_states(level_obs_lists[0], [D_child])
            refined_soft[0] = jnp.where(
                valid_masks[0], qs_l1_refined[0][:, 0, :], 0.0
            )
        else:
            child_input_here = level_soft_beliefs[lv_idx - 1]
            child_grid_here = int(round(child_input_here.shape[0] ** 0.5))  # static
            obs_here = _extract_soft_obs(child_input_here, child_grid_here)
            qs_refined = child_level.infer_states(obs_here, [D_child])
            refined_soft[lv_idx] = jnp.where(
                valid_masks[lv_idx], qs_refined[0][:, 0, :], 0.0
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


def _top_down_D_from_cls(
    cls_A: list[jnp.ndarray],
    q_cls: jnp.ndarray,
) -> jnp.ndarray:
    """Top-down D for the top hierarchical level from classification beliefs.

    Computes P(top_state | data) = sum_d P(top_state | digit=d) * P(digit=d | data).

    Args:
        cls_A: classification agent A matrices; cls_A[0] shape (1, n_top_states, n_classes)
        q_cls: (1, n_classes) classification posterior

    Returns:
        (1, n_top_states) top-down prior for the top hierarchical level
    """
    D_top = jnp.einsum('sc,c->s', cls_A[0][0], q_cls[0])  # (n_top_states,)
    return D_top[None, :]  # (1, n_top_states)


def _top_down_D_hierarchical(
    parent_A: list[jnp.ndarray],
    q_parent: jnp.ndarray,
    child_grid: int,
    max_child_states: int,
) -> jnp.ndarray:
    """Top-down D for a child hierarchical level from parent beliefs.

    For each parent location, marginalises P(child_obs | parent_state) over the
    parent posterior to produce a top-down prediction over child states.

    Modality ordering matches extract_2x2_children:
    (0,0)=top-left, (0,1)=top-right, (1,0)=bottom-left, (1,1)=bottom-right.

    Args:
        parent_A: list of 4 arrays, each (n_parent_patches, max_child_states, max_parent_states)
        q_parent: (n_parent_patches, max_parent_states) parent posterior
        child_grid: child grid size (= 2 * parent_grid)
        max_child_states: max states at child level

    Returns:
        (n_child_patches, max_child_states) top-down D prior for child level
    """
    n_parent = q_parent.shape[0]
    parent_grid = child_grid // 2
    n_child = child_grid * child_grid
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]

    parent_rows = jnp.arange(n_parent) // parent_grid
    parent_cols = jnp.arange(n_parent) % parent_grid

    D_child = jnp.zeros((n_child, max_child_states))
    for m, (di, dj) in enumerate(offsets):
        pred_m = jnp.einsum('pcs,ps->pc', parent_A[m], q_parent)  # (n_parent, max_child)
        child_rows = 2 * parent_rows + di
        child_cols = 2 * parent_cols + dj
        child_idx = child_rows * child_grid + child_cols  # (n_parent,)
        D_child = D_child.at[child_idx, :].set(pred_m)

    return D_child


@eqx.filter_jit
def _generate_image_distributional(
    top_prior: jnp.ndarray,
    level_A_tuple: tuple,
    child_grids: tuple,
    max_child_states: tuple,
    l1_A_stack: jnp.ndarray,
    l1_valid_mask: jnp.ndarray,
    bin_centres: jnp.ndarray,
    V: jnp.ndarray,
    mean: jnp.ndarray,
    image_shape: tuple,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled distributional top-down generation.

    Propagates a prior over top-level states down through all hierarchy levels
    using the trained A matrices and returns the reconstructed image and
    expected bin-index observations.

    Args:
        top_prior: (n_top,) prior distribution over top-level states
        level_A_tuple: per-level A matrices, ordered from top-1 down to L2
        child_grids: child grid sizes per descent step
        max_child_states: max child state count per descent step
        l1_A_stack: (n_comp, n_patches, n_levels, max_states) stacked L1 A
        l1_valid_mask: (n_patches, max_states) boolean mask
        bin_centres: (ng, ng, n_comp, n_levels) continuous bin centre values
        V: (ng, ng, n_pixels, k) SVD basis vectors
        mean: (ng, ng, n_pixels) per-group pixel mean
        image_shape: (C, H, W) static tuple

    Returns:
        (image, obs) where image is (1, C, H, W) and obs is
        (1, ng, ng, n_comp) expected bin indices.
    """
    beliefs = top_prior[None, :]
    for parent_A, cg, mcs in zip(level_A_tuple, child_grids, max_child_states):
        beliefs = _top_down_D_hierarchical(list(parent_A), beliefs, cg, mcs)
    # beliefs: (n_l1_patches, max_l1_states) where n_l1_patches = (ng//2)^2

    q_masked = beliefs * l1_valid_mask
    n_mod_l1, n_patches, n_levels_l1, _ = l1_A_stack.shape  # n_mod_l1 = 4k
    ng2 = int(round(n_patches ** 0.5))  # L1 parent grid = ng//2
    k = n_mod_l1 // 4
    ng = 2 * ng2  # full SVD tile grid
    q_grid = q_masked.reshape(ng2, ng2, -1)  # (ng2, ng2, max_states)

    A_grid = l1_A_stack.reshape(n_mod_l1, ng2, ng2, n_levels_l1, -1)
    marginal = jnp.einsum('kijls,ijs->kijl', A_grid, q_grid)  # (4k, ng2, ng2, n_levels)
    marginal_4k = marginal.reshape(4, k, ng2, ng2, n_levels_l1)  # (4, k, ng2, ng2, n_levels)

    levels_vec = jnp.arange(n_levels_l1, dtype=bin_centres.dtype)

    # Scatter expected obs and variates from group patches back to SVD tile positions
    expected_obs_full = jnp.zeros((ng, ng, k))
    expected_variates_full = jnp.zeros((ng, ng, k))
    for t, (di, dj) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        child_rows = 2 * jnp.arange(ng2) + di  # (ng2,)
        child_cols = 2 * jnp.arange(ng2) + dj  # (ng2,)
        m_t = marginal_4k[t]  # (k, ng2, ng2, n_levels)
        ev_obs_t = jnp.einsum('cijn,n->ijc', m_t, levels_vec)  # (ng2, ng2, k)
        expected_obs_full = expected_obs_full.at[
            child_rows[:, None], child_cols[None, :], :
        ].set(ev_obs_t)
        bc_t = bin_centres[child_rows[:, None], child_cols[None, :], :, :]  # (ng2, ng2, k, n_levels)
        ev_var_t = jnp.einsum('cijn,ijcn->ijc', m_t, bc_t)  # (ng2, ng2, k)
        expected_variates_full = expected_variates_full.at[
            child_rows[:, None], child_cols[None, :], :
        ].set(ev_var_t)

    # Decode: linear unproject + tile-sum
    variates_t = expected_variates_full[None].transpose(1, 2, 0, 3)  # (ng, ng, 1, k)
    recon_weighted = jnp.einsum('ijnk,ijfk->ijnf', variates_t, V)
    recon_weighted = recon_weighted + mean[:, :, None, :]
    recon = recon_weighted.sum(axis=(0, 1))  # (1, n_pixels)

    C, H, W = image_shape
    image = recon.reshape(1, C, H, W)
    obs = expected_obs_full[None]  # (1, ng, ng, k)
    return image, obs


# ---------------------------------------------------------------------------
# Level 1: group agents (2×2 blocks of raw SVD observations, matching MATLAB)
# ---------------------------------------------------------------------------


def create_group_agents(
    stats: StateStats,
    config: DiscretiseConfig,
) -> tuple[Agent, jnp.ndarray]:
    """Build L1 batched agents taking raw 4-tile SVD observations.

    Each of the (ng//2)^2 group locations becomes one batch element with 4k
    observation modalities (4 tiles × k SVD components), each with
    config.n_levels possible values (SVD bin indices).

    Args:
        stats: StateStats from compute_group_states()
        config: DiscretiseConfig

    Returns:
        (agent, valid_mask)
    """
    ng2 = stats.num_states.shape[0]
    n_patches = ng2 * ng2
    k = config.max_components
    n_mod = 4 * k
    n_levels = config.n_levels

    num_states_flat = stats.num_states.flatten()
    max_states = int(num_states_flat.max())

    valid_mask = jnp.arange(max_states)[None, :] < num_states_flat[:, None]

    # A[m] shape: (n_patches, n_levels, max_states)
    # For valid state s: one-hot at the observed bin index for modality m
    # For padded states: uniform 1/n_levels
    A_np = np.full((n_mod, n_patches, n_levels, max_states), 1.0 / n_levels)
    for flat_idx in range(n_patches):
        i, j = divmod(flat_idx, ng2)
        patterns = np.asarray(stats.state_patterns[i][j])  # (n_s, 4k)
        n_s = patterns.shape[0]
        A_np[:, flat_idx, :, :n_s] = 0.0
        for m in range(n_mod):
            s_idx = np.arange(n_s)
            A_np[m, flat_idx, patterns[:n_s, m], s_idx] = 1.0
    A = [jnp.array(A_np[m]) for m in range(n_mod)]

    # categorical_obs=False: observations are integer bin indices
    agent = _build_agent(A, n_patches, max_states, n_mod, valid_mask, categorical_obs=False)
    return agent, valid_mask


def _raw_obs_to_group_obs_list(obs_image: jnp.ndarray) -> list[jnp.ndarray]:
    """Build a 4k-element obs_list for a group-level agent from raw tile obs.

    Each of the 4k modalities corresponds to one (tile, component) pair in the
    2×2 block. Ordered: tile 0 components 0…k-1, tile 1 components 0…k-1, …

    Args:
        obs_image: (ng, ng, k) integer bin indices

    Returns:
        list of 4k arrays, each (ng//2 * ng//2,) integer indices
    """
    ng = obs_image.shape[0]
    k = obs_image.shape[2]
    ng2 = ng // 2
    n_parent = ng2 * ng2
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]

    parent_rows = jnp.arange(n_parent) // ng2
    parent_cols = jnp.arange(n_parent) % ng2

    obs_list = []
    for di, dj in offsets:
        child_rows = 2 * parent_rows + di
        child_cols = 2 * parent_cols + dj
        tile_obs = obs_image[child_rows, child_cols, :]  # (n_parent, k)
        for c in range(k):
            obs_list.append(tile_obs[:, c])
    return obs_list


# ---------------------------------------------------------------------------
# Levels 2+: Generic hierarchical agents (2x2 pooling)
# ---------------------------------------------------------------------------


def extract_2x2_children(child_map_states: jnp.ndarray) -> jnp.ndarray:
    """Pool child MAP states into parent observations via 2x2 blocks.

    Args:
        child_map_states: (child_grid, child_grid) MAP state indices

    Returns:
        (n_parent_patches, 4) child state indices per parent, ordered
        top-left, top-right, bottom-left, bottom-right
    """
    g = child_map_states.shape[0]
    n = g // 2
    # Reshape into (n, 2, n, 2) then pick the four corners
    blocks = child_map_states.reshape(n, 2, n, 2)
    # Stack: top-left, top-right, bottom-left, bottom-right → (n, n, 4)
    result = jnp.stack([
        blocks[:, 0, :, 0],
        blocks[:, 0, :, 1],
        blocks[:, 1, :, 0],
        blocks[:, 1, :, 1],
    ], axis=-1)
    return result.reshape(n * n, 4)


def create_hierarchical_agents(
    stats: StateStats,
    child_stats: StateStats,
) -> tuple[Agent, jnp.ndarray]:
    """Build a batched pymdp Agent for one hierarchical level.

    Each parent location becomes one batch element with 4 observation
    modalities (one per child in its 2x2 block). The A matrix maps parent
    states to child state indices deterministically.

    Args:
        stats: StateStats for this level
        child_stats: stats from the child level (for num_states per child)

    Returns:
        (agent, valid_states_mask) where valid_states_mask is (n_patches, max_states)
    """
    n_grid = stats.num_states.shape[0]
    n_patches = n_grid * n_grid
    n_mod = 4  # always 4 modalities (2x2 children)

    num_states_flat = stats.num_states.flatten()  # (n_patches,)
    max_states = int(num_states_flat.max())

    # Max child states across entire child grid — observation dimension
    max_child_states = int(np.asarray(child_stats.num_states).max())

    # Valid states mask: (n_patches, max_states)
    valid_mask = jnp.arange(max_states)[None, :] < num_states_flat[:, None]

    # --- Build A matrices in NumPy, convert once ---
    # A[m] shape: (n_patches, max_child_states, max_states)
    # For valid parent state s, modality m: one-hot at the child state index
    # For padded states: uniform 1/child_ns[m]
    A_np = np.full(
        (n_mod, n_patches, max_child_states, max_states), 1.0 / max_child_states
    )
    m_idx = np.arange(n_mod)[:, None]  # (n_mod, 1) for broadcasting
    for flat_idx in range(n_patches):
        i, j = divmod(flat_idx, n_grid)
        patterns = np.asarray(stats.state_patterns[i][j])  # (n_s, 4)
        n_s = patterns.shape[0]
        s_idx = np.arange(n_s)[None, :]  # (1, n_s)
        # Padded states: uniform over each child's actual states
        for m in range(n_mod):
            child_ns = int(stats.child_num_states[i][j][m])
            if child_ns < max_child_states:
                A_np[m, flat_idx, :, n_s:max_states] = 0.0
                A_np[m, flat_idx, :child_ns, n_s:max_states] = 1.0 / child_ns
        # Valid states: one-hot at child state index (all modalities at once)
        A_np[:, flat_idx, :, :n_s] = 0.0
        A_np[m_idx, flat_idx, patterns[:n_s, :].T, s_idx] = 1.0
    A = [jnp.array(A_np[m]) for m in range(n_mod)]

    # categorical_obs=True: observations are soft probability vectors (child beliefs),
    # not integer MAP states. This lets agent.infer_states and infer_parameters handle
    # soft child posteriors natively via expected log-likelihood.
    agent = _build_agent(A, n_patches, max_states, n_mod, valid_mask, categorical_obs=True)
    return agent, valid_mask


def infer_hierarchical_states(
    agent: Agent,
    valid_mask: jnp.ndarray,
    child_map_states: jnp.ndarray,
) -> jnp.ndarray:
    """Run inference at one hierarchical level for a single image.

    Converts integer child MAP states to one-hot vectors before passing to the
    agent, which is built with ``categorical_obs=True``. One-hot is a special
    case of a probability distribution and gives identical results to integer
    indexing.

    Args:
        agent: Batched Agent from create_hierarchical_agents()
        valid_mask: (n_patches, max_states) boolean mask
        child_map_states: (child_grid, child_grid) MAP state indices at child level

    Returns:
        (parent_grid, parent_grid) MAP state indices at this level
    """
    child_grid = child_map_states.shape[0]
    parent_grid = child_grid // 2
    obs_block = extract_2x2_children(child_map_states)  # (n_parent, 4) integer indices
    max_child_states = agent.num_obs[0]
    obs_list = [
        jax.nn.one_hot(obs_block[:, m], max_child_states)
        for m in range(4)
    ]
    return _infer_map_states(agent, valid_mask, obs_list, (parent_grid, parent_grid))


def _extract_soft_obs(child_beliefs: jnp.ndarray, child_grid: int) -> list[jnp.ndarray]:
    """Build a 4-element soft obs_list for a hierarchical level from child beliefs.

    Each element corresponds to one child position in a 2x2 block (TL, TR, BL, BR).
    The result is passed directly to agent.infer_states for a categorical_obs=True agent.

    Args:
        child_beliefs: (n_child_patches, max_child_states) soft child posteriors
        child_grid: child grid size (n_child_patches = child_grid^2)

    Returns:
        list of 4 arrays each (n_parent_patches, max_child_states)
    """
    n_parent_grid = child_grid // 2
    n_parent = n_parent_grid * n_parent_grid
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]

    parent_rows = jnp.arange(n_parent) // n_parent_grid
    parent_cols = jnp.arange(n_parent) % n_parent_grid

    obs_list = []
    for di, dj in offsets:
        child_rows = 2 * parent_rows + di
        child_cols = 2 * parent_cols + dj
        child_idx = child_rows * child_grid + child_cols  # (n_parent,)
        obs_list.append(child_beliefs[child_idx])  # (n_parent, max_child_states)
    return obs_list


# ---------------------------------------------------------------------------
# Classification level: top-level states → digit classes
# ---------------------------------------------------------------------------


def create_classification_agent(
    top_stats: StateStats,
    y_exemplars: np.ndarray,
    n_classes: int = 10,
) -> tuple[Agent, jnp.ndarray]:
    """Build an agent mapping top-level states to digit classes.

    The A matrix encodes P(top_state | digit), learned from exemplar label
    counts. Ambiguous top states (shared by multiple digits) naturally spread
    posterior mass, capturing similarity structure (e.g. 1 and 7 share states).

    Args:
        top_stats: StateStats from the final hierarchical level (1x1 grid)
        y_exemplars: (N,) digit labels for the exemplar images
        n_classes: number of output classes

    Returns:
        (agent, valid_mask) where valid_mask is (1, n_classes) all-True
    """
    n_top_states = int(top_stats.num_states[0, 0])

    # A shape: (1 batch, n_top_states obs, n_classes hidden)
    # Count how many exemplars of each digit land in each top state
    A_np = np.zeros((1, n_top_states, n_classes))
    for s in range(n_top_states):
        for idx in top_stats.state_to_images[0][0][s]:
            A_np[0, s, int(y_exemplars[idx])] += 1

    # Normalize columns: P(top_state | digit)
    col_sums = A_np[0].sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1, col_sums)
    A_np[0] = A_np[0] / col_sums

    A = [jnp.array(A_np)]
    valid_mask = jnp.ones((1, n_classes), dtype=bool)
    # categorical_obs=True: observation is a soft distribution over top-level states
    agent = _build_agent(
        A, n_patches=1, max_states=n_classes, n_mod=1, valid_mask=valid_mask,
        categorical_obs=True,
    )
    return agent, valid_mask


def infer_classification(
    agent: Agent,
    valid_mask: jnp.ndarray,
    top_state: jnp.ndarray,
) -> jnp.ndarray:
    """Run classification inference: top-level state → digit posterior.

    Args:
        agent: Agent from create_classification_agent()
        valid_mask: (1, n_classes) boolean mask
        top_state: scalar top-level MAP state index

    Returns:
        (1, 1) MAP digit class index
    """
    n_top_states = agent.num_obs[0]
    obs_list = [jax.nn.one_hot(jnp.array([top_state]), n_top_states)]
    return _infer_map_states(agent, valid_mask, obs_list, (1, 1))


# ---------------------------------------------------------------------------
# Helper functions for top-down generation
# ---------------------------------------------------------------------------


def _expand_grid(
    parent_grid: np.ndarray,
    stats: StateStats,
) -> np.ndarray:
    """Expand parent state grid to child state grid via state_patterns.

    Each parent cell at (i, j) with state s is expanded into a 2x2 block of
    child states using stats.state_patterns[i][j][s] → (4,) children ordered
    top-left, top-right, bottom-left, bottom-right.

    Args:
        parent_grid: (n, n) parent MAP state indices
        stats: StateStats for this level

    Returns:
        (2n, 2n) child MAP state indices
    """
    n = parent_grid.shape[0]
    child_grid = np.zeros((2 * n, 2 * n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            s = int(parent_grid[i, j])
            children = np.asarray(stats.state_patterns[i][j][s])
            ci, cj = 2 * i, 2 * j
            child_grid[ci, cj] = children[0]
            child_grid[ci, cj + 1] = children[1]
            child_grid[ci + 1, cj] = children[2]
            child_grid[ci + 1, cj + 1] = children[3]
    return child_grid


def _expand_to_obs(
    l1_grid: np.ndarray,
    l1_stats: StateStats,
    config: DiscretiseConfig,
) -> np.ndarray:
    """Expand L1 group state grid to raw SVD observation grid.

    l1_grid is (ng//2, ng//2) and each state's pattern has shape (4k,)
    encoding the 4 child tiles' k SVD components.

    Args:
        l1_grid: (ng//2, ng//2) L1 MAP state indices
        l1_stats: StateStats from compute_group_states
        config: DiscretiseConfig

    Returns:
        (ng, ng, k) discrete bin indices
    """
    ng = config.image_size // config.group_size
    k = config.max_components
    ng2 = ng // 2
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
    obs_grid = np.zeros((ng, ng, k), dtype=np.int32)
    for i in range(ng2):
        for j in range(ng2):
            s = int(l1_grid[i, j])
            pattern = np.asarray(l1_stats.state_patterns[i][j][s])  # (4k,)
            for t, (di, dj) in enumerate(offsets):
                obs_grid[2 * i + di, 2 * j + dj, :] = pattern[t * k:(t + 1) * k]
    return obs_grid


# ---------------------------------------------------------------------------
# Compiled per-image classify helpers (used by RGMHierarchy.classify)
# ---------------------------------------------------------------------------


def _classify_one(
    obs_i: jnp.ndarray,
    level_agents: tuple,
    level_masks: tuple,
    cls_agent,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-image bidirectional inference, safe for use under jax.lax.map.

    Runs one bottom-up pass, one initial classification, then one top-down
    refinement pass, and a final re-classification with the refined top beliefs.

    Args:
        obs_i: (n_groups, n_groups, max_components) encoded observations.
        level_agents: tuple of Agent objects for the hierarchical levels
            (L1 patch agent first, then L2+ agents; excludes cls agent).
        level_masks: tuple of (n_patches, max_states) boolean masks,
            one per hierarchical level.
        cls_agent: classification Agent.

    Returns:
        (predicted_digit, top_state, cls_beliefs) as on-device scalars/arrays.
    """
    n_hier = len(level_agents)

    # === Bottom-up pass ===
    l1_obs_list = _raw_obs_to_group_obs_list(obs_i)
    qs_l1 = level_agents[0].infer_states(l1_obs_list, level_agents[0].D)
    child_beliefs = jnp.where(level_masks[0], qs_l1[0][:, 0, :], 0.0)
    level_soft_beliefs = [child_beliefs]
    level_obs_lists = [l1_obs_list]

    for lv_idx in range(1, n_hier):
        child_grid = int(round(child_beliefs.shape[0] ** 0.5))
        obs_list = _extract_soft_obs(child_beliefs, child_grid)
        level_obs_lists.append(obs_list)
        qs = level_agents[lv_idx].infer_states(obs_list, level_agents[lv_idx].D)
        child_beliefs = jnp.where(level_masks[lv_idx], qs[0][:, 0, :], 0.0)
        level_soft_beliefs.append(child_beliefs)

    # === Initial classification ===
    qs_cls = cls_agent.infer_states([child_beliefs], cls_agent.D)
    q_cls = qs_cls[0][:, 0, :]  # (1, n_classes)

    # === Top-down refinement pass ===
    refined_soft = [None] * n_hier

    D_top = _top_down_D_from_cls(cls_agent.A, q_cls)
    child_input = level_soft_beliefs[n_hier - 2] if n_hier > 1 else level_soft_beliefs[0]
    child_grid_top = int(round(child_input.shape[0] ** 0.5))
    obs_top = _extract_soft_obs(child_input, child_grid_top)
    qs_top_refined = level_agents[n_hier - 1].infer_states(obs_top, [D_top])
    refined_soft[n_hier - 1] = jnp.where(
        level_masks[n_hier - 1], qs_top_refined[0][:, 0, :], 0.0
    )

    for lv_idx in range(n_hier - 2, -1, -1):
        parent_lv = lv_idx + 1
        q_parent = refined_soft[parent_lv]
        n_child_patches = level_agents[lv_idx].batch_size
        child_grid = int(round(n_child_patches ** 0.5))
        max_child_states = level_agents[lv_idx].num_states[0]

        D_child = _top_down_D_hierarchical(
            level_agents[parent_lv].A,
            q_parent,
            child_grid,
            max_child_states,
        )

        if lv_idx == 0:
            qs_l1_refined = level_agents[0].infer_states(level_obs_lists[0], [D_child])
            refined_soft[0] = jnp.where(level_masks[0], qs_l1_refined[0][:, 0, :], 0.0)
        else:
            child_input = level_soft_beliefs[lv_idx - 1]
            child_grid_here = int(round(child_input.shape[0] ** 0.5))
            obs_here = _extract_soft_obs(child_input, child_grid_here)
            qs_refined = level_agents[lv_idx].infer_states(obs_here, [D_child])
            refined_soft[lv_idx] = jnp.where(
                level_masks[lv_idx], qs_refined[0][:, 0, :], 0.0
            )

    refined_top = refined_soft[n_hier - 1]
    qs_cls_final = cls_agent.infer_states([refined_top], cls_agent.D)
    cls_beliefs = qs_cls_final[0][:, 0, :]  # (1, n_classes)

    predicted_digit = jnp.argmax(cls_beliefs[0])
    top_state = jnp.argmax(refined_top[0])
    return predicted_digit, top_state, cls_beliefs[0]


@jax.jit
def _classify_batch(
    level_agents: tuple,
    level_masks: tuple,
    cls_agent,
    obs_batch: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Map _classify_one over N images in a single compiled program.

    Decorated with @jax.jit so the first call compiles once and subsequent
    calls (even after training updates the agent weights) reuse the same XLA
    program — agent arrays flow through as dynamic inputs, not compile-time
    constants.  jax.lax.map keeps peak memory comparable to single-image
    inference by executing the body sequentially on-device.

    Args:
        level_agents: tuple of Agent objects (L1 … Ltop, no cls).
        level_masks: tuple of valid-state boolean masks, one per level.
        cls_agent: classification Agent.
        obs_batch: (N, n_groups, n_groups, max_components) encoded images.

    Returns:
        (predicted_digits, top_states, digit_beliefs) as JAX arrays.
    """
    def classify_one(obs_i):
        return _classify_one(obs_i, level_agents, level_masks, cls_agent)

    return jax.lax.map(classify_one, obs_batch)


# ---------------------------------------------------------------------------
# RGMHierarchy: unified classification and generation
# ---------------------------------------------------------------------------


class RGMHierarchy:
    """N-level Renormalised Generative Model hierarchy.

    Wraps pre-built agents and state statistics for bottom-up classification
    (image → digit) and top-down generation (digit → image).

    The number of levels is derived from the config:
        n_groups = image_size // group_size
        n_levels = log2(n_groups) + 1

    levels[0] is the patch level (L1), levels[1:-1] are hierarchical levels,
    levels[-1] is the classification level (top states → digit classes).
    """

    def __init__(
        self,
        config: DiscretiseConfig,
        basis: OverlappingSVDBasis,
        y_exemplars: np.ndarray,
        levels: list[RGMLevel],
        n_classes: int = 10,
    ):
        self.config = config
        self.basis = basis
        self.y_exemplars = np.asarray(y_exemplars)
        self.levels = levels
        self.n_classes = n_classes

    @classmethod
    def from_exemplars(
        cls,
        x_exemplars: jnp.ndarray,
        y_exemplars: np.ndarray,
        config: DiscretiseConfig | None = None,
        n_classes: int = 10,
    ) -> "RGMHierarchy":
        """Build the full N-level hierarchy from preprocessed exemplar images.

        The number of levels is derived from the config:
            n_groups = image_size // group_size
            n_levels = log2(n_groups) + 1

        Args:
            x_exemplars: (N, H, W) or (N, C, H, W) preprocessed structure images
            y_exemplars: (N,) digit labels
            config: discretisation config (default: DiscretiseConfig())
            n_classes: number of output digit classes

        Returns:
            Fully constructed RGMHierarchy
        """
        if config is None:
            config = DiscretiseConfig()

        n_groups = config.image_size // config.group_size
        if n_groups & (n_groups - 1) != 0:
            raise ValueError(
                f"n_groups={n_groups} must be a power of 2 "
                f"(image_size={config.image_size}, group_size={config.group_size})"
            )
        n_halvings = int(math.log2(n_groups))

        # SVD basis and discretisation
        basis = compute_svd_basis_overlapping(x_exemplars, config)
        observations = encode_images_overlapping(x_exemplars, basis)

        # Level 1: group states (2×2 blocks of raw SVD observations, matching MATLAB)
        l1_stats = compute_group_states(observations, config)
        l1_agent, l1_valid_mask = create_group_agents(l1_stats, config)

        # L1 MAP states for structure learning at L2+
        l1_maps = jnp.stack([
            _infer_map_states(
                l1_agent, l1_valid_mask,
                _raw_obs_to_group_obs_list(observations[idx]),
                output_shape=(n_groups // 2, n_groups // 2),
            )
            for idx in range(len(observations))
        ])

        levels = [RGMLevel(l1_agent, l1_valid_mask, l1_stats)]
        maps = l1_maps
        prev_stats = l1_stats
        grid_size = n_groups // 2  # already halved by the group step

        for _ in range(n_halvings - 1):  # one fewer halving
            stats = compute_hierarchical_state_stats(maps, prev_stats)
            agent, valid_mask = create_hierarchical_agents(stats, prev_stats)
            levels.append(RGMLevel(agent, valid_mask, stats))
            grid_size //= 2
            if grid_size > 1:
                # Infer states at this level for all exemplars
                maps = jnp.stack([
                    infer_hierarchical_states(agent, valid_mask, maps[idx])
                    for idx in range(len(maps))
                ])
            prev_stats = stats

        # Classification level
        cls_agent, cls_valid_mask = create_classification_agent(
            prev_stats, np.asarray(y_exemplars), n_classes
        )
        levels.append(RGMLevel(cls_agent, cls_valid_mask, prev_stats))

        return cls(
            config=config,
            basis=basis,
            y_exemplars=np.asarray(y_exemplars),
            levels=levels,
            n_classes=n_classes,
        )

    @property
    def cls_agent(self) -> Agent:
        """The classification agent (final level)."""
        return self.levels[-1].agent

    @property
    def cls_valid_mask(self) -> jnp.ndarray:
        """Valid mask for the classification agent."""
        return self.levels[-1].valid_mask

    @property
    def hierarchical_levels(self) -> list[RGMLevel]:
        """All levels except the classification level."""
        return self.levels[:-1]

    def classify(
        self, images: jnp.ndarray
    ) -> tuple[np.ndarray, np.ndarray, jnp.ndarray]:
        """Bidirectional inference: images → digit predictions.

        Runs the full hierarchy bottom-up, then applies a top-down refinement
        pass using the classification posterior to sharpen beliefs, then
        re-classifies with the refined top-level beliefs.

        The per-image computation is compiled once via ``jax.jit`` and iterated
        over the batch with ``jax.lax.map``, eliminating the Python loop and all
        per-image host syncs.  The first call incurs a one-time compilation cost;
        subsequent calls (including after training) reuse the same XLA program
        because agent weights flow through as dynamic inputs.

        Args:
            images: (N, H, W) or (N, C, H, W) preprocessed images

        Returns:
            (predicted_digits, top_states, digit_beliefs) where:
            - predicted_digits: (N,) MAP digit class per image
            - top_states: (N,) top-level state index per image
            - digit_beliefs: (N, n_classes) posterior over digits per image
        """
        observations = encode_images_overlapping(images, self.basis)
        level_agents = tuple(lv.agent for lv in self.hierarchical_levels)
        level_masks = tuple(lv.valid_mask for lv in self.hierarchical_levels)

        preds, tops, beliefs = _classify_batch(
            level_agents, level_masks, self.cls_agent, observations
        )
        return np.asarray(preds), np.asarray(tops), beliefs

    def generate(
        self,
        digit: int | None = None,
        prior: jnp.ndarray | None = None,
        sample: bool = False,
        key: jax.Array | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Top-down generation: digit prior → reconstructed image.

        By default propagates the full prior distribution down through every
        level using the learned A matrices, producing an image that is the
        expected reconstruction under the distribution.  Each level's A matrix
        marginalises over parent states to give child-state beliefs
        (``_top_down_D_hierarchical``), and at L1 the expected bin centres are
        fed directly into the linear unproject + tile-sum decoder — no argmax
        anywhere in the pipeline.

        When ``sample=True``, a single top-level state is drawn from the prior
        and the existing deterministic pattern-lookup path is used instead.

        Args:
            digit: target digit (builds prior from classification A column)
            prior: explicit prior distribution over top-level states (overrides digit)
            sample: if True, sample one top state and expand deterministically;
                    if False (default), propagate the full distribution
            key: JAX PRNG key for sampling (required if sample=True)

        Returns:
            (image, observations) where image is (1, C, H, W).  For
            sample=False, observations is (1, n_groups, n_groups, max_components)
            with float expected bin indices; for sample=True, integer bin indices.
        """
        top_stats = self.hierarchical_levels[-1].stats
        n_top = int(top_stats.num_states[0, 0])

        # Build top-level prior from the classification agent's A matrix
        # A[0] shape: (1, n_top_states, n_classes) — P(top_state | digit)
        cls_A = jnp.asarray(self.cls_agent.A[0][0])  # (n_top_states, n_classes)
        if prior is not None:
            top_prior = jnp.asarray(prior)
        elif digit is not None:
            top_prior = cls_A[:, digit]
            if top_prior.sum() == 0:
                raise ValueError(f"No top-level states found for digit {digit}")
            top_prior = top_prior / top_prior.sum()
        else:
            top_prior = jnp.ones(n_top) / n_top

        if sample:
            # --- Point-estimate path: sample one state, expand deterministically ---
            if key is None:
                key = jr.PRNGKey(0)
            s_top = int(jr.choice(key, n_top, p=top_prior))
            grid = np.array([[s_top]], dtype=np.int32)
            # Expand from top through L2+ levels; L1 is group-based (handled by _expand_to_obs)
            for level in reversed(self.hierarchical_levels[1:]):
                grid = _expand_grid(grid, level.stats)
            # grid is now (ng//2, ng//2) — L1 group states; expand to (ng, ng, k)
            obs_grid = _expand_to_obs(grid, self.levels[0].stats, self.config)
            obs_jnp = jnp.array(obs_grid)[None]
            image = decode_observations_overlapping(obs_jnp, self.basis)
            return image, obs_jnp

        # --- Distributional path: propagate full prior through trained A matrices ---
        levels = self.hierarchical_levels  # [L1, L2, ..., Ltop]
        level_A_tuple = tuple(
            tuple(levels[i].agent.A)
            for i in range(len(levels) - 1, 0, -1)
        )
        child_grids = tuple(
            int(round(levels[i - 1].agent.batch_size ** 0.5))
            for i in range(len(levels) - 1, 0, -1)
        )
        max_child_states_tuple = tuple(
            int(levels[i - 1].agent.num_states[0])
            for i in range(len(levels) - 1, 0, -1)
        )

        l1_level = self.levels[0]
        l1_A_stack = jnp.stack(list(l1_level.agent.A), axis=0)
        # l1_A_stack: (n_comp, n_patches, n_levels, max_states)

        C = self.config.n_channels
        H = W = self.config.image_size
        image_shape = (C, H, W)

        image, obs_jnp = _generate_image_distributional(
            top_prior,
            level_A_tuple,
            child_grids,
            max_child_states_tuple,
            l1_A_stack,
            l1_level.valid_mask,
            self.basis.bin_centres,
            self.basis.V,
            self.basis.mean,
            image_shape,
        )
        return image, obs_jnp

    def train(
        self,
        x_train: jnp.ndarray,
        y_train: np.ndarray,
        concentration_lower: float = 1 / 16,
        concentration_cls: float = 1 / 128,
        lr_pA: float = 1.0,
        beta: float = 512.0,
        eta: float = 512.0,
        log_every: int = 500,
        scan_chunk_size: int = 50,
    ) -> dict:
        """Parametric training via sequential Dirichlet updates with soft beliefs
        and MI-gated learning, following SPM's DEM_MNIST_RGM.m.

        Inference uses soft beliefs throughout: child posterior distributions are
        passed between levels via expected log-likelihood rather than MAP states,
        removing the argmax bottleneck that amplifies errors during learning.

        For each training image:
        1. Bottom-up soft pass: L1 → soft beliefs, L2+ → soft beliefs via expected
           log-likelihood via ``agent.infer_states`` with ``categorical_obs=True``.
        2. Classification: soft top beliefs + supervised one-hot D → digit posterior.
        3. Top-down pass: propagate classification posterior back through all levels,
           refining beliefs using top-down D priors (soft at L2+, infer_states at L1).
        4. MI-gated Dirichlet updates at every level:
               Pa = softmax(beta * [MI(pa), MI(qa)])
               pA_new = (Pa[0]*pa + Pa[1]*qa) * eta / (eta + Pa[1])

        The inner loop is compiled as a ``lax.scan`` over sub-chunks of
        ``scan_chunk_size`` images (one XLA program per unique sub-chunk size),
        eliminating per-operation Python dispatch overhead.  ``log_every`` controls
        only how often progress is printed; ``scan_chunk_size`` controls the XLA
        program size and peak memory.  On GPU, reduce ``scan_chunk_size`` (e.g. 50)
        if you hit OOM during compilation.

        Args:
            x_train: (N, H, W) preprocessed training images
            y_train: (N,) integer digit labels
            concentration_lower: initial Dirichlet concentration for L1/L2+ levels
            concentration_cls: initial Dirichlet concentration for the classification level
            lr_pA: learning rate passed to infer_parameters
            beta: softmax sharpness for MI gate (SPM uses 512)
            eta: asymptote / forgetting parameter (SPM uses 512)
            log_every: how often (in images) to print a progress line and record MI.
            scan_chunk_size: number of images per ``lax.scan`` call. Smaller values
                use less peak memory at the cost of more XLA compilations (one per
                unique chunk size). Must be ≤ log_every. Reduce to 25–50 on GPU if
                compilation OOMs.

        Returns:
            dict with running_accuracy, mi_history, mi_checkpoints, mi_upper_bounds
        """
        # --- Phase A: initialize learnable agents at ALL levels ---
        for lv_idx, level in enumerate(self.levels):
            is_cls = lv_idx == len(self.levels) - 1
            concentration = concentration_cls if is_cls else concentration_lower
            pA = [A_m + concentration for A_m in level.agent.A]
            updated_agent = _build_agent(
                list(level.agent.A),
                level.agent.batch_size,
                level.agent.num_states[0],
                level.agent.num_modalities,
                level.valid_mask,
                learn_A=True,
                pA=pA,
                categorical_obs=level.agent.categorical_obs,
            )
            self.levels[lv_idx] = RGMLevel(updated_agent, level.valid_mask, level.stats)

        # --- Phase B: lax.scan-based training loop ---
        n_hier = len(self.hierarchical_levels)
        N = len(x_train)
        running_accuracy = []
        mi_history = []
        mi_checkpoints = []

        # Encode in chunks to avoid materialising the (ng, ng, N, n_pixels) intermediate
        # tensor all at once.  The output (integer bin indices) is tiny; only the SVD
        # projection step is large.  3 000 images ≈ 0.8 GB intermediate; 10 000 ≈ 2.6 GB.
        _encode_bs = 3_000
        obs_all = jnp.concatenate(
            [
                encode_images_overlapping(x_train[s : s + _encode_bs], self.basis)
                for s in range(0, N, _encode_bs)
            ],
            axis=0,
        )  # (N, n_i, n_j, n_mod)
        labels_all = jnp.asarray(y_train, dtype=jnp.int32)

        # Build initial carry from the freshly-initialised agents.
        level_agents = tuple(lv.agent for lv in self.levels[:n_hier])
        cls_agent_init = self.levels[-1].agent
        valid_masks = tuple(lv.valid_mask for lv in self.levels[:n_hier])

        initial_carry = _TrainCarry(levels=level_agents, cls_agent=cls_agent_init)

        # Partition into dynamic (JAX arrays) and static (non-array metadata).
        # lax.scan requires all carry leaves to be JAX arrays; static fields
        # (batch_size, num_states, A_dependencies, …) are captured in static_carry
        # and re-combined at the start of every scan body call.
        dynamic_carry, static_carry = eqx.partition(initial_carry, eqx.is_array)

        # Build the scan body with static args closed over via partial.
        step_fn = functools.partial(
            _train_step,
            valid_masks=valid_masks,
            n_classes=self.n_classes,
            beta=beta,
            eta=eta,
        )

        def _scan_body(dynamic_carry, x):
            carry = eqx.combine(dynamic_carry, static_carry)
            new_carry, metrics = step_fn(carry, x)
            new_dynamic, _ = eqx.partition(new_carry, eqx.is_array)
            return new_dynamic, metrics

        @eqx.filter_jit
        def _scan_chunk(dc, obs_c, lbl_c):
            return lax.scan(_scan_body, dc, (obs_c, lbl_c))

        # Record MI at initialisation (before any images).
        mi_history.append([compute_level_mi(lv.agent) for lv in self.levels])
        mi_checkpoints.append(0)

        prior_correct = 0  # cumulative correct count across all chunks

        for chunk_start in range(0, N, log_every):
            chunk_end = min(chunk_start + log_every, N)

            # Break the log_every window into scan_chunk_size sub-chunks so XLA
            # compiles a smaller program (fewer traced steps → less peak memory).
            correct_inc_parts = []
            for sub_start in range(chunk_start, chunk_end, scan_chunk_size):
                sub_end = min(sub_start + scan_chunk_size, chunk_end)
                obs_sub = obs_all[sub_start:sub_end]
                lbl_sub = labels_all[sub_start:sub_end]
                dynamic_carry, sub_metrics = _scan_chunk(dynamic_carry, obs_sub, lbl_sub)
                correct_inc_parts.append(np.asarray(sub_metrics["correct_inc"]))

            # Reconstruct running accuracy for this chunk without device syncs.
            correct_inc = np.concatenate(correct_inc_parts)  # (chunk_size,)
            cumcorrect = np.cumsum(correct_inc) + prior_correct
            total_seen = np.arange(chunk_start + 1, chunk_end + 1)
            running_accuracy.extend((cumcorrect / total_seen).tolist())
            prior_correct = int(cumcorrect[-1])

            # MI snapshot (one host-device sync per chunk, not per image).
            carry_here = eqx.combine(dynamic_carry, static_carry)
            mi_snap = (
                [compute_level_mi(lv) for lv in carry_here.levels]
                + [compute_level_mi(carry_here.cls_agent)]
            )
            mi_history.append(mi_snap)
            mi_checkpoints.append(chunk_end)

            mi_str = "  ".join(
                f"L{i+1}={v:.3f}" if i < n_hier else f"cls={v:.3f}"
                for i, v in enumerate(mi_snap)
            )
            print(
                f"  [{chunk_end}/{N}] "
                f"acc={running_accuracy[-1]:.3f}  MI: {mi_str}"
            )

        # Write updated agents back into self.levels.
        final_carry = eqx.combine(dynamic_carry, static_carry)
        for lv_idx in range(n_hier):
            level = self.levels[lv_idx]
            self.levels[lv_idx] = RGMLevel(
                final_carry.levels[lv_idx], level.valid_mask, level.stats
            )
        cls_level = self.levels[-1]
        self.levels[-1] = RGMLevel(
            final_carry.cls_agent, cls_level.valid_mask, cls_level.stats
        )

        # MI upper bounds: sum over modalities and patches of min(log(n_obs_m), log(n_states))
        # Matches the aggregation in _mi_from_pA (per-patch, per-modality summation).
        mi_upper_bounds = []
        for lv in self.levels:
            n_patches = lv.agent.batch_size
            n_states = lv.agent.num_states[0]
            bound = n_patches * sum(
                min(np.log(n_obs_m), np.log(n_states))
                for n_obs_m in lv.agent.num_obs
            )
            mi_upper_bounds.append(float(bound))

        print(f"Training complete. Final running accuracy: {running_accuracy[-1]:.3f}")
        return {
            "running_accuracy": running_accuracy,
            "mi_history": mi_history,
            "mi_checkpoints": mi_checkpoints,
            "mi_upper_bounds": mi_upper_bounds,
        }
