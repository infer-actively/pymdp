"""Levels 2+ hierarchical agents and bidirectional inference.

Generic 2×2-pooling agents that pass soft probability distributions between
levels, preserving uncertainty and avoiding the argmax bottleneck.  Also
provides the top-down D helpers used in both inference refinement and
distributional generation.
"""

import jax
import jax.numpy as jnp
import numpy as np

from pymdp.agent import Agent

from state_stats import StateStats
from agent_build import _build_agent, _infer_map_states
from group_agents import _raw_obs_to_group_obs_list


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
# Top-down D helpers (used in inference refinement and distributional generation)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Bidirectional inference passes
# ---------------------------------------------------------------------------


def _bottom_up_pass(
    levels: tuple,
    valid_masks: tuple,
    obs_image: jnp.ndarray,
) -> tuple[list, list]:
    """Run the bottom-up soft inference pass through all hierarchical levels.

    Args:
        levels: tuple of Agent objects (L1 first)
        valid_masks: tuple of (n_patches, max_states) boolean masks
        obs_image: (ng, ng, k) encoded observation image

    Returns:
        (level_soft_beliefs, level_obs_lists) ordered bottom to top
    """
    l1_obs_list = _raw_obs_to_group_obs_list(obs_image)
    qs_l1 = levels[0].infer_states(l1_obs_list, levels[0].D)
    child_beliefs = jnp.where(valid_masks[0], qs_l1[0][:, 0, :], 0.0)

    level_soft_beliefs = [child_beliefs]
    level_obs_lists = [l1_obs_list]

    for lv_idx in range(1, len(levels)):
        child_grid = int(round(child_beliefs.shape[0] ** 0.5))
        obs_list = _extract_soft_obs(child_beliefs, child_grid)
        level_obs_lists.append(obs_list)
        qs = levels[lv_idx].infer_states(obs_list, levels[lv_idx].D)
        child_beliefs = jnp.where(valid_masks[lv_idx], qs[0][:, 0, :], 0.0)
        level_soft_beliefs.append(child_beliefs)

    return level_soft_beliefs, level_obs_lists


def _top_down_refinement_pass(
    levels: tuple,
    valid_masks: tuple,
    cls_agent: Agent,
    q_cls: jnp.ndarray,
    level_soft_beliefs: list,
    level_obs_lists: list,
) -> list:
    """Run the top-down refinement pass through all hierarchical levels.

    Args:
        levels: tuple of Agent objects (L1 first)
        valid_masks: tuple of (n_patches, max_states) boolean masks
        cls_agent: classification Agent
        q_cls: (1, n_classes) classification posterior
        level_soft_beliefs: bottom-up beliefs per level (from _bottom_up_pass)
        level_obs_lists: obs lists per level (from _bottom_up_pass)

    Returns:
        refined_soft: list of (n_patches, max_states) refined beliefs per level
    """
    n_hier = len(levels)
    refined_soft = [None] * n_hier

    D_top = _top_down_D_from_cls(cls_agent.A, q_cls)
    child_input = level_soft_beliefs[n_hier - 2] if n_hier > 1 else level_soft_beliefs[0]
    child_grid_top = int(round(child_input.shape[0] ** 0.5))
    obs_top = _extract_soft_obs(child_input, child_grid_top)
    qs_top_refined = levels[n_hier - 1].infer_states(obs_top, [D_top])
    refined_soft[n_hier - 1] = jnp.where(
        valid_masks[n_hier - 1], qs_top_refined[0][:, 0, :], 0.0
    )

    for lv_idx in range(n_hier - 2, -1, -1):
        parent_lv = lv_idx + 1
        q_parent = refined_soft[parent_lv]
        child_level = levels[lv_idx]
        n_child_patches = child_level.batch_size
        child_grid = int(round(n_child_patches ** 0.5))
        max_child_states = child_level.num_states[0]
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
            child_grid_here = int(round(child_input_here.shape[0] ** 0.5))
            obs_here = _extract_soft_obs(child_input_here, child_grid_here)
            qs_refined = child_level.infer_states(obs_here, [D_child])
            refined_soft[lv_idx] = jnp.where(
                valid_masks[lv_idx], qs_refined[0][:, 0, :], 0.0
            )

    return refined_soft
