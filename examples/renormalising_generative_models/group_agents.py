"""Level 1 group agents for Renormalised Generative Models.

Builds batched pymdp Agents for 2×2 blocks of raw SVD observations,
matching MATLAB's spm_MB_structure_learning at L1.  No argmax bottleneck
between raw observations and the first pooling level.
"""

import numpy as np
import jax.numpy as jnp

from pymdp.agent import Agent

from discretise import DiscretiseConfig
from state_stats import StateStats
from agent_build import _build_agent


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
