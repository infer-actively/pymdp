"""Top-down generation for Renormalised Generative Models.

Propagates a prior over top-level states down through all hierarchy levels
using trained A matrices, and expands state grids to raw SVD observations
for deterministic (sample-based) generation.
"""

import numpy as np
import jax
import jax.numpy as jnp

import equinox as eqx

from discretise import DiscretiseConfig
from state_stats import StateStats
from hierarchical import _top_down_D_hierarchical


# ---------------------------------------------------------------------------
# Top-down generation helpers
# ---------------------------------------------------------------------------


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
