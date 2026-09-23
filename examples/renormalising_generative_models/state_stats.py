"""State vocabulary statistics for Renormalised Generative Models.

Computes unique joint observation patterns per 2×2 group (L1) and unique
child-state tuples per 2x2 block of children (L2+).  These state vocabularies
form the basis for building the batched pymdp Agents at each level.
"""

import numpy as np
from jax import numpy as jnp
from typing import NamedTuple

from discretise import DiscretiseConfig


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
