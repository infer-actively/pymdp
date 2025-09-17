from __future__ import annotations

from typing import Iterable, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import jax.tree_util as jtu

from .env import Env


class GridWorld(Env):
    """
    Classic 2D grid world as a JAX-compatible Env.

    Hidden state factors
    --------------------
    - factor 0 (location): a single discrete factor with `rows * cols` states (linear index row*cols + col)

    Observations
    ------------
    - modality 0 (location observation): identity over location (the agent observes its true grid index)

    Controls / Actions
    ------------------
    - One control factor with 4 or 5 discrete actions:
        0: UP
        1: RIGHT
        2: DOWN
        3: LEFT
        4: STAY   (optional; controlled by include_stay=True)

    Parameters
    ----------
    shape : tuple[int, int]
        Grid shape as (rows, cols).
    walls : Optional[Iterable[Tuple[int, int]]]
        Iterable of (row, col) coordinates that are *blocked* and cannot be entered.
        Invalid attempts to enter a wall result in staying put.
    initial_position : Optional[Tuple[int, int] | int]
        If provided, sets the initial position deterministically (one-hot D).
        Accepts (row, col) or linear index (0..rows*cols-1).
        If None, D is uniform over all *free* cells.
    include_stay : bool
        If True (default), include a "STAY" action as the 5th action.
    success_prob : float
        Probability (in [0, 1]) that a movement action succeeds and moves to its intended neighbor.
        The remaining probability (1 - success_prob) results in staying in place.
        Invalid moves (out-of-bounds or into walls) always result in staying put with probability 1.
    batch_size : int
        Number of parallel environments.

    Notes
    -----
    - Shapes follow the JAX Env API:
        A[0].shape == (batch, num_obs, num_states)
        B[0].shape == (batch, num_states, num_states, num_actions)
        D[0].shape == (batch, num_states)
        dependencies["A"] == [[0]]
        dependencies["B"] == [[0]]
    - Works directly with `pymdp.envs.rollout.rollout` and the JAX `Agent`.
    """

    # Action ids
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4  # only present if include_stay=True

    def __init__(
        self,
        shape: Tuple[int, int] = (3, 3),
        walls: Optional[Iterable[Tuple[int, int]]] = None,
        initial_position: Optional[Tuple[int, int] | int] = None,
        include_stay: bool = True,
        success_prob: float = 1.0,
        batch_size: int = 1,
    ):
        rows, cols = shape
        assert rows >= 1 and cols >= 1, "Grid shape must be positive."
        assert 0.0 <= success_prob <= 1.0, "`success_prob` must be in [0, 1]."

        # Precompute data we need
        walls_flat = _flatten_walls(shape, walls)
        n_states = rows * cols
        n_actions = 4 + int(include_stay)

        # --- Build A (likelihood), B (transitions), D (initial state) ---
        A, A_deps = _generate_A(n_states)
        B, B_deps = _generate_B(shape, walls_flat, include_stay, success_prob)
        D = _generate_D(shape, walls_flat, initial_position)

        # Broadcast all arrays to batch dimension like other JAX envs
        expand = lambda x: jnp.broadcast_to(jnp.array(x), (batch_size,) + x.shape)
        params = {
            "A": jtu.tree_map(expand, [A]),
            "B": jtu.tree_map(expand, [B]),
            "D": jtu.tree_map(expand, [D]),
        }
        dependencies = {"A": A_deps, "B": B_deps}

        super().__init__(params, dependencies)

    # ---------------------------------------------------------------------
    # Optional convenience accessors (not required by the API)
    # ---------------------------------------------------------------------
    @staticmethod
    def coords_to_index(shape: Tuple[int, int], coord: Tuple[int, int]) -> int:
        """Convert (row, col) -> linear index in 0..rows*cols-1."""
        return int(jnp.ravel_multi_index(jnp.array(coord), jnp.array(shape)))

    @staticmethod
    def index_to_coords(shape: Tuple[int, int], idx: int) -> Tuple[int, int]:
        """Convert linear index -> (row, col)."""
        return tuple(map(int, jnp.unravel_index(jnp.array(idx), jnp.array(shape))))


# =============================================================================
# Helpers to build A, B, D
# =============================================================================

def _flatten_walls(shape: Tuple[int, int], walls: Optional[Iterable[Tuple[int, int]]]) -> set[int]:
    """Return a set of flattened indices that are blocked."""
    if walls is None:
        return set()
    rows, cols = shape
    walls_flat = set()
    for (r, c) in walls:
        if 0 <= r < rows and 0 <= c < cols:
            walls_flat.add(int(jnp.ravel_multi_index(jnp.array((r, c)), jnp.array(shape))))
    return walls_flat


def _generate_A(n_states: int):
    """
    Identity observation over location.
    Returns:
        A: (num_obs, num_states) where num_obs == n_states
        A_dependencies: [[0]]  (depends only on location state factor)
    """
    A = jnp.eye(n_states)  # observe exact location
    A_dependencies = [[0]]
    return A, A_dependencies


def _neighbors(shape: Tuple[int, int], s: int) -> Tuple[int, int, int, int]:
    """Return potential neighbor indices for UP, RIGHT, DOWN, LEFT (no validity checks)."""
    rows, cols = shape
    r, c = divmod(s, cols)
    up = r - 1, c
    right = r, c + 1
    down = r + 1, c
    left = r, c - 1
    # Convert to flattened, but keep invalid ones negative for now (we’ll validate after)
    to_idx = lambda rc: (
        int(jnp.ravel_multi_index(jnp.array(rc), jnp.array(shape))) if (0 <= rc[0] < rows and 0 <= rc[1] < cols) else -1
    )
    return tuple(map(to_idx, (up, right, down, left)))


def _generate_B(
    shape: Tuple[int, int],
    walls_flat: set[int],
    include_stay: bool,
    success_prob: float,
):
    """
    Build transition tensor B with shape (num_states, num_states, num_actions):
        B[next_state, current_state, action]
    - Movement actions that target invalid cells or walls result in staying in place.
    - If success_prob < 1, the remainder (1 - success_prob) stays in place.
    """
    rows, cols = shape
    n_states = rows * cols
    n_actions = 4 + int(include_stay)

    # Initialize
    B = np.zeros((n_states, n_states, n_actions), dtype=float)

    for s in range(n_states):
        # If current state is a wall (shouldn't happen if D avoids it), just reflect self
        stay_state = s
        up_idx, right_idx, down_idx, left_idx = _neighbors(shape, s)

        # Resolve invalid or walled neighbors to "stay"
        nb = [up_idx, right_idx, down_idx, left_idx]
        nb = [ni if (ni >= 0 and ni not in walls_flat) else stay_state for ni in nb]

        # Four movement actions: UP, RIGHT, DOWN, LEFT
        for a, target in enumerate(nb):
            if target == stay_state:
                # invalid move -> always stay
                B[stay_state, s, a] = 1.0
            else:
                if success_prob >= 1.0:
                    B[target, s, a] = 1.0
                else:
                    # move succeeds with prob p; otherwise stay
                    B[target, s, a] = success_prob
                    B[stay_state, s, a] = 1.0 - success_prob

        # Optional STAY action
        if include_stay:
            B[stay_state, s, 4] = 1.0

    # Sanity: columns over next_state must be normalized per (s, a)
    # (they already are by construction above)
    B = jnp.array(B)
    B_dependencies = [[0]]  # depends only on its own factor (location)
    return B, B_dependencies


def _generate_D(
    shape: Tuple[int, int],
    walls_flat: set[int],
    initial_position: Optional[Tuple[int, int] | int],
):
    """
    Initial state distribution over location.
      - If initial_position is given (r,c) or linear index -> one-hot at that state.
      - Else -> uniform over free cells (not walls).
    """
    rows, cols = shape
    n_states = rows * cols
    if initial_position is None:
        # Uniform over free cells
        mask = jnp.ones(n_states, dtype=float)
        if walls_flat:
            idx = jnp.array(sorted(list(walls_flat)))
            mask = mask.at[idx].set(0.0)
        mass = mask.sum()
        # Handle the (rare) degenerate case of all walls (avoid div by zero)
        D = jnp.where(mass > 0, mask / jnp.clip(mass, 1e-16), jnp.ones(n_states) / n_states)
    else:
        if isinstance(initial_position, tuple):
            start_idx = int(jnp.ravel_multi_index(jnp.array(initial_position), jnp.array(shape)))
        else:
            start_idx = int(initial_position)
        assert start_idx not in walls_flat, "initial_position cannot be a wall."
        D = jnp.zeros(n_states)
        D = D.at[start_idx].set(1.0)
    return D