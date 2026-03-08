"""Cue-chaining grid world implemented as a JAX-compatible ``PymdpEnv``.

This module ports the legacy cue-chaining demo environment to the modern
JAX-first environment API. The environment encodes a two-stage epistemic chain:

1. Visiting ``cue1_location`` reveals which second-level cue location is active.
2. Visiting that active second-level cue location reveals the reward condition.
3. Visiting one of the reward locations yields reward or punishment depending on
   the latent reward condition.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from .env import PymdpEnv


class CueChainingEnv(PymdpEnv):
    """Grid-world cue-chaining task using factorized categorical dynamics.

    Hidden state factors
    --------------------
    - factor 0 (location): flattened grid position
    - factor 1 (cue-2 state): which second-level cue location is informative
    - factor 2 (reward condition): which reward location contains reward

    Observation modalities
    ----------------------
    - modality 0 (location): identity observation over location
    - modality 1 (cue-1): null everywhere except at ``cue1_location``, where it
      reveals the active cue-2 state
    - modality 2 (cue-2): null everywhere except at the active cue-2 location,
      where it reveals reward condition
    - modality 3 (reward): null away from reward locations; at reward locations
      yields reward or punishment based on reward condition
    """

    ACTION_LABELS = ("UP", "DOWN", "LEFT", "RIGHT", "STAY")

    def __init__(
        self,
        grid_shape: tuple[int, int] = (5, 7),
        start_location: tuple[int, int] = (0, 0),
        cue1_location: tuple[int, int] = (2, 0),
        cue2_locations: Sequence[tuple[int, int]] = ((0, 2), (1, 3), (3, 3), (4, 2)),
        reward_locations: Sequence[tuple[int, int]] = ((1, 5), (3, 5)),
        cue2_state: int | None = None,
        reward_condition: int | None = None,
        cue2_names: Sequence[str] | None = None,
        reward_condition_names: Sequence[str] | None = None,
        categorical_obs: bool = False,
    ) -> None:
        """Initialize the cue-chaining environment.

        Parameters
        ----------
        grid_shape : tuple[int, int], optional
            Number of rows and columns in the grid.
        start_location : tuple[int, int], optional
            Initial agent location.
        cue1_location : tuple[int, int], optional
            Location of the first cue.
        cue2_locations : sequence[tuple[int, int]], optional
            Candidate locations of the second cue. The latent ``cue2_state``
            selects which one is informative in an episode.
        reward_locations : sequence[tuple[int, int]], optional
            Candidate reward locations. Exactly two are supported, matching the
            ``Cheese``/``Shock`` reward observations.
        cue2_state : int | None, optional
            Fixed initial state for factor 1. If ``None``, use a uniform prior.
        reward_condition : int | None, optional
            Fixed initial state for factor 2. If ``None``, use a uniform prior.
        cue2_names : sequence[str] | None, optional
            Labels for cue-2 states. Defaults to ``("L1", "L2", ...)``.
        reward_condition_names : sequence[str] | None, optional
            Labels for reward conditions. Defaults to ``("TOP", "BOTTOM")``.
        categorical_obs : bool, default=False
            If ``True``, ``reset()`` and ``step()`` emit one-hot categorical
            observation vectors with shape ``(1, num_obs_m)`` for each
            modality. If ``False``, they emit discrete observation indices with
            shape ``(1,)``.
        """
        rows, cols = grid_shape
        if rows <= 0 or cols <= 0:
            raise ValueError("`grid_shape` must contain strictly positive integers.")
        if len(cue2_locations) == 0:
            raise ValueError("`cue2_locations` must contain at least one location.")
        if len(reward_locations) != 2:
            raise ValueError("`reward_locations` must contain exactly two locations.")

        self.grid_shape = tuple(grid_shape)
        self.start_location = tuple(start_location)
        self.cue1_location = tuple(cue1_location)
        self.cue2_locations = tuple(tuple(c) for c in cue2_locations)
        self.reward_locations = tuple(tuple(r) for r in reward_locations)

        for coord in (self.start_location, self.cue1_location):
            self._validate_coord(coord)
        for coord in self.cue2_locations:
            self._validate_coord(coord)
        for coord in self.reward_locations:
            self._validate_coord(coord)

        self.num_location_states = rows * cols
        self.num_cue2_states = len(self.cue2_locations)
        self.num_reward_states = len(self.reward_locations)

        if cue2_state is not None and not (0 <= cue2_state < self.num_cue2_states):
            raise ValueError("`cue2_state` out of bounds for provided `cue2_locations`.")
        if reward_condition is not None and not (0 <= reward_condition < self.num_reward_states):
            raise ValueError("`reward_condition` out of bounds for provided `reward_locations`.")

        if cue2_names is None:
            cue2_names = tuple(f"L{i + 1}" for i in range(self.num_cue2_states))
        if len(cue2_names) != self.num_cue2_states:
            raise ValueError("`cue2_names` length must match `cue2_locations`.")

        if reward_condition_names is None:
            reward_condition_names = ("TOP", "BOTTOM")
        if len(reward_condition_names) != self.num_reward_states:
            raise ValueError(
                "`reward_condition_names` length must match `reward_locations`."
            )

        self.cue2_names = tuple(cue2_names)
        self.reward_condition_names = tuple(reward_condition_names)

        self.location_obs_names = tuple(
            f"({row}, {col})" for row in range(rows) for col in range(cols)
        )
        self.cue1_obs_names = ("Null",) + self.cue2_names
        self.cue2_obs_names = ("Null",) + tuple(
            f"reward_on_{name.lower()}" for name in self.reward_condition_names
        )
        self.reward_obs_names = ("Null", "Cheese", "Shock")

        self.start_index = self.coords_to_index(self.start_location)
        self.cue1_index = self.coords_to_index(self.cue1_location)
        self.cue2_indices = tuple(self.coords_to_index(coord) for coord in self.cue2_locations)
        self.reward_indices = tuple(
            self.coords_to_index(coord) for coord in self.reward_locations
        )

        A, A_dependencies = self._generate_A()
        B, B_dependencies = self._generate_B()
        D = self._generate_D(cue2_state=cue2_state, reward_condition=reward_condition)

        super().__init__(
            A=A,
            B=B,
            D=D,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            categorical_obs=categorical_obs,
        )

    def coords_to_index(self, coord: tuple[int, int]) -> int:
        """Convert ``(row, col)`` coordinates to a flattened location index."""
        self._validate_coord(coord)
        row, col = coord
        return int(row * self.grid_shape[1] + col)

    def index_to_coords(self, index: int) -> tuple[int, int]:
        """Convert a flattened location index to ``(row, col)`` coordinates."""
        if not (0 <= index < self.num_location_states):
            raise ValueError("`index` out of bounds for location state space.")
        return (index // self.grid_shape[1], index % self.grid_shape[1])

    def _validate_coord(self, coord: tuple[int, int]) -> None:
        row, col = coord
        rows, cols = self.grid_shape
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"Coordinate {coord} is outside grid bounds {self.grid_shape}.")

    def _generate_A(self) -> tuple[list[jnp.ndarray], list[list[int]]]:
        num_loc = self.num_location_states
        num_cue2 = self.num_cue2_states
        num_reward = self.num_reward_states

        # m0: location observation
        A_loc = jnp.eye(num_loc, dtype=jnp.float32)

        # m1: cue-1 observation (null + cue2 labels)
        A_cue1 = jnp.zeros((1 + num_cue2, num_loc, num_cue2), dtype=jnp.float32)
        A_cue1 = A_cue1.at[0, :, :].set(1.0)
        for cue_state in range(num_cue2):
            A_cue1 = A_cue1.at[0, self.cue1_index, cue_state].set(0.0)
            A_cue1 = A_cue1.at[1 + cue_state, self.cue1_index, cue_state].set(1.0)

        # m2: cue-2 observation (null + reward condition labels)
        A_cue2 = jnp.zeros((1 + num_reward, num_loc, num_cue2, num_reward), dtype=jnp.float32)
        A_cue2 = A_cue2.at[0, :, :, :].set(1.0)
        for cue_state, cue_loc_idx in enumerate(self.cue2_indices):
            A_cue2 = A_cue2.at[0, cue_loc_idx, cue_state, :].set(0.0)
            for reward_state in range(num_reward):
                A_cue2 = A_cue2.at[1 + reward_state, cue_loc_idx, cue_state, reward_state].set(1.0)

        # m3: reward observation (null, cheese, shock)
        A_reward = jnp.zeros((3, num_loc, num_reward), dtype=jnp.float32)
        A_reward = A_reward.at[0, :, :].set(1.0)
        for reward_loc_state, reward_loc_idx in enumerate(self.reward_indices):
            A_reward = A_reward.at[0, reward_loc_idx, :].set(0.0)
            for reward_state in range(num_reward):
                obs_idx = 1 if reward_loc_state == reward_state else 2
                A_reward = A_reward.at[obs_idx, reward_loc_idx, reward_state].set(1.0)

        A = [A_loc, A_cue1, A_cue2, A_reward]
        A_dependencies = [[0], [0, 1], [0, 1, 2], [0, 2]]
        return A, A_dependencies

    def _generate_B(self) -> tuple[list[jnp.ndarray], list[list[int]]]:
        rows, cols = self.grid_shape
        num_loc = self.num_location_states
        num_actions = len(self.ACTION_LABELS)

        B_loc = jnp.zeros((num_loc, num_loc, num_actions), dtype=jnp.float32)

        for curr_idx in range(num_loc):
            row, col = self.index_to_coords(curr_idx)

            # UP
            up_idx = self.coords_to_index((max(row - 1, 0), col))
            B_loc = B_loc.at[up_idx, curr_idx, 0].set(1.0)

            # DOWN
            down_idx = self.coords_to_index((min(row + 1, rows - 1), col))
            B_loc = B_loc.at[down_idx, curr_idx, 1].set(1.0)

            # LEFT
            left_idx = self.coords_to_index((row, max(col - 1, 0)))
            B_loc = B_loc.at[left_idx, curr_idx, 2].set(1.0)

            # RIGHT
            right_idx = self.coords_to_index((row, min(col + 1, cols - 1)))
            B_loc = B_loc.at[right_idx, curr_idx, 3].set(1.0)

            # STAY
            B_loc = B_loc.at[curr_idx, curr_idx, 4].set(1.0)

        B_cue2 = jnp.eye(self.num_cue2_states, dtype=jnp.float32).reshape(
            self.num_cue2_states, self.num_cue2_states, 1
        )
        B_reward = jnp.eye(self.num_reward_states, dtype=jnp.float32).reshape(
            self.num_reward_states, self.num_reward_states, 1
        )

        B = [B_loc, B_cue2, B_reward]
        B_dependencies = [[0], [1], [2]]
        return B, B_dependencies

    def _generate_D(
        self,
        cue2_state: int | None,
        reward_condition: int | None,
    ) -> list[jnp.ndarray]:
        D_location = jnp.zeros((self.num_location_states,), dtype=jnp.float32)
        D_location = D_location.at[self.start_index].set(1.0)

        if cue2_state is None:
            D_cue2 = jnp.ones((self.num_cue2_states,), dtype=jnp.float32) / self.num_cue2_states
        else:
            D_cue2 = jnp.zeros((self.num_cue2_states,), dtype=jnp.float32)
            D_cue2 = D_cue2.at[cue2_state].set(1.0)

        if reward_condition is None:
            D_reward = (
                jnp.ones((self.num_reward_states,), dtype=jnp.float32) / self.num_reward_states
            )
        else:
            D_reward = jnp.zeros((self.num_reward_states,), dtype=jnp.float32)
            D_reward = D_reward.at[reward_condition].set(1.0)

        return [D_location, D_cue2, D_reward]
