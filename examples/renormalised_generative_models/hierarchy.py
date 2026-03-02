"""Hierarchical agents for Renormalised Generative Models.

Level 1: batched pymdp Agent (one per patch location) mapping discrete SVD
observations to learned patch states.

Levels 2+: generic hierarchical agents that pool 2x2 blocks of child MAP
states into parent states via unique child-state tuples.

The number of levels is derived automatically from the config:
    n_groups = image_size // group_size
    n_levels = log2(n_groups) + 1
"""

import math

import numpy as np
from jax import numpy as jnp
from typing import NamedTuple

from pymdp.agent import Agent
from pymdp.control import Policies

from discretise import (
    DiscretiseConfig,
    SVDBasis,
    compute_svd_basis,
    encode_images,
    decode_observations,
)


# ---------------------------------------------------------------------------
# State statistics: unique state vocabularies from discrete observations
# ---------------------------------------------------------------------------


class PatchStateStats(NamedTuple):
    num_states: jnp.ndarray  # (n_groups, n_groups) int — unique state count per patch
    state_patterns: list  # [i][j] -> (num_states_ij, max_components) unique observation tuples
    state_to_images: list  # [i][j][s] -> list of image indices mapping to state s


class HierarchicalStateStats(NamedTuple):
    num_states: jnp.ndarray    # (n_parent, n_parent) int — unique state count per parent
    state_patterns: list       # [i][j] -> (num_states_ij, 4) child-state tuples
    state_to_images: list      # [i][j][s] -> list of image indices mapping to state s
    child_num_states: list     # [i][j] -> (4,) num_states of each child in the 2x2 block


def compute_patch_states(observations: jnp.ndarray) -> PatchStateStats:
    """Find unique observation patterns per patch location across structure images.

    Args:
        observations: (N, n_groups, n_groups, max_components) integer bin indices
                      from encode_images()

    Returns:
        PatchStateStats with unique states per patch location
    """
    obs_np = np.asarray(observations)
    _, n_i, n_j, _ = obs_np.shape

    num_states = np.zeros((n_i, n_j), dtype=np.int32)
    state_patterns = [[None for _ in range(n_j)] for _ in range(n_i)]
    state_to_images = [[None for _ in range(n_j)] for _ in range(n_i)]

    for i in range(n_i):
        for j in range(n_j):
            patch_obs = obs_np[:, i, j, :]  # (N, k)
            unique, inverse = np.unique(patch_obs, axis=0, return_inverse=True)
            num_states[i, j] = len(unique)
            state_patterns[i][j] = jnp.array(unique)
            # Map each state to its source image indices
            state_to_images[i][j] = [
                list(np.where(inverse == s)[0]) for s in range(len(unique))
            ]

    return PatchStateStats(
        num_states=jnp.array(num_states),
        state_patterns=state_patterns,
        state_to_images=state_to_images,
    )


def compute_hierarchical_state_stats(
    child_map_states: np.ndarray,
    child_stats: "PatchStateStats | HierarchicalStateStats",
) -> HierarchicalStateStats:
    """Find unique child-state tuples per 2x2 block of children.

    Each parent location pools a 2x2 block of child MAP states. The unique
    4-tuples across all structure images become the parent's state vocabulary.

    Args:
        child_map_states: (N, child_grid, child_grid) MAP state indices at child level
        child_stats: stats from the child level (for reading child num_states)

    Returns:
        HierarchicalStateStats for the parent level
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

    return HierarchicalStateStats(
        num_states=jnp.array(num_states),
        state_patterns=state_patterns,
        state_to_images=state_to_images,
        child_num_states=child_num_states,
    )


class RGMLevel(NamedTuple):
    """One level of the RGM hierarchy."""

    agent: Agent
    valid_mask: jnp.ndarray
    stats: "PatchStateStats | HierarchicalStateStats"


def create_patch_agents(
    stats: PatchStateStats,
    config: DiscretiseConfig,
) -> tuple[Agent, jnp.ndarray]:
    """Build a single batched pymdp Agent for all patch locations.

    Each patch location becomes one element in the batch. All patches share:
    - max_components observation modalities, each with n_levels levels
    - 1 hidden state factor whose size = max unique states across all patches

    Args:
        stats: PatchStateStats from compute_patch_states()
        config: DiscretiseConfig used for discretisation

    Returns:
        (agent, valid_states_mask) where valid_states_mask is (n_patches, max_states)
    """
    n_groups = config.image_size // config.group_size
    n_patches = n_groups * n_groups
    n_mod = config.max_components
    n_levels = config.n_levels

    num_states_flat = stats.num_states.flatten()  # (n_patches,)
    max_states = int(num_states_flat.max())

    # Valid states mask: (n_patches, max_states)
    valid_mask = jnp.arange(max_states)[None, :] < num_states_flat[:, None]

    # --- Build A matrices in NumPy, convert once ---
    # A[m] shape: (n_patches, n_levels, max_states)
    # For valid state s at patch p, modality m: one-hot at the observation level
    # For padded states: uniform 1/n_levels
    A_np = np.full((n_mod, n_patches, n_levels, max_states), 1.0 / n_levels)
    m_idx = np.arange(n_mod)[:, None]  # (n_mod, 1) for broadcasting
    for flat_idx in range(n_patches):
        i, j = divmod(flat_idx, n_groups)
        patterns = np.asarray(stats.state_patterns[i][j])  # (n_s, max_components)
        n_s = patterns.shape[0]
        s_idx = np.arange(n_s)[None, :]  # (1, n_s)
        A_np[:, flat_idx, :, :n_s] = 0.0
        A_np[m_idx, flat_idx, patterns[:n_s, :].T, s_idx] = 1.0
    A = [jnp.array(A_np[m]) for m in range(n_mod)]

    # --- Build B matrix ---
    # Identity transitions, single (trivial) action
    # B[0] shape: (n_patches, max_states, max_states, 1)
    B_eye = jnp.broadcast_to(
        jnp.eye(max_states)[None, :, :, None],
        (n_patches, max_states, max_states, 1),
    )
    B = [B_eye]

    # --- Build D prior ---
    # Uniform over valid states, zero for padded
    D_vals = jnp.where(valid_mask, 1.0, 0.0)
    D_vals = D_vals / D_vals.sum(axis=1, keepdims=True)
    D = [D_vals]

    # --- Trivial policies (no control) ---
    policies = Policies(jnp.zeros((1, 1, 1), dtype=jnp.int32))

    # A_dependencies: each modality depends on factor 0
    A_dependencies = [[0]] * n_mod
    # B_dependencies: factor 0 depends on itself
    B_dependencies = [[0]]

    agent = Agent(
        A=A,
        B=B,
        D=D,
        A_dependencies=A_dependencies,
        B_dependencies=B_dependencies,
        policies=policies,
        num_controls=[1],
        batch_size=n_patches,
        inference_algo="fpi",
        num_iter=16,
    )

    return agent, valid_mask


def infer_patch_states(
    agent: Agent,
    valid_mask: jnp.ndarray,
    observations: jnp.ndarray,
) -> jnp.ndarray:
    """Run L1 inference for a single image.

    Args:
        agent: Batched Agent from create_patch_agents()
        valid_mask: (n_patches, max_states) boolean mask
        observations: (n_groups, n_groups, max_components) single image observations

    Returns:
        (n_groups, n_groups) MAP state indices
    """
    n_i, n_j, n_mod = observations.shape
    n_patches = n_i * n_j

    # Flatten spatial dims: (n_patches, max_components)
    obs_flat = observations.reshape(n_patches, n_mod)

    # Split into list of modalities, each (n_patches,) integer
    obs_list = [obs_flat[:, m] for m in range(n_mod)]

    # Run inference with D as the prior
    qs = agent.infer_states(obs_list, agent.D)

    # qs is a list with one factor: qs[0] shape (n_patches, 1, max_states)
    beliefs = qs[0][:, 0, :]  # (n_patches, max_states)

    # Mask padded states with -inf before argmax
    masked = jnp.where(valid_mask, beliefs, -jnp.inf)
    map_states = jnp.argmax(masked, axis=1)  # (n_patches,)

    return map_states.reshape(n_i, n_j)


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
    stats: HierarchicalStateStats,
    child_stats: "PatchStateStats | HierarchicalStateStats",
) -> tuple[Agent, jnp.ndarray]:
    """Build a batched pymdp Agent for one hierarchical level.

    Each parent location becomes one batch element with 4 observation
    modalities (one per child in its 2x2 block). The A matrix maps parent
    states to child state indices deterministically.

    Args:
        stats: HierarchicalStateStats for this level
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

    # --- Build B matrix ---
    B_eye = jnp.broadcast_to(
        jnp.eye(max_states)[None, :, :, None],
        (n_patches, max_states, max_states, 1),
    )
    B = [B_eye]

    # --- Build D prior ---
    D_vals = jnp.where(valid_mask, 1.0, 0.0)
    D_vals = D_vals / D_vals.sum(axis=1, keepdims=True)
    D = [D_vals]

    # --- Trivial policies ---
    policies = Policies(jnp.zeros((1, 1, 1), dtype=jnp.int32))

    A_dependencies = [[0]] * n_mod
    B_dependencies = [[0]]

    agent = Agent(
        A=A,
        B=B,
        D=D,
        A_dependencies=A_dependencies,
        B_dependencies=B_dependencies,
        policies=policies,
        num_controls=[1],
        batch_size=n_patches,
        inference_algo="fpi",
        num_iter=16,
    )

    return agent, valid_mask


def infer_hierarchical_states(
    agent: Agent,
    valid_mask: jnp.ndarray,
    child_map_states: jnp.ndarray,
) -> jnp.ndarray:
    """Run inference at one hierarchical level for a single image.

    Args:
        agent: Batched Agent from create_hierarchical_agents()
        valid_mask: (n_patches, max_states) boolean mask
        child_map_states: (child_grid, child_grid) MAP state indices at child level

    Returns:
        (parent_grid, parent_grid) MAP state indices at this level
    """
    child_grid = child_map_states.shape[0]
    parent_grid = child_grid // 2

    # Extract 2x2 child observations: (n_patches, 4)
    obs_block = extract_2x2_children(child_map_states)

    # Split into 4 modalities, each (n_patches,) int
    obs_list = [obs_block[:, m] for m in range(4)]

    # Run inference
    qs = agent.infer_states(obs_list, agent.D)

    # qs[0] shape: (n_patches, 1, max_states)
    beliefs = qs[0][:, 0, :]

    # Mask padded states
    masked = jnp.where(valid_mask, beliefs, -jnp.inf)
    map_states = jnp.argmax(masked, axis=1)

    return map_states.reshape(parent_grid, parent_grid)


# ---------------------------------------------------------------------------
# Helper functions for top-down generation
# ---------------------------------------------------------------------------


def _expand_grid(
    parent_grid: np.ndarray,
    stats: HierarchicalStateStats,
) -> np.ndarray:
    """Expand parent state grid to child state grid via state_patterns.

    Each parent cell at (i, j) with state s is expanded into a 2x2 block of
    child states using stats.state_patterns[i][j][s] → (4,) children ordered
    top-left, top-right, bottom-left, bottom-right.

    Args:
        parent_grid: (n, n) parent MAP state indices
        stats: HierarchicalStateStats for this level

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
    l1_stats: PatchStateStats,
    config: DiscretiseConfig,
) -> np.ndarray:
    """Expand L1 state grid to observation grid via state_patterns.

    Args:
        l1_grid: (n_groups, n_groups) L1 MAP state indices
        l1_stats: PatchStateStats for level 1
        config: DiscretiseConfig

    Returns:
        (n_groups, n_groups, max_components) discrete bin indices
    """
    n_groups = config.image_size // config.group_size
    n_comp = config.max_components
    obs_grid = np.zeros((n_groups, n_groups, n_comp), dtype=np.int32)
    for i in range(n_groups):
        for j in range(n_groups):
            s = int(l1_grid[i, j])
            obs_grid[i, j] = np.asarray(l1_stats.state_patterns[i][j][s])
    return obs_grid


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

    levels[0] is the patch level (L1), levels[1:] are hierarchical levels.
    """

    def __init__(
        self,
        config: DiscretiseConfig,
        basis: SVDBasis,
        y_exemplars: np.ndarray,
        levels: list[RGMLevel],
    ):
        self.config = config
        self.basis = basis
        self.y_exemplars = np.asarray(y_exemplars)
        self.levels = levels

        # Build digit_state_map from top-level state_to_images + y_exemplars
        top_stats = levels[-1].stats
        n_top_states = int(top_stats.num_states[0, 0])
        self.digit_state_map = np.full(n_top_states, -1, dtype=np.int32)
        self.state_digit_lists = []
        for s in range(n_top_states):
            img_indices = top_stats.state_to_images[0][0][s]
            digits = list(set(int(self.y_exemplars[idx]) for idx in img_indices))
            self.state_digit_lists.append(digits)
            if len(digits) == 1:
                self.digit_state_map[s] = digits[0]

    @classmethod
    def from_exemplars(
        cls,
        x_exemplars: jnp.ndarray,
        y_exemplars: np.ndarray,
        config: DiscretiseConfig | None = None,
    ) -> "RGMHierarchy":
        """Build the full N-level hierarchy from preprocessed exemplar images.

        The number of levels is derived from the config:
            n_groups = image_size // group_size
            n_levels = log2(n_groups) + 1

        Args:
            x_exemplars: (N, H, W) or (N, C, H, W) preprocessed structure images
            y_exemplars: (N,) digit labels
            config: discretisation config (default: DiscretiseConfig())

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
        basis = compute_svd_basis(x_exemplars, config)
        observations = encode_images(x_exemplars, basis, config)

        # Level 1: patch states
        l1_stats = compute_patch_states(observations)
        l1_agent, l1_valid_mask = create_patch_agents(l1_stats, config)

        # L1 inference on all exemplars
        l1_maps = jnp.stack([
            infer_patch_states(l1_agent, l1_valid_mask, observations[idx])
            for idx in range(len(observations))
        ])

        levels = [RGMLevel(l1_agent, l1_valid_mask, l1_stats)]
        maps = l1_maps
        prev_stats = l1_stats
        grid_size = n_groups

        for _ in range(n_halvings):
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

        return cls(
            config=config,
            basis=basis,
            y_exemplars=np.asarray(y_exemplars),
            levels=levels,
        )

    def classify(
        self, images: jnp.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bottom-up inference: images → digit predictions.

        Args:
            images: (N, H, W) or (N, C, H, W) preprocessed images

        Returns:
            (predicted_digits, top_states) both shape (N,)
            predicted_digits[i] = -1 if the top-level state is ambiguous
        """
        observations = encode_images(images, self.basis, self.config)
        N = observations.shape[0]
        predicted_digits = np.empty(N, dtype=np.int32)
        top_states = np.empty(N, dtype=np.int32)

        for idx in range(N):
            # L1 patch inference
            map_states = infer_patch_states(
                self.levels[0].agent, self.levels[0].valid_mask, observations[idx]
            )
            # Hierarchical levels
            for level in self.levels[1:]:
                map_states = infer_hierarchical_states(
                    level.agent, level.valid_mask, map_states
                )
            s_top = int(map_states[0, 0])
            top_states[idx] = s_top
            predicted_digits[idx] = self.digit_state_map[s_top]

        return predicted_digits, top_states

    def generate(
        self,
        digit: int | None = None,
        prior: np.ndarray | None = None,
        sample: bool = False,
        rng: np.random.Generator | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Top-down generation: digit prior → reconstructed image.

        Traverses state_patterns from the top level down to L1 observations,
        then decodes.

        Args:
            digit: target digit (builds uniform prior over matching top states)
            prior: explicit top-level prior distribution (overrides digit)
            sample: if True, sample top state from prior; else argmax
            rng: numpy random generator for sampling

        Returns:
            (image, observations) where image is (1, C, H, W) and
            observations is (1, n_groups, n_groups, max_components) discrete bin indices
        """
        top_stats = self.levels[-1].stats
        n_top = int(top_stats.num_states[0, 0])

        # Build top-level prior
        if prior is not None:
            top_prior = np.asarray(prior, dtype=np.float64)
        elif digit is not None:
            top_prior = (self.digit_state_map == digit).astype(np.float64)
            if top_prior.sum() == 0:
                raise ValueError(f"No top-level states found for digit {digit}")
            top_prior /= top_prior.sum()
        else:
            top_prior = np.ones(n_top, dtype=np.float64) / n_top

        # Select top-level state
        if sample:
            if rng is None:
                rng = np.random.default_rng()
            s_top = rng.choice(n_top, p=top_prior)
        else:
            s_top = int(np.argmax(top_prior))

        # Top → L1: expand through hierarchical levels in reverse
        grid = np.array([[s_top]], dtype=np.int32)
        for level in reversed(self.levels[1:]):
            grid = _expand_grid(grid, level.stats)

        # L1 → observations
        obs_grid = _expand_to_obs(grid, self.levels[0].stats, self.config)

        obs_jnp = jnp.array(obs_grid)[None]
        image = decode_observations(obs_jnp, self.basis, self.config)

        return image, obs_jnp
