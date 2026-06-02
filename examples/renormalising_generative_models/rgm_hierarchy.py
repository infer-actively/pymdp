"""RGMHierarchy: unified N-level Renormalised Generative Model.

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

import equinox as eqx

from pymdp.agent import Agent

from discretise import (
    DiscretiseConfig,
    OverlappingSVDBasis,
    compute_svd_basis_overlapping,
    encode_images_overlapping,
    decode_observations_overlapping,
)

from state_stats import compute_group_states, compute_hierarchical_state_stats
from agent_build import RGMLevel, _build_agent, _infer_map_states
from mutual_information import compute_level_mi
from group_agents import create_group_agents, _raw_obs_to_group_obs_list
from hierarchical import create_hierarchical_agents, infer_hierarchical_states
from classification import create_classification_agent, _classify_batch
from generation import _generate_image_distributional, _expand_grid, _expand_to_obs
from training import _TrainCarry, _train_step


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
