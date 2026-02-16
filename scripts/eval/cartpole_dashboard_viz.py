#!/usr/bin/env python3
"""Run-and-render dashboard GIFs for Gymnax CartPole active inference.

This script reruns a single-seed configuration (using the same hyperparameters
as `cartpole_eval_harness.py`) and emits one stitched GIF over selected
episodes. The dashboard includes:
- Pixel-style CartPole renderer with action arrows and numeric sensor text boxes
- A/B expected-value heatmaps and delta heatmaps
- Column-wise Dirichlet count vectors for A/B (`pA.sum(axis=0)`, `pB.sum(axis=0)`)
- Posterior state history (`qs`) over a trailing window
- D / empirical-prior panel
- Prior preference C-vector panels (raw/prob/both)
- 2x2 predicted-vs-actual observation heatmaps for all modalities
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

from equinox import tree_at
from jax import jit
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory


REPO_ROOT = Path(__file__).resolve().parents[2]

from pymdp.envs.rollout import rollout
from pymdp import inference as pymdp_inference
try:
    # Works when executed from the pymdp repo root.
    from scripts.eval.cartpole_eval_harness import (
        EPS,
        MODALITY_LABELS,
        GymnaxCartPoleEnv,
        HarnessConfig,
        _build_agent,
        _build_model_template,
        _steps_until_done,
        _trim_outcomes,
    )
except ModuleNotFoundError:
    # Works when executed directly from scripts/eval/.
    from cartpole_eval_harness import (
        EPS,
        MODALITY_LABELS,
        GymnaxCartPoleEnv,
        HarnessConfig,
        _build_agent,
        _build_model_template,
        _steps_until_done,
        _trim_outcomes,
    )

ACTION_LABELS = {
    0: "left push",
    1: "right push",
}

DELTA_CMAP = "viridis"


@dataclass
class DashboardEpisodeTrace:
    episode: int
    actions: np.ndarray  # (T,)
    states: Dict[str, np.ndarray]  # x/x_dot/theta/theta_dot/time/done
    qs: np.ndarray  # (T, S), filtering posterior at each step
    # Smoothing snapshots captured at rollout block boundaries.
    qs_patch_end_steps: np.ndarray  # (P,), each is exclusive global step index
    qs_patch_lengths: np.ndarray  # (P,), valid length per patch in qs_patch_values
    qs_patch_values: np.ndarray  # (P, Lmax, S), padded with NaN beyond length
    empirical_prior: np.ndarray | None  # (T, S) if available
    predicted_probs: List[np.ndarray]  # modality m: (T, O_m)
    actual_indices: List[np.ndarray]  # modality m: (T,)
    A: List[np.ndarray]  # modality m: (T, O_m, S)
    pA: List[np.ndarray]  # modality m: (T, O_m, S)
    B: np.ndarray  # (T, S, S, A)
    pB: np.ndarray  # (T, S, S, A)
    # Incremental parameter deltas relative to the most recent learning bout.
    A_bout_delta: List[np.ndarray]  # modality m: (T, O_m, S)
    B_bout_delta: np.ndarray  # (T, S, S, A)


@dataclass
class DashboardRunTrace:
    seed: int
    compile_time_sec: float
    config: HarnessConfig
    selected_episodes: List[int]
    episode_traces: Dict[int, DashboardEpisodeTrace]
    A_initial: List[np.ndarray]  # modality m: (O_m, S)
    B_initial: np.ndarray  # (S, S, A)
    C_vectors: List[np.ndarray]  # modality m: (O_m,)
    D_initial: np.ndarray  # (S,)


@dataclass
class FrameDescriptor:
    episode: int
    step: int
    local_order: int
    is_reset: bool
    jump_from_episode: int | None


def _parse_render_episodes(spec: str, num_episodes: int) -> List[int]:
    if num_episodes <= 0:
        raise ValueError("`num_episodes` must be positive.")

    ordered: List[int] = []
    seen = set()

    def _append_if_new(idx: int):
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)

    tokens = [tok.strip().lower() for tok in spec.split(",") if tok.strip()]
    if not tokens:
        raise ValueError("`--render-episodes` is empty.")

    for token in tokens:
        if token == "last":
            _append_if_new(num_episodes - 1)
            continue

        if "-" in token:
            parts = token.split("-", maxsplit=1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid range token in --render-episodes: '{token}'")
            start = int(parts[0])
            end = int(parts[1])
            if start > end:
                raise ValueError(
                    f"Invalid descending range in --render-episodes: '{token}'"
                )
            if start < 0 or end >= num_episodes:
                raise ValueError(
                    f"Range '{token}' out of bounds for num_episodes={num_episodes}."
                )
            for idx in range(start, end + 1):
                _append_if_new(idx)
            continue

        idx = int(token)
        if idx < 0 or idx >= num_episodes:
            raise ValueError(
                f"Episode index '{idx}' out of bounds for num_episodes={num_episodes}."
            )
        _append_if_new(idx)

    if not ordered:
        raise ValueError("No valid episodes resolved from --render-episodes.")
    return ordered


def _parse_render_episodes_from_available(
    spec: str,
    available_episodes: Sequence[int],
) -> List[int]:
    available_sorted = sorted(set(int(x) for x in available_episodes))
    if not available_sorted:
        raise ValueError("No cached episodes found.")

    available_set = set(available_sorted)
    max_ep = max(available_sorted)

    ordered: List[int] = []
    seen = set()

    def _append_if_new(idx: int):
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)

    tokens = [tok.strip().lower() for tok in spec.split(",") if tok.strip()]
    if not tokens:
        raise ValueError("`--render-episodes` is empty.")

    for token in tokens:
        if token == "last":
            _append_if_new(max_ep)
            continue

        if "-" in token:
            parts = token.split("-", maxsplit=1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid range token in --render-episodes: '{token}'")
            start = int(parts[0])
            end = int(parts[1])
            if start > end:
                raise ValueError(
                    f"Invalid descending range in --render-episodes: '{token}'"
                )
            for idx in range(start, end + 1):
                if idx in available_set:
                    _append_if_new(idx)
            continue

        idx = int(token)
        if idx in available_set:
            _append_if_new(idx)

    if not ordered:
        raise ValueError(
            "No requested episodes exist in cache. "
            f"Available episodes: {available_sorted[:20]}"
            + ("..." if len(available_sorted) > 20 else "")
        )
    return ordered


def _softmax_np(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / max(exp_z.sum(), EPS)


def _obs_indices_from_block(obs_m: np.ndarray, obs_dim: int) -> np.ndarray:
    arr = np.asarray(obs_m)
    if arr.ndim == 1:
        idx = arr.astype(np.int32)
    elif arr.ndim >= 2 and arr.shape[-1] == obs_dim:
        idx = np.argmax(arr, axis=-1).astype(np.int32)
    elif arr.ndim >= 2 and arr.shape[-1] == 1:
        idx = arr[..., 0].astype(np.int32)
    else:
        raise ValueError(
            f"Unsupported observation shape for index extraction: {arr.shape} (obs_dim={obs_dim})"
        )
    return np.clip(idx, 0, obs_dim - 1)


def _safe_prob_rows(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    y = np.clip(y, EPS, 1.0)
    y = y / np.maximum(y.sum(axis=1, keepdims=True), EPS)
    return y


def _draw_column_aligned_counts(ax, counts: np.ndarray, precision: int = 1, y: float = -0.48):
    vals = np.asarray(counts, dtype=np.float64).ravel()
    xform = blended_transform_factory(ax.transData, ax.transAxes)
    for col, value in enumerate(vals):
        ax.text(
            float(col),
            float(y),
            f"{value:.{precision}f}",
            fontsize=5,
            family="monospace",
            va="top",
            ha="center",
            transform=xform,
            clip_on=False,
        )


def _make_output_dir(path_arg: str | None) -> Path:
    if path_arg:
        out_dir = Path(path_arg).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (REPO_ROOT / "artifacts" / f"cartpole_dashboard_viz_{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _pack_qs_patches(
    patch_end_steps: Sequence[int],
    patch_windows: Sequence[np.ndarray],
    hidden_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not patch_end_steps or not patch_windows:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 0, hidden_dim), dtype=np.float32),
        )

    lengths = np.asarray([int(w.shape[0]) for w in patch_windows], dtype=np.int32)
    max_len = int(lengths.max()) if lengths.size else 0
    packed = np.full(
        (len(patch_windows), max_len, hidden_dim),
        np.nan,
        dtype=np.float32,
    )
    for i, w in enumerate(patch_windows):
        l = int(lengths[i])
        if l <= 0:
            continue
        packed[i, :l, :] = np.asarray(w, dtype=np.float32)
    return (
        np.asarray(patch_end_steps, dtype=np.int32),
        lengths,
        packed,
    )


def _compute_incremental_param_delta(
    series: np.ndarray,
    change_tol: float = 1e-9,
) -> np.ndarray:
    """Delta per frame against the parameter snapshot before the current update bout."""
    arr = np.asarray(series, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError(f"Expected time-major array for delta computation, got shape={arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros_like(arr, dtype=np.float32)

    delta = np.zeros_like(arr, dtype=np.float32)
    baseline = np.asarray(arr[0], dtype=np.float32).copy()
    for t in range(1, arr.shape[0]):
        if np.any(np.abs(arr[t] - arr[t - 1]) > change_tol):
            # New bout begins: baseline is the pre-update snapshot.
            baseline = np.asarray(arr[t - 1], dtype=np.float32)
        delta[t] = np.asarray(arr[t], dtype=np.float32) - baseline
    return delta


def _compute_offline_smoothing_patch(
    filtered_qs: np.ndarray,
    actions: np.ndarray,
    B_pre_update: np.ndarray,
    inference_algo: str,
) -> np.ndarray | None:
    """Reconstruct the smoothing marginals used by offline parameter learning."""
    if inference_algo not in ("exact", "ovf"):
        return None

    filtered = np.asarray(filtered_qs, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.int32).reshape(-1)
    if filtered.ndim != 2 or filtered.shape[0] <= 1:
        return None

    actions_tm1 = actions_arr[: filtered.shape[0] - 1]
    if actions_tm1.shape[0] != (filtered.shape[0] - 1):
        return None
    if np.any(actions_tm1 < 0):
        return None

    filtered_post = [jnp.asarray(filtered, dtype=jnp.float32)]
    B_list = [jnp.asarray(B_pre_update, dtype=jnp.float32)]
    past_actions = jnp.asarray(actions_tm1[:, None], dtype=jnp.int32)

    if inference_algo == "exact":
        marginals, _ = pymdp_inference.smoothing_exact(filtered_post, B_list, past_actions)
        smooth = marginals[0]
    else:
        marginals_and_joints = pymdp_inference.smoothing_ovf(filtered_post, B_list, past_actions)
        smooth = marginals_and_joints[0][0]
    smooth_np = np.asarray(smooth, dtype=np.float32)
    if smooth_np.shape != filtered.shape:
        return None
    return smooth_np


def _reconstruct_offline_smoothing_patches_from_series(
    qs: np.ndarray,
    actions: np.ndarray,
    B_series: np.ndarray,
    inference_algo: str,
    change_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backfill smoothing patches from cached per-step traces when patch files are missing."""
    qs_arr = np.asarray(qs, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.int32).reshape(-1)
    B_arr = np.asarray(B_series, dtype=np.float32)

    if qs_arr.ndim != 2 or B_arr.ndim != 4 or qs_arr.shape[0] == 0:
        hidden_dim = int(qs_arr.shape[1]) if (qs_arr.ndim == 2) else 0
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 0, hidden_dim), dtype=np.float32),
        )

    T = int(qs_arr.shape[0])
    if B_arr.shape[0] != T:
        hidden_dim = int(qs_arr.shape[1])
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 0, hidden_dim), dtype=np.float32),
        )

    if T <= 1:
        hidden_dim = int(qs_arr.shape[1])
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 0, hidden_dim), dtype=np.float32),
        )

    B_flat = B_arr.reshape(T, -1)
    changed = np.any(np.abs(B_flat[1:] - B_flat[:-1]) > change_tol, axis=1)
    change_steps = (np.where(changed)[0] + 1).astype(np.int32)
    seg_starts = np.concatenate([np.asarray([0], dtype=np.int32), change_steps], axis=0)
    seg_ends = np.concatenate([change_steps, np.asarray([T], dtype=np.int32)], axis=0)

    patch_end_steps: List[int] = []
    patch_windows: List[np.ndarray] = []
    for start, end in zip(seg_starts.tolist(), seg_ends.tolist()):
        if end <= start:
            continue
        patch = _compute_offline_smoothing_patch(
            filtered_qs=qs_arr[start:end],
            actions=actions_arr[start:end],
            B_pre_update=B_arr[start],
            inference_algo=inference_algo,
        )
        if patch is None or patch.shape[0] == 0:
            continue
        patch_end_steps.append(int(end))
        patch_windows.append(np.asarray(patch, dtype=np.float32))

    return _pack_qs_patches(
        patch_end_steps,
        patch_windows,
        hidden_dim=int(qs_arr.shape[1]),
    )


def _compose_qs_window_for_step(
    trace: DashboardEpisodeTrace,
    t: int,
    win_start: int,
) -> np.ndarray:
    """Build display q(s) window: filtering baseline + causal smoothing overwrites."""
    qs_win = np.asarray(trace.qs[win_start : t + 1], dtype=np.float32).copy()
    if qs_win.size == 0:
        return qs_win

    if trace.qs_patch_end_steps.size == 0:
        return qs_win

    num_patches = int(trace.qs_patch_end_steps.shape[0])
    for i in range(num_patches):
        end_step = int(trace.qs_patch_end_steps[i])  # exclusive
        if end_step <= 0 or (end_step - 1) > t:
            continue
        patch_len = int(trace.qs_patch_lengths[i])
        if patch_len <= 0:
            continue
        patch = trace.qs_patch_values[i, :patch_len, :]
        start_step = end_step - patch_len

        ov_start = max(start_step, win_start)
        ov_end = min(end_step, t + 1)
        if ov_start >= ov_end:
            continue

        src0 = ov_start - start_step
        src1 = ov_end - start_step
        dst0 = ov_start - win_start
        dst1 = ov_end - win_start
        qs_win[dst0:dst1, :] = patch[src0:src1, :]

    return qs_win


def _resolve_seed_cache_dir(trace_cache_path: Path, seed: int) -> Path:
    path = trace_cache_path.expanduser().resolve()
    candidates = []

    if path.is_dir() and path.name == f"seed_{seed}":
        candidates.append(path)
    if path.is_dir():
        candidates.append(path / "trace_cache" / f"seed_{seed}")
        candidates.append(path / f"seed_{seed}")
        if path.name == "trace_cache":
            candidates.append(path / f"seed_{seed}")

    for cand in candidates:
        if cand.is_dir():
            return cand

    if path.is_dir():
        seed_dirs = sorted([p for p in path.glob("seed_*") if p.is_dir()])
        if len(seed_dirs) == 1:
            return seed_dirs[0]

    raise FileNotFoundError(
        f"Could not resolve cache directory for seed={seed} under: {path}"
    )


def _discover_cached_episodes(seed_cache_dir: Path) -> List[int]:
    episodes: List[int] = []
    for ep_dir in seed_cache_dir.glob("episode_*"):
        if not ep_dir.is_dir():
            continue
        token = ep_dir.name.replace("episode_", "")
        try:
            episodes.append(int(token))
        except ValueError:
            continue
    return sorted(set(episodes))


def _capture_single_seed_dashboard_trace(
    seed: int,
    config: HarnessConfig,
    selected_episodes: Sequence[int],
    batch_index: int = 0,
    verbose: bool = True,
) -> DashboardRunTrace:
    if not selected_episodes:
        raise ValueError("selected_episodes must be non-empty")

    env = GymnaxCartPoleEnv(
        num_obs=[
            config.obs_division,
            config.obs_division,
            config.obs_division,
            config.action_division,
        ],
        action_division=config.action_division,
    )
    env_params = env.generate_env_params(batch_size=config.batch_size)
    template = _build_model_template(config, env)
    agent = _build_agent(template, config)
    jit_rollout = jit(rollout, static_argnums=(1, 2))

    if batch_index < 0 or batch_index >= config.batch_size:
        raise ValueError(
            f"batch_index={batch_index} invalid for batch_size={config.batch_size}"
        )

    selected_set = set(selected_episodes)
    max_episode_to_simulate = int(max(selected_episodes))
    n_modalities = len(template.num_obs)
    hidden_dim = int(template.D[0].shape[-1])

    # Warmup compile.
    warmup_steps = max(min(config.rollout_block_size, config.max_steps_per_episode), 1)
    master_key = jr.PRNGKey(seed)
    master_key, warmup_key = jr.split(master_key)
    t0_compile = time.perf_counter()
    _ = jit_rollout(
        agent,
        env,
        warmup_steps - 1,
        warmup_key,
        initial_carry=None,
        env_params=env_params,
    )
    compile_time_sec = time.perf_counter() - t0_compile

    episode_traces: Dict[int, DashboardEpisodeTrace] = {}

    D_initial = agent.D
    pD = [jnp.zeros_like(d) for d in agent.D]
    qs_t0 = [jnp.zeros_like(d) for d in agent.D]

    for episode in range(max_episode_to_simulate + 1):
        if verbose:
            print(
                f"[seed={seed}] episode={episode + 1}/{max_episode_to_simulate + 1} "
                f"(capture={'yes' if episode in selected_set else 'no'})"
            )

        if episode > 0:
            pD = [pd + q0 for pd, q0 in zip(pD, qs_t0)]
            D_learned = [pd / (jnp.sum(pd, axis=1, keepdims=True) + EPS) for pd in pD]
            if config.learn_D:
                agent = tree_at(lambda x: x.D, agent, D_learned)
            else:
                agent = tree_at(lambda x: x.D, agent, D_initial)

        capture_this_episode = episode in selected_set
        first_block = True
        done = False
        step_count = 0
        block_carry = None

        actions_chunks: List[np.ndarray] = []
        qs_chunks: List[np.ndarray] = []
        qs_patch_end_steps: List[int] = []
        qs_patch_windows: List[np.ndarray] = []
        prior_chunks: List[np.ndarray] = []
        pred_prob_chunks: List[List[np.ndarray]] = [[] for _ in range(n_modalities)]
        obs_index_chunks: List[List[np.ndarray]] = [[] for _ in range(n_modalities)]
        A_chunks: List[List[np.ndarray]] = [[] for _ in range(n_modalities)]
        pA_chunks: List[List[np.ndarray]] = [[] for _ in range(n_modalities)]
        B_chunks: List[np.ndarray] = []
        pB_chunks: List[np.ndarray] = []
        state_chunks: Dict[str, List[np.ndarray]] = {
            "x": [],
            "x_dot": [],
            "theta": [],
            "theta_dot": [],
            "time": [],
            "done": [],
        }

        while (not done) and (step_count < config.max_steps_per_episode):
            step_count_before_block = step_count
            block_steps = min(config.rollout_block_size, config.max_steps_per_episode - step_count)
            block_steps = max(block_steps, 1)

            agent_before_block = agent
            master_key, block_key = jr.split(master_key)
            last_block, info_block = jit_rollout(
                agent,
                env,
                block_steps - 1,
                block_key,
                initial_carry=block_carry,
                env_params=env_params,
            )

            agent = last_block["agent"]
            valid_steps, done = _steps_until_done(info_block, last_block)
            history_len = int(last_block["qs"][0].shape[1])
            history_start = max(history_len - valid_steps, 0)

            obs_block = _trim_outcomes(info_block["observation"])
            obs_block = jtu.tree_map(lambda x: x[:, :valid_steps], obs_block)
            qs_block = jtu.tree_map(lambda x: x[:, :valid_steps, :], info_block["qs"])

            if first_block:
                qs_t0 = jtu.tree_map(lambda x: x[:, history_start], last_block["qs"])
                first_block = False

            agent = tree_at(
                lambda x: x.D,
                agent,
                jtu.tree_map(lambda x: x[:, history_start], last_block["qs"]),
            )

            if capture_this_episode:
                if "A" not in info_block or "B" not in info_block:
                    raise RuntimeError(
                        "Rollout info is missing A/B. Ensure agent learning flags for A/B are enabled."
                    )
                if "pA" not in info_block or "pB" not in info_block:
                    raise RuntimeError(
                        "Rollout info is missing pA/pB. Ensure parameter learning traces are available."
                    )

                qs_np = np.asarray(qs_block[0][batch_index], dtype=np.float32)  # (T, S)
                actions_np = np.asarray(
                    info_block["action"][batch_index, :valid_steps, 0],
                    dtype=np.int32,
                )

                A_block = [
                    np.asarray(info_block["A"][m][batch_index, :valid_steps], dtype=np.float32)
                    for m in range(n_modalities)
                ]
                pA_block = [
                    np.asarray(info_block["pA"][m][batch_index, :valid_steps], dtype=np.float32)
                    for m in range(n_modalities)
                ]
                B_block = np.asarray(
                    info_block["B"][0][batch_index, :valid_steps],
                    dtype=np.float32,
                )  # (T, S, S, A)
                pB_block = np.asarray(
                    info_block["pB"][0][batch_index, :valid_steps],
                    dtype=np.float32,
                )  # (T, S, S, A)

                if "empirical_prior" in info_block:
                    prior_block = np.asarray(
                        info_block["empirical_prior"][0][batch_index, :valid_steps],
                        dtype=np.float32,
                    )
                else:
                    d0 = np.asarray(agent_before_block.D[0][batch_index], dtype=np.float32)
                    prior_block = np.repeat(d0[None, :], valid_steps, axis=0)

                pred_probs_block: List[np.ndarray] = []
                obs_indices_block: List[np.ndarray] = []
                for m in range(n_modalities):
                    pred_m = np.einsum("tos,ts->to", A_block[m], qs_np, optimize=True)
                    pred_m = _safe_prob_rows(pred_m).astype(np.float32)
                    pred_probs_block.append(pred_m)

                    obs_idx_m = _obs_indices_from_block(
                        np.asarray(obs_block[m][batch_index, :valid_steps]),
                        template.num_obs[m],
                    )
                    obs_indices_block.append(obs_idx_m.astype(np.int32))

                env_state_block = info_block["env_state"]
                actions_chunks.append(actions_np)
                qs_chunks.append(qs_np)
                step_end_exclusive = step_count_before_block + valid_steps
                if history_len > 1:
                    patch_len = min(history_len, step_end_exclusive)
                    if patch_len > 0:
                        patch_window = np.asarray(
                            last_block["qs"][0][batch_index, -patch_len:, :],
                            dtype=np.float32,
                        )
                        qs_patch_end_steps.append(int(step_end_exclusive))
                        qs_patch_windows.append(patch_window)
                elif (
                    config.learning_mode == "offline"
                    and config.inference_algo in ("exact", "ovf")
                ):
                    patch_window = _compute_offline_smoothing_patch(
                        filtered_qs=qs_np,
                        actions=actions_np,
                        B_pre_update=np.asarray(
                            agent_before_block.B[0][batch_index],
                            dtype=np.float32,
                        ),
                        inference_algo=config.inference_algo,
                    )
                    if patch_window is not None and patch_window.shape[0] > 0:
                        qs_patch_end_steps.append(int(step_end_exclusive))
                        qs_patch_windows.append(np.asarray(patch_window, dtype=np.float32))
                prior_chunks.append(prior_block)
                B_chunks.append(B_block)
                pB_chunks.append(pB_block)
                for m in range(n_modalities):
                    pred_prob_chunks[m].append(pred_probs_block[m])
                    obs_index_chunks[m].append(obs_indices_block[m])
                    A_chunks[m].append(A_block[m])
                    pA_chunks[m].append(pA_block[m])

                for key in state_chunks.keys():
                    arr = np.asarray(
                        getattr(env_state_block, key)[batch_index, :valid_steps]
                    )
                    state_chunks[key].append(arr)

            step_count += valid_steps
            if done:
                break

            block_carry = dict(last_block)
            block_carry["agent"] = agent

        if capture_this_episode:
            patch_end_arr, patch_len_arr, patch_values_arr = _pack_qs_patches(
                qs_patch_end_steps,
                qs_patch_windows,
                hidden_dim=hidden_dim,
            )
            actions_arr = (
                np.concatenate(actions_chunks, axis=0)
                if actions_chunks
                else np.zeros((0,), dtype=np.int32)
            )
            qs_arr = (
                np.concatenate(qs_chunks, axis=0)
                if qs_chunks
                else np.zeros((0, hidden_dim), dtype=np.float32)
            )
            A_arr = [
                (
                    np.concatenate(A_chunks[m], axis=0)
                    if A_chunks[m]
                    else np.zeros((0, template.num_obs[m], hidden_dim), dtype=np.float32)
                )
                for m in range(n_modalities)
            ]
            pA_arr = [
                (
                    np.concatenate(pA_chunks[m], axis=0)
                    if pA_chunks[m]
                    else np.zeros((0, template.num_obs[m], hidden_dim), dtype=np.float32)
                )
                for m in range(n_modalities)
            ]
            B_arr = (
                np.concatenate(B_chunks, axis=0)
                if B_chunks
                else np.zeros((0, hidden_dim, hidden_dim, config.action_division), dtype=np.float32)
            )
            pB_arr = (
                np.concatenate(pB_chunks, axis=0)
                if pB_chunks
                else np.zeros((0, hidden_dim, hidden_dim, config.action_division), dtype=np.float32)
            )
            trace = DashboardEpisodeTrace(
                episode=episode,
                actions=actions_arr,
                states={
                    key: (
                        np.concatenate(chunks, axis=0)
                        if chunks
                        else np.zeros((0,), dtype=np.float32)
                    )
                    for key, chunks in state_chunks.items()
                },
                qs=qs_arr,
                qs_patch_end_steps=patch_end_arr,
                qs_patch_lengths=patch_len_arr,
                qs_patch_values=patch_values_arr,
                empirical_prior=(
                    np.concatenate(prior_chunks, axis=0)
                    if prior_chunks
                    else np.zeros((0, hidden_dim), dtype=np.float32)
                ),
                predicted_probs=[
                    (
                        np.concatenate(pred_prob_chunks[m], axis=0)
                        if pred_prob_chunks[m]
                        else np.zeros((0, template.num_obs[m]), dtype=np.float32)
                    )
                    for m in range(n_modalities)
                ],
                actual_indices=[
                    (
                        np.concatenate(obs_index_chunks[m], axis=0)
                        if obs_index_chunks[m]
                        else np.zeros((0,), dtype=np.int32)
                    )
                    for m in range(n_modalities)
                ],
                A=A_arr,
                pA=pA_arr,
                B=B_arr,
                pB=pB_arr,
                A_bout_delta=[_compute_incremental_param_delta(a) for a in A_arr],
                B_bout_delta=_compute_incremental_param_delta(B_arr),
            )
            episode_traces[episode] = trace

    return DashboardRunTrace(
        seed=seed,
        compile_time_sec=float(compile_time_sec),
        config=config,
        selected_episodes=list(selected_episodes),
        episode_traces=episode_traces,
        A_initial=[np.asarray(a, dtype=np.float32) for a in template.A],
        B_initial=np.asarray(template.B[0], dtype=np.float32),
        C_vectors=[np.asarray(c, dtype=np.float32) for c in template.C],
        D_initial=np.asarray(template.D[0], dtype=np.float32),
    )


def _load_run_trace_from_cache(
    trace_cache_path: Path,
    seed: int,
    selected_episodes: Sequence[int],
) -> DashboardRunTrace:
    seed_cache_dir = _resolve_seed_cache_dir(trace_cache_path, seed)
    available_eps = _discover_cached_episodes(seed_cache_dir)
    if not available_eps:
        raise FileNotFoundError(f"No episode_* folders found in cache: {seed_cache_dir}")

    static_path = seed_cache_dir / "run_static.npz"
    meta_path = seed_cache_dir / "run_meta.json"

    run_meta = {}
    if meta_path.exists():
        with meta_path.open() as f:
            run_meta = json.load(f)
    cfg_dict = run_meta.get("config", {})
    cache_inference_algo = str(cfg_dict.get("inference_algo", "exact"))
    cache_learning_mode = str(cfg_dict.get("learning_mode", "offline"))

    if static_path.exists():
        static = np.load(static_path)
        modality_ids = sorted(
            [
                int(k.replace("A_initial_modality_", ""))
                for k in static.files
                if k.startswith("A_initial_modality_")
            ]
        )
        A_initial = [np.asarray(static[f"A_initial_modality_{m}"], dtype=np.float32) for m in modality_ids]
        C_vectors = [np.asarray(static[f"C_modality_{m}"], dtype=np.float32) for m in modality_ids]
        B_initial = np.asarray(static["B_initial"], dtype=np.float32)
        D_initial = np.asarray(static["D_initial"], dtype=np.float32)
    else:
        # Fallback for older caches without static snapshots.
        ep0_dir = seed_cache_dir / f"episode_{available_eps[0]:04d}"
        a_files = sorted(ep0_dir.glob("A_modality_*.npy"))
        modality_ids = [int(p.stem.replace("A_modality_", "")) for p in a_files]
        A_initial = []
        C_vectors = []
        for m in modality_ids:
            arr = np.load(ep0_dir / f"A_modality_{m}.npy")
            A_initial.append(np.asarray(arr[0], dtype=np.float32))
            # Not recoverable from old cache; keep neutral placeholder.
            C_vectors.append(np.zeros((arr.shape[1],), dtype=np.float32))
        b_arr = np.load(ep0_dir / "B.npy")
        B_initial = np.asarray(b_arr[0], dtype=np.float32)
        if (ep0_dir / "empirical_prior.npy").exists():
            d_arr = np.load(ep0_dir / "empirical_prior.npy")
            D_initial = np.asarray(d_arr[0], dtype=np.float32)
        else:
            D_initial = np.asarray(B_initial[:, 0], dtype=np.float32)
            D_initial = D_initial / max(float(D_initial.sum()), EPS)

    episode_traces: Dict[int, DashboardEpisodeTrace] = {}
    for ep in selected_episodes:
        ep_dir = seed_cache_dir / f"episode_{ep:04d}"
        if not ep_dir.exists():
            continue

        actions = np.load(ep_dir / "actions.npy")
        qs = np.load(ep_dir / "qs.npy")
        qs_patch_end_steps_path = ep_dir / "qs_patch_end_steps.npy"
        qs_patch_lengths_path = ep_dir / "qs_patch_lengths.npy"
        qs_patch_values_path = ep_dir / "qs_patch_values.npy"
        has_patch_files = (
            qs_patch_end_steps_path.exists()
            and qs_patch_lengths_path.exists()
            and qs_patch_values_path.exists()
        )
        states_npz = np.load(ep_dir / "states.npz")
        states = {k: np.asarray(states_npz[k]) for k in states_npz.files}

        prior_path = ep_dir / "empirical_prior.npy"
        empirical_prior = np.load(prior_path) if prior_path.exists() else None

        # Discover modalities from files.
        mod_ids = sorted(
            [
                int(p.stem.replace("predicted_probs_modality_", ""))
                for p in ep_dir.glob("predicted_probs_modality_*.npy")
            ]
        )
        predicted_probs = [
            np.asarray(np.load(ep_dir / f"predicted_probs_modality_{m}.npy"), dtype=np.float32)
            for m in mod_ids
        ]
        actual_indices = [
            np.asarray(np.load(ep_dir / f"actual_indices_modality_{m}.npy"), dtype=np.int32)
            for m in mod_ids
        ]
        A = [
            np.asarray(np.load(ep_dir / f"A_modality_{m}.npy"), dtype=np.float32)
            for m in mod_ids
        ]
        pA = [
            np.asarray(np.load(ep_dir / f"pA_modality_{m}.npy"), dtype=np.float32)
            for m in mod_ids
        ]
        B = np.asarray(np.load(ep_dir / "B.npy"), dtype=np.float32)
        pB = np.asarray(np.load(ep_dir / "pB.npy"), dtype=np.float32)
        if has_patch_files:
            qs_patch_end_steps = np.load(qs_patch_end_steps_path).astype(np.int32)
            qs_patch_lengths = np.load(qs_patch_lengths_path).astype(np.int32)
            qs_patch_values = np.load(qs_patch_values_path).astype(np.float32)
        elif (
            cache_learning_mode == "offline"
            and cache_inference_algo in ("exact", "ovf")
        ):
            qs_patch_end_steps, qs_patch_lengths, qs_patch_values = (
                _reconstruct_offline_smoothing_patches_from_series(
                    qs=qs,
                    actions=actions,
                    B_series=B,
                    inference_algo=cache_inference_algo,
                )
            )
        else:
            hidden_dim = int(qs.shape[1]) if qs.ndim == 2 else 0
            qs_patch_end_steps = np.zeros((0,), dtype=np.int32)
            qs_patch_lengths = np.zeros((0,), dtype=np.int32)
            qs_patch_values = np.zeros((0, 0, hidden_dim), dtype=np.float32)
        A_bout_delta = [_compute_incremental_param_delta(a) for a in A]
        B_bout_delta = _compute_incremental_param_delta(B)

        episode_traces[ep] = DashboardEpisodeTrace(
            episode=ep,
            actions=np.asarray(actions, dtype=np.int32),
            states=states,
            qs=np.asarray(qs, dtype=np.float32),
            qs_patch_end_steps=qs_patch_end_steps,
            qs_patch_lengths=qs_patch_lengths,
            qs_patch_values=qs_patch_values,
            empirical_prior=(
                np.asarray(empirical_prior, dtype=np.float32)
                if empirical_prior is not None
                else None
            ),
            predicted_probs=predicted_probs,
            actual_indices=actual_indices,
            A=A,
            pA=pA,
            B=B,
            pB=pB,
            A_bout_delta=A_bout_delta,
            B_bout_delta=B_bout_delta,
        )

    if not episode_traces:
        raise FileNotFoundError(
            f"No requested cached episodes found under {seed_cache_dir}: {selected_episodes}"
        )

    # Only a subset is needed for rendering; defaults keep compatibility.
    config = HarnessConfig(
        obs_division=int(cfg_dict.get("obs_division", A_initial[0].shape[0])),
        action_division=int(cfg_dict.get("action_division", B_initial.shape[-1])),
        policy_length=int(cfg_dict.get("policy_length", 2)),
        inference_horizon=cfg_dict.get("inference_horizon", None),
        rollout_block_size=int(cfg_dict.get("rollout_block_size", 1)),
        max_steps_per_episode=int(cfg_dict.get("max_steps_per_episode", 500)),
        num_episodes=int(cfg_dict.get("num_episodes", max(available_eps) + 1)),
        inference_algo=str(cfg_dict.get("inference_algo", "exact")),
        learning_mode=str(cfg_dict.get("learning_mode", "offline")),
        learn_D=bool(cfg_dict.get("learn_D", True)),
        include_action_modality=bool(cfg_dict.get("include_action_modality", False)),
        num_iter=int(cfg_dict.get("num_iter", 16)),
        batch_size=int(cfg_dict.get("batch_size", 1)),
    )

    return DashboardRunTrace(
        seed=seed,
        compile_time_sec=float(run_meta.get("compile_time_sec", 0.0)),
        config=config,
        selected_episodes=list(selected_episodes),
        episode_traces=episode_traces,
        A_initial=A_initial,
        B_initial=B_initial,
        C_vectors=C_vectors,
        D_initial=D_initial,
    )


def _validate_dirichlet_count_formulas(run_trace: DashboardRunTrace):
    # Formula checks requested in the plan:
    # count_A_m = pA_t[m].sum(axis=0)
    # count_B_a = pB_t[:, :, a].sum(axis=0)
    for ep in run_trace.selected_episodes:
        trace = run_trace.episode_traces.get(ep)
        if trace is None or trace.actions.size == 0:
            continue

        sample_ts = sorted(
            set(
                [
                    0,
                    trace.actions.size // 2,
                    max(trace.actions.size - 1, 0),
                ]
            )
        )

        for t in sample_ts:
            for m in range(len(trace.pA)):
                counts = trace.pA[m][t].sum(axis=0)
                formula = np.sum(trace.pA[m][t], axis=0)
                np.testing.assert_allclose(counts, formula, rtol=0, atol=0)

            for a in range(trace.pB.shape[-1]):
                counts = trace.pB[t, :, :, a].sum(axis=0)
                formula = np.sum(trace.pB[t, :, :, a], axis=0)
                np.testing.assert_allclose(counts, formula, rtol=0, atol=0)

            # Sanity that expected-value distributions are bounded [0,1].
            for m in range(len(trace.A)):
                if np.any(trace.A[m][t] < -1e-6) or np.any(trace.A[m][t] > 1.0 + 1e-6):
                    raise AssertionError(f"A[{m}] out of [0,1] bounds at episode={ep}, t={t}")
            if np.any(trace.B[t] < -1e-6) or np.any(trace.B[t] > 1.0 + 1e-6):
                raise AssertionError(f"B out of [0,1] bounds at episode={ep}, t={t}")


def _build_frame_descriptors(
    run_trace: DashboardRunTrace,
    frame_stride: int,
    max_frames_per_episode: int,
) -> tuple[List[FrameDescriptor], Dict[int, int]]:
    descriptors: List[FrameDescriptor] = []
    frames_per_episode: Dict[int, int] = {}
    prev_episode: int | None = None

    step = max(1, int(frame_stride))
    max_frames = max(1, int(max_frames_per_episode))

    for ep in run_trace.selected_episodes:
        trace = run_trace.episode_traces.get(ep)
        if trace is None or trace.actions.size == 0:
            frames_per_episode[ep] = 0
            prev_episode = ep
            continue

        upper = int(min(trace.actions.size, max_frames))
        idxs = list(range(0, upper, step))
        frames_per_episode[ep] = len(idxs)

        for i, t in enumerate(idxs):
            jump = None
            if i == 0 and prev_episode is not None and ep != prev_episode + 1:
                jump = prev_episode
            descriptors.append(
                FrameDescriptor(
                    episode=ep,
                    step=t,
                    local_order=i,
                    is_reset=(i == 0),
                    jump_from_episode=jump,
                )
            )
        prev_episode = ep

    return descriptors, frames_per_episode


def _compute_global_delta_limit(run_trace: DashboardRunTrace) -> float:
    max_abs = 0.0
    for ep in run_trace.selected_episodes:
        trace = run_trace.episode_traces.get(ep)
        if trace is None or trace.actions.size == 0:
            continue
        for m in range(len(trace.A_bout_delta)):
            diff = np.abs(trace.A_bout_delta[m])
            if diff.size > 0:
                max_abs = max(max_abs, float(np.max(diff)))
        diff_b = np.abs(trace.B_bout_delta)
        if diff_b.size > 0:
            max_abs = max(max_abs, float(np.max(diff_b)))
    return max(max_abs, 1e-6)


def _render_env_panel_image(
    state: Dict[str, float],
    action: int,
    step: int,
    episode: int,
    width: int = 720,
    height: int = 430,
) -> Image.Image:
    bg = (247, 248, 250)
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    scene_h = int(height * 0.72)
    ground_y = int(scene_h * 0.82)
    draw.line([(0, ground_y), (width, ground_y)], fill=(80, 80, 80), width=3)

    x_threshold = 2.4
    track_half_px = int(width * 0.33)
    x = float(np.clip(state["x"], -x_threshold, x_threshold))
    theta = float(state["theta"])

    cart_x = width // 2 + int((x / x_threshold) * track_half_px)
    cart_w, cart_h = 95, 40
    cart_y = ground_y - 14
    cart_box = [
        cart_x - cart_w // 2,
        cart_y - cart_h // 2,
        cart_x + cart_w // 2,
        cart_y + cart_h // 2,
    ]
    draw.rectangle(cart_box, fill=(65, 112, 176), outline=(20, 20, 20), width=2)

    pole_len = 140
    pivot_x = cart_x
    pivot_y = cart_y - cart_h // 2 + 2
    end_x = pivot_x + int(pole_len * np.sin(theta))
    end_y = pivot_y - int(pole_len * np.cos(theta))
    draw.line([(pivot_x, pivot_y), (end_x, end_y)], fill=(200, 66, 60), width=9)
    draw.ellipse([pivot_x - 6, pivot_y - 6, pivot_x + 6, pivot_y + 6], fill=(25, 25, 25))

    draw.text((18, 17), f"episode={episode}  step={step}  action={action}", fill=(18, 18, 18))

    # Numeric modality text boxes.
    theta_deg = float(theta * 180.0 / math.pi)
    value_lines = [
        f"x = {float(state['x']):+0.4f}",
        f"x_dot = {float(state['x_dot']):+0.4f}",
        f"theta = {theta:+0.4f} rad ({theta_deg:+0.2f} deg)",
        f"theta_dot = {float(state['theta_dot']):+0.4f}",
        f"done = {bool(state['done'])}",
    ]
    y0 = int(scene_h * 0.05) + 36
    for i, line in enumerate(value_lines):
        y = y0 + i * 27
        draw.text((18, y + 5), line, fill=(22, 22, 22))

    # Action arrows below scene.
    arrow_top = scene_h + 18
    arrow_h = 74
    left_box = [int(width * 0.22), arrow_top, int(width * 0.43), arrow_top + arrow_h]
    right_box = [int(width * 0.57), arrow_top, int(width * 0.78), arrow_top + arrow_h]

    left_active = action == 0
    right_active = action == 1
    draw.rectangle(
        left_box,
        fill=(137, 214, 155) if left_active else (226, 226, 226),
        outline=(70, 70, 70),
        width=2,
    )
    draw.rectangle(
        right_box,
        fill=(137, 214, 155) if right_active else (226, 226, 226),
        outline=(70, 70, 70),
        width=2,
    )

    ly0, ly1 = left_box[1], left_box[3]
    lx0, lx1 = left_box[0], left_box[2]
    left_arrow = [
        (lx0 + 18, (ly0 + ly1) // 2),
        (lx0 + 42, ly0 + 13),
        (lx0 + 42, ly0 + 29),
        (lx1 - 16, ly0 + 29),
        (lx1 - 16, ly1 - 29),
        (lx0 + 42, ly1 - 29),
        (lx0 + 42, ly1 - 13),
    ]
    draw.polygon(left_arrow, fill=(60, 60, 60))

    ry0, ry1 = right_box[1], right_box[3]
    rx0, rx1 = right_box[0], right_box[2]
    right_arrow = [
        (rx1 - 18, (ry0 + ry1) // 2),
        (rx1 - 42, ry0 + 13),
        (rx1 - 42, ry0 + 29),
        (rx0 + 16, ry0 + 29),
        (rx0 + 16, ry1 - 29),
        (rx1 - 42, ry1 - 29),
        (rx1 - 42, ry1 - 13),
    ]
    draw.polygon(right_arrow, fill=(60, 60, 60))

    draw.text((left_box[0] + 8, left_box[3] + 3), "LEFT PUSH (0)", fill=(35, 35, 35))
    draw.text((right_box[0] + 8, right_box[3] + 3), "RIGHT PUSH (1)", fill=(35, 35, 35))
    return img


def _figure_to_image(fig) -> Image.Image:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    image = Image.fromarray(rgba[..., :3])
    plt.close(fig)
    return image


def _render_dashboard_frame(
    run_trace: DashboardRunTrace,
    descriptor: FrameDescriptor,
    effective_window: int,
    delta_limit: float,
    c_view: str,
) -> Image.Image:
    trace = run_trace.episode_traces[descriptor.episode]
    t = descriptor.step
    win_start = max(0, t - effective_window + 1)

    state_t = {k: float(v[t]) for k, v in trace.states.items()}
    action_t = int(trace.actions[t]) if trace.actions.size else -1
    env_img = _render_env_panel_image(
        state=state_t,
        action=action_t,
        step=t,
        episode=descriptor.episode,
    )

    fig = plt.figure(figsize=(21, 12), dpi=110)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.22)

    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 2.45], wspace=0.15)
    ax_env = fig.add_subplot(top[0, 0])
    ax_env.imshow(np.asarray(env_img))
    ax_env.set_title("CartPole Pixel Renderer + Action Arrows + Raw Sensor Values", fontsize=10)
    ax_env.axis("off")

    model_grid = top[0, 1].subgridspec(3, 4, wspace=0.38, hspace=1.05)
    n_modalities = len(trace.A)
    n_actions = trace.B.shape[-1]

    # A panels: 4 modalities -> 2 rows x 2 modality-pairs; each pair (abs, delta).
    for m in range(n_modalities):
        row = m // 2
        col0 = (m % 2) * 2
        ax_abs = fig.add_subplot(model_grid[row, col0])
        ax_delta = fig.add_subplot(model_grid[row, col0 + 1])

        A_t = trace.A[m][t]
        A_delta = trace.A_bout_delta[m][t]
        pA_counts = trace.pA[m][t].sum(axis=0)
        modality_label = MODALITY_LABELS.get(m, f"Modality {m}")

        ax_abs.imshow(A_t, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        ax_abs.set_title(f"A[{m}] ({modality_label}) abs", fontsize=8)
        ax_abs.set_xlabel("hidden state", fontsize=8)
        ax_abs.set_ylabel("obs idx", fontsize=8)
        ax_abs.tick_params(axis="both", labelsize=7)
        ax_abs.tick_params(axis="x", pad=1)
        ax_abs.set_xticks(np.arange(A_t.shape[1]))
        _draw_column_aligned_counts(ax_abs, pA_counts, precision=1)

        ax_delta.imshow(
            A_delta,
            aspect="auto",
            origin="lower",
            cmap=DELTA_CMAP,
            vmin=-delta_limit,
            vmax=delta_limit,
        )
        ax_delta.set_title(f"A[{m}] ({modality_label}) delta (last bout)", fontsize=8)
        ax_delta.set_xlabel("hidden state", fontsize=8)
        ax_delta.set_ylabel("obs idx", fontsize=8)
        ax_delta.tick_params(axis="both", labelsize=7)
        ax_delta.tick_params(axis="x", pad=1)
        ax_delta.set_xticks(np.arange(A_t.shape[1]))

    # B panels: 2 actions (abs + delta) on row 2.
    for a in range(n_actions):
        col0 = a * 2
        ax_abs = fig.add_subplot(model_grid[2, col0])
        ax_delta = fig.add_subplot(model_grid[2, col0 + 1])

        B_t = trace.B[t, :, :, a]
        B_delta = trace.B_bout_delta[t, :, :, a]
        pB_counts = trace.pB[t, :, :, a].sum(axis=0)
        action_label = ACTION_LABELS.get(a, f"action {a}")

        ax_abs.imshow(B_t, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        ax_abs.set_title(f"B[action={a} ({action_label})] abs", fontsize=8)
        ax_abs.set_xlabel("state_t", fontsize=8)
        ax_abs.set_ylabel("state_t+1", fontsize=8)
        ax_abs.tick_params(axis="both", labelsize=7)
        ax_abs.tick_params(axis="x", pad=1)
        ax_abs.set_xticks(np.arange(B_t.shape[1]))
        ax_abs.set_yticks(np.arange(B_t.shape[0]))
        _draw_column_aligned_counts(ax_abs, pB_counts, precision=1)

        ax_delta.imshow(
            B_delta,
            aspect="auto",
            origin="lower",
            cmap=DELTA_CMAP,
            vmin=-delta_limit,
            vmax=delta_limit,
        )
        ax_delta.set_title(f"B[action={a} ({action_label})] delta (last bout)", fontsize=8)
        ax_delta.set_xlabel("state_t", fontsize=8)
        ax_delta.set_ylabel("state_t+1", fontsize=8)
        ax_delta.tick_params(axis="both", labelsize=7)
        ax_delta.tick_params(axis="x", pad=1)
        ax_delta.set_xticks(np.arange(B_t.shape[1]))
        ax_delta.set_yticks(np.arange(B_t.shape[0]))

    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.62], wspace=0.18)
    left_bottom = bottom[0, 0].subgridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.62)

    d_qs_grid = left_bottom[0, 0].subgridspec(1, 2, width_ratios=[0.27, 1.0], wspace=0.08)
    ax_d = fig.add_subplot(d_qs_grid[0, 0])
    ax_qs = fig.add_subplot(d_qs_grid[0, 1], sharey=ax_d)

    c_grid = left_bottom[1, 0].subgridspec(1, len(run_trace.C_vectors), wspace=0.42)
    c_axes = [fig.add_subplot(c_grid[0, i]) for i in range(len(run_trace.C_vectors))]

    hidden_dim = int(trace.qs.shape[1]) if trace.qs.ndim == 2 else 0
    qs_display = _compose_qs_window_for_step(trace, t=t, win_start=win_start)
    qs_win = qs_display.T
    ax_qs.imshow(qs_win, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=1.0)

    has_smoothing_patches = trace.qs_patch_end_steps.size > 0
    qs_title = f"Posterior q(s) | trailing window={effective_window}"
    if has_smoothing_patches:
        qs_title += " | filter + causal smoothing patches"
    else:
        qs_title += " | filtering-only"
    ax_qs.set_title(qs_title, fontsize=10)
    ax_qs.set_xlabel(f"local episode step ({win_start}..{t})", fontsize=8)
    ax_qs.set_ylabel("")
    ax_qs.tick_params(axis="both", labelsize=7)
    if hidden_dim > 0:
        ax_qs.set_yticks(np.arange(hidden_dim))
    if descriptor.is_reset:
        ax_qs.axvline(0, color="cyan", linestyle="--", linewidth=1.2)
    if hidden_dim > 0:
        ax_qs.set_ylim(-0.5, hidden_dim - 0.5)

    # Highlight the active backward-horizon extent within the trailing window.
    horizon = run_trace.config.inference_horizon
    if horizon is not None and int(horizon) > 1 and qs_display.shape[0] > 0:
        horizon_start = max(win_start, t - int(horizon) + 1)
        x0 = float(horizon_start - win_start) - 0.5
        width = float(t - horizon_start + 1)
        if width > 0:
            rect = Rectangle(
                (x0, -0.5),
                width,
                float(hidden_dim),
                fill=False,
                edgecolor="deepskyblue",
                linewidth=1.8,
                linestyle="-",
            )
            ax_qs.add_patch(rect)

    if run_trace.config.learn_D and trace.empirical_prior is not None and trace.empirical_prior.size > 0:
        d_now = trace.empirical_prior[t]
        ax_d.imshow(
            d_now[:, None],
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        ax_d.set_title("D", fontsize=10)
        ax_d.set_xlabel("prior", fontsize=8)
        ax_d.set_ylabel("hidden state idx", fontsize=8)
        ax_d.tick_params(axis="both", labelsize=7)
        ax_d.set_xticks([])
        if hidden_dim > 0:
            ax_d.set_ylim(-0.5, hidden_dim - 0.5)
            ax_d.set_yticks(np.arange(hidden_dim))
    else:
        ax_d.set_title("D", fontsize=10)
        ax_d.text(0.5, 0.5, "disabled", ha="center", va="center", fontsize=10, transform=ax_d.transAxes)
        ax_d.set_xticks([])
        if hidden_dim > 0:
            ax_d.set_ylim(-0.5, hidden_dim - 0.5)
            ax_d.set_yticks(np.arange(hidden_dim))
        ax_d.set_ylabel("hidden state idx", fontsize=8)
        ax_d.tick_params(axis="y", labelsize=7)

    for m, ax in enumerate(c_axes):
        c_raw = run_trace.C_vectors[m]
        c_prob = _softmax_np(c_raw)
        label = MODALITY_LABELS.get(m, f"Modality {m}")
        ax.set_title(f"C[{m}]\n{label}", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)

        if c_view in ("raw", "both"):
            ax.plot(c_raw, color="tab:blue", linewidth=1.2, label="raw C")
            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)

        if c_view == "prob":
            ax.plot(c_prob, color="tab:green", linewidth=1.3, label="softmax(C)")
            ax.set_ylim(0.0, 1.0)
        elif c_view == "both":
            ax2 = ax.twinx()
            ax2.plot(c_prob, color="tab:green", linewidth=1.1, linestyle="--", label="softmax(C)")
            ax2.set_ylim(0.0, 1.0)
            ax2.tick_params(axis="y", labelsize=6)
            if m == 0:
                handles1, labels1 = ax.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(handles1 + handles2, labels1 + labels2, fontsize=6, loc="best")
        elif c_view == "raw":
            if c_raw.size > 0:
                lo = float(np.min(c_raw))
                hi = float(np.max(c_raw))
                if lo == hi:
                    lo -= 1.0
                    hi += 1.0
                ax.set_ylim(lo, hi)
            if m == 0:
                ax.legend(fontsize=6, loc="best")

    obs_grid = bottom[0, 1].subgridspec(2, 2, wspace=0.24, hspace=0.34)
    for m in range(n_modalities):
        ax = fig.add_subplot(obs_grid[m // 2, m % 2])
        pred_win = trace.predicted_probs[m][win_start : t + 1].T
        actual_win = trace.actual_indices[m][win_start : t + 1]
        x_axis = np.arange(actual_win.shape[0], dtype=np.int32)

        ax.imshow(
            pred_win,
            aspect="auto",
            origin="lower",
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
        )
        ax.scatter(x_axis, actual_win, c="tab:blue", s=10, label="actual idx")
        ax.set_title(MODALITY_LABELS.get(m, f"Modality {m}"), fontsize=9)
        ax.set_xlabel("window step", fontsize=8)
        ax.set_ylabel("obs idx", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        if m == 0:
            ax.legend(loc="upper right", fontsize=7)

    suptitle = (
        f"CartPole Active-Inference Dashboard | seed={run_trace.seed} "
        f"| episode={descriptor.episode} | step={descriptor.step} | action={action_t}"
    )
    fig.suptitle(suptitle, fontsize=13, y=0.99)

    if descriptor.local_order < 4:
        if descriptor.is_reset:
            fig.text(
                0.015,
                0.953,
                "RESET",
                fontsize=12,
                color="black",
                bbox=dict(facecolor="gold", edgecolor="black", boxstyle="round,pad=0.25"),
            )
        if descriptor.jump_from_episode is not None:
            fig.text(
                0.11,
                0.953,
                f"jump: ep {descriptor.jump_from_episode} -> ep {descriptor.episode}",
                fontsize=11,
                color="black",
                bbox=dict(facecolor="orange", edgecolor="black", boxstyle="round,pad=0.25"),
            )

    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.03, top=0.95)
    return _figure_to_image(fig)


def _save_trace_cache(
    run_trace: DashboardRunTrace,
    out_dir: Path,
):
    cache_root = out_dir / "trace_cache" / f"seed_{run_trace.seed}"
    cache_root.mkdir(parents=True, exist_ok=True)
    for ep in run_trace.selected_episodes:
        trace = run_trace.episode_traces.get(ep)
        if trace is None:
            continue
        ep_dir = cache_root / f"episode_{ep:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        np.save(ep_dir / "actions.npy", trace.actions)
        np.save(ep_dir / "qs.npy", trace.qs)
        np.save(ep_dir / "qs_patch_end_steps.npy", trace.qs_patch_end_steps)
        np.save(ep_dir / "qs_patch_lengths.npy", trace.qs_patch_lengths)
        np.save(ep_dir / "qs_patch_values.npy", trace.qs_patch_values)
        if trace.empirical_prior is not None:
            np.save(ep_dir / "empirical_prior.npy", trace.empirical_prior)
        np.savez(ep_dir / "states.npz", **{k: np.asarray(v) for k, v in trace.states.items()})
        np.save(ep_dir / "B.npy", trace.B)
        np.save(ep_dir / "pB.npy", trace.pB)

        for m in range(len(trace.predicted_probs)):
            np.save(ep_dir / f"predicted_probs_modality_{m}.npy", trace.predicted_probs[m])
            np.save(ep_dir / f"actual_indices_modality_{m}.npy", trace.actual_indices[m])
            np.save(ep_dir / f"A_modality_{m}.npy", trace.A[m])
            np.save(ep_dir / f"pA_modality_{m}.npy", trace.pA[m])

    static_items = {
        "B_initial": np.asarray(run_trace.B_initial, dtype=np.float32),
        "D_initial": np.asarray(run_trace.D_initial, dtype=np.float32),
    }
    for m in range(len(run_trace.A_initial)):
        static_items[f"A_initial_modality_{m}"] = np.asarray(run_trace.A_initial[m], dtype=np.float32)
        static_items[f"C_modality_{m}"] = np.asarray(run_trace.C_vectors[m], dtype=np.float32)
    np.savez(cache_root / "run_static.npz", **static_items)

    with (cache_root / "run_meta.json").open("w") as f:
        json.dump(
            {
                "seed": int(run_trace.seed),
                "compile_time_sec": float(run_trace.compile_time_sec),
                "selected_episodes": list(run_trace.selected_episodes),
                "config": asdict(run_trace.config),
            },
            f,
            indent=2,
        )


def _stream_stitched_gif(
    run_trace: DashboardRunTrace,
    descriptors: Sequence[FrameDescriptor],
    out_path: Path,
    fps: int,
    effective_window: int,
    delta_limit: float,
    c_view: str,
) -> None:
    if not descriptors:
        raise ValueError("No frames to render.")

    duration_ms = int(1000 / max(fps, 1))
    total = len(descriptors)

    def _render_at(index: int) -> Image.Image:
        if (index + 1) % 25 == 0 or index == 0 or index + 1 == total:
            print(f"[render] frame {index + 1}/{total}")
        return _render_dashboard_frame(
            run_trace=run_trace,
            descriptor=descriptors[index],
            effective_window=effective_window,
            delta_limit=delta_limit,
            c_view=c_view,
        )

    # Use a single global palette to reduce apparent per-frame color flicker.
    first_frame = _render_at(0).convert(
        "P",
        palette=Image.ADAPTIVE,
        colors=256,
        dither=Image.NONE,
    )

    def _iter_remaining():
        for idx in range(1, total):
            frame = _render_at(idx)
            yield frame.quantize(palette=first_frame, dither=Image.NONE)

    first_frame.save(
        out_path,
        save_all=True,
        append_images=_iter_remaining(),
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone run-and-render dashboard GIF for Gymnax CartPole active inference."
    )

    # Hyperparameter parity with cartpole_eval_harness.py
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--obs-division", type=int, default=64)
    parser.add_argument("--action-division", type=int, default=2)
    parser.add_argument("--policy-length", type=int, default=2)
    parser.add_argument(
        "--inference-horizon",
        type=int,
        default=8,
        help="Set 0 to disable finite horizon.",
    )
    parser.add_argument(
        "--rollout-block-size",
        type=int,
        default=0,
        help="If 0, uses inference_horizon or max-steps-per-episode.",
    )
    parser.add_argument(
        "--inference-algo",
        type=str,
        default="mmp",
        choices=["fpi", "vmp", "mmp", "ovf", "exact"],
    )
    parser.add_argument(
        "--learning-mode",
        type=str,
        default="offline",
        choices=["offline", "online"],
    )
    parser.add_argument("--num-iter", type=int, default=16)

    parser.add_argument("--learn-D", dest="learn_D", action="store_true")
    parser.add_argument("--no-learn-D", dest="learn_D", action="store_false")
    parser.set_defaults(learn_D=True)

    # Visualization-specific arguments
    parser.add_argument(
        "--render-episodes",
        type=str,
        default="last",
        help="Episode spec: supports tokens like '0-5,8,21,last'.",
    )
    parser.add_argument("--window-to-show", type=int, default=32)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-episode", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--c-view",
        type=str,
        choices=["raw", "prob", "both"],
        default="both",
        help="How to visualize C vectors per modality.",
    )
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--save-trace-cache", action="store_true")
    parser.add_argument(
        "--trace-cache-dir",
        type=str,
        default=None,
        help=(
            "Optional path to a previously saved trace cache root, "
            "trace_cache folder, or seed_<id> folder. When provided, "
            "the script loads traces and skips rerunning rollout."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.num_episodes <= 0:
        raise SystemExit("--num-episodes must be positive.")
    if args.max_steps_per_episode <= 0:
        raise SystemExit("--max-steps-per-episode must be positive.")
    if args.window_to_show <= 0:
        raise SystemExit("--window-to-show must be positive.")
    if args.frame_stride <= 0:
        raise SystemExit("--frame-stride must be positive.")
    if args.max_frames_per_episode <= 0:
        raise SystemExit("--max-frames-per-episode must be positive.")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive.")

    inference_horizon = None if args.inference_horizon <= 0 else args.inference_horizon
    rollout_block_size = args.rollout_block_size
    if rollout_block_size <= 0:
        rollout_block_size = (
            inference_horizon if inference_horizon is not None else args.max_steps_per_episode
        )
    rollout_block_size = max(int(rollout_block_size), 1)

    config = HarnessConfig(
        obs_division=args.obs_division,
        action_division=args.action_division,
        policy_length=args.policy_length,
        inference_horizon=inference_horizon,
        rollout_block_size=rollout_block_size,
        max_steps_per_episode=args.max_steps_per_episode,
        num_episodes=args.num_episodes,
        inference_algo=args.inference_algo,
        learning_mode=args.learning_mode,
        learn_D=args.learn_D,
        include_action_modality=False,
        num_iter=args.num_iter,
    )

    effective_window = max(args.window_to_show, inference_horizon or 1)
    out_dir = _make_output_dir(args.output_dir)
    gifs_dir = out_dir / "gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)

    if args.trace_cache_dir:
        seed_cache_dir = _resolve_seed_cache_dir(Path(args.trace_cache_dir), args.seed)
        available_eps = _discover_cached_episodes(seed_cache_dir)
        try:
            selected_episodes = _parse_render_episodes_from_available(
                args.render_episodes,
                available_episodes=available_eps,
            )
        except Exception as e:
            raise SystemExit(
                f"Invalid --render-episodes '{args.render_episodes}' for cache mode: {e}"
            ) from e
    else:
        try:
            selected_episodes = _parse_render_episodes(args.render_episodes, args.num_episodes)
        except Exception as e:
            raise SystemExit(f"Invalid --render-episodes '{args.render_episodes}': {e}") from e

    if not args.quiet:
        print("=== CartPole Dashboard Viz ===")
        print(f"seed={args.seed}")
        print(f"requested render episodes={args.render_episodes}")
        print(f"resolved render episodes={selected_episodes}")
        print(f"effective window={effective_window}")

    t0 = time.perf_counter()
    if args.trace_cache_dir:
        run_trace = _load_run_trace_from_cache(
            trace_cache_path=Path(args.trace_cache_dir),
            seed=args.seed,
            selected_episodes=selected_episodes,
        )
    else:
        run_trace = _capture_single_seed_dashboard_trace(
            seed=args.seed,
            config=config,
            selected_episodes=selected_episodes,
            batch_index=args.batch_index,
            verbose=not args.quiet,
        )
    capture_runtime_sec = time.perf_counter() - t0

    _validate_dirichlet_count_formulas(run_trace)

    descriptors, frames_per_episode = _build_frame_descriptors(
        run_trace=run_trace,
        frame_stride=args.frame_stride,
        max_frames_per_episode=args.max_frames_per_episode,
    )
    if not descriptors:
        raise SystemExit("No frames available for selected episodes after stride/frame limits.")

    delta_limit = _compute_global_delta_limit(run_trace)

    gif_path = (
        gifs_dir
        / f"cartpole_dashboard_seed_{args.seed}_episodes_{selected_episodes[0]:04d}_to_{selected_episodes[-1]:04d}.gif"
    )
    t1 = time.perf_counter()
    _stream_stitched_gif(
        run_trace=run_trace,
        descriptors=descriptors,
        out_path=gif_path,
        fps=args.fps,
        effective_window=effective_window,
        delta_limit=delta_limit,
        c_view=args.c_view,
    )
    render_runtime_sec = time.perf_counter() - t1

    if args.save_trace_cache:
        _save_trace_cache(run_trace, out_dir)

    metadata = {
        "seed": int(args.seed),
        "config": asdict(config),
        "requested_render_episodes": str(args.render_episodes),
        "resolved_render_episodes": list(selected_episodes),
        "frames_per_episode": {str(k): int(v) for k, v in frames_per_episode.items()},
        "total_frames": int(len(descriptors)),
        "fps": int(args.fps),
        "frame_stride": int(args.frame_stride),
        "max_frames_per_episode": int(args.max_frames_per_episode),
        "window_to_show": int(args.window_to_show),
        "effective_window": int(effective_window),
        "c_view": str(args.c_view),
        "batch_index": int(args.batch_index),
        "compile_time_sec": float(run_trace.compile_time_sec),
        "capture_runtime_sec": float(capture_runtime_sec),
        "render_runtime_sec": float(render_runtime_sec),
        "output_gif": str(gif_path),
    }
    metadata_path = out_dir / "dashboard_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Dashboard GIF Complete ===")
    print(f"output_dir={out_dir}")
    print(f"gif_path={gif_path}")
    print(f"metadata_path={metadata_path}")
    print(f"total_frames={len(descriptors)}")
    print(f"compile_time_sec={run_trace.compile_time_sec:.3f}")
    print(f"capture_runtime_sec={capture_runtime_sec:.3f}")
    print(f"render_runtime_sec={render_runtime_sec:.3f}")


if __name__ == "__main__":
    main()
