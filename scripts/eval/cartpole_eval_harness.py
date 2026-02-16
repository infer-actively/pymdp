#!/usr/bin/env python3
"""Evaluation harness for Gymnax CartPole + pymdp active inference agents.

This script reproduces the block-wise rollout workflow from:
- examples/envs/CartPole_pymdp_JAX_alpha.ipynb
- examples/envs/gymnax_CartPole_pymdp_JAX_alpha.ipynb

It adds:
1) Multi-seed evaluation and aggregate statistics across runs
2) Episode metrics (Tmax proxy, EFE, inference quality via cross-entropy/accuracy)
3) Plot/CSV/JSON artifact generation
4) Predicted-vs-actual heatmap PNG export
5) Optional gameplay GIF export
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence

from equinox import tree_at
import gymnax
from gymnax.environments.classic_control.cartpole import EnvState as GymnaxEnvState
from jax import jit, lax, nn
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]

from pymdp.agent import Agent
from pymdp.control import compute_expected_obs
from pymdp.envs.env import Env
from pymdp.envs.rollout import rollout
from pymdp.legacy import utils


EPS = 1e-16

MODALITY_LABELS = {
    0: "Cart velocity (x_dot)",
    1: "Pole angle (theta)",
    2: "Pole angular velocity (theta_dot)",
    3: "Action observation (left/right push)",
}


class GymnaxCartPoleState(NamedTuple):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: jnp.ndarray
    done: jnp.ndarray


class GymnaxCartPoleEnv(Env):
    """Gymnax CartPole wrapped to match the pymdp.Env API."""

    def __init__(self, num_obs: Sequence[int], action_division: int = 2):
        self.num_obs = list(num_obs)
        self.num_actions = action_division
        self.env, self._default_env_params = gymnax.make("CartPole-v1")

        self.cart_velocity_max = 2.5 * 2
        self.pole_angle_max = 0.418 / 2 * 2
        self.pole_ang_velocity_max = 3.0 * 2

        self.goal_pole_angle = 0
        self.goal_pole_angle = (
            (self.goal_pole_angle + self.pole_angle_max)
            / (2 * self.pole_angle_max / (self.num_obs[1] - 1))
        )

    def generate_env_params(self, key=None, batch_size=None):
        params = self._default_env_params
        if batch_size is None:
            return params

        def _expand(x):
            x = jnp.asarray(x)
            return jnp.broadcast_to(x, (batch_size,) + x.shape)

        return jtu.tree_map(_expand, params)

    def _discretize(self, x, x_max, n_bins):
        bin_size = (2 * x_max) / (n_bins - 1)
        idx = jnp.rint((x + x_max) / bin_size)
        idx = jnp.clip(idx, 0, n_bins - 1)
        return idx.astype(jnp.int32)

    def _obs_to_onehot(self, obs, action):
        cart_velocity = self._discretize(obs[1], self.cart_velocity_max, self.num_obs[0])
        pole_angle = self._discretize(obs[2], self.pole_angle_max, self.num_obs[1])
        pole_ang_velocity = self._discretize(obs[3], self.pole_ang_velocity_max, self.num_obs[2])
        action = action.astype(jnp.int32)
        return [
            nn.one_hot(cart_velocity, self.num_obs[0])[None, :],
            nn.one_hot(pole_angle, self.num_obs[1])[None, :],
            nn.one_hot(pole_ang_velocity, self.num_obs[2])[None, :],
            nn.one_hot(action, self.num_obs[3])[None, :],
        ]

    def _pack_state(self, state, done):
        return GymnaxCartPoleState(
            x=state.x,
            x_dot=state.x_dot,
            theta=state.theta,
            theta_dot=state.theta_dot,
            time=state.time,
            done=done,
        )

    def _unpack_state(self, state):
        return GymnaxEnvState(
            x=state.x,
            x_dot=state.x_dot,
            theta=state.theta,
            theta_dot=state.theta_dot,
            time=state.time,
        )

    @partial(jit, static_argnums=(0,))
    def reset(self, key, state=None, env_params=None):
        if env_params is None:
            env_params = self._default_env_params
        obs, raw_state = self.env.reset(key, env_params)
        wrapped_state = self._pack_state(raw_state, jnp.array(False))
        return self._obs_to_onehot(obs, jnp.array(0, dtype=jnp.int32)), wrapped_state

    @partial(jit, static_argnums=(0,))
    def step(self, key, state, action, env_params=None):
        if env_params is None:
            env_params = self._default_env_params

        action = jnp.ravel(action)[0].astype(jnp.int32)
        raw_state = self._unpack_state(state)

        obs_prev = self.env.get_obs(raw_state)
        obs_next, raw_state_next, _, done_next, _ = self.env.step_env(
            key, raw_state, action, env_params
        )

        raw_state_out = jtu.tree_map(
            lambda new, old: lax.select(state.done, old, new),
            raw_state_next,
            raw_state,
        )
        obs_out = lax.select(state.done, obs_prev, obs_next)
        done_out = jnp.logical_or(state.done, done_next)

        wrapped_state_out = self._pack_state(raw_state_out, done_out)
        return self._obs_to_onehot(obs_out, action), wrapped_state_out


@dataclass
class HarnessConfig:
    obs_division: int
    action_division: int
    policy_length: int
    inference_horizon: int | None
    rollout_block_size: int
    max_steps_per_episode: int
    num_episodes: int
    inference_algo: str
    learning_mode: str
    learn_D: bool
    include_action_modality: bool
    num_iter: int
    batch_size: int = 1


@dataclass
class ModelTemplate:
    num_obs: List[int]
    num_controls: List[int]
    A: List[np.ndarray]
    B: List[np.ndarray]
    C: List[np.ndarray]
    D: List[np.ndarray]
    pA: List[np.ndarray]
    pB: List[np.ndarray]


@dataclass
class EpisodeTrajectory:
    predicted_probs: List[np.ndarray]  # per modality: (T, O_m)
    actual_indices: List[np.ndarray]  # per modality: (T,)
    actions: np.ndarray  # (T,)
    states: Dict[str, np.ndarray]  # keys: x, x_dot, theta, theta_dot, time, done


@dataclass
class RunResult:
    seed: int
    compile_time_sec: float
    episode_lengths: np.ndarray
    episode_efe_means: np.ndarray
    episode_pred_ce: np.ndarray
    episode_pred_acc: np.ndarray
    episode_pred_cont_l2: np.ndarray
    episode_pred_cont_diracc: np.ndarray
    episode_pred_ce_by_modality: np.ndarray
    episode_pred_acc_by_modality: np.ndarray
    episode_pred_cont_abs_error_by_modality: np.ndarray
    episode_pred_cont_signed_error_by_modality: np.ndarray
    episode_pred_cont_diracc_by_modality: np.ndarray
    episode_runtime_sec: np.ndarray
    episode_steps_per_sec: np.ndarray
    trajectories: Dict[int, EpisodeTrajectory]


def _to_jax_batched(obj_arr, batch_size: int):
    return jtu.tree_map(
        lambda x: jnp.broadcast_to(jnp.asarray(x), (batch_size,) + x.shape),
        list(obj_arr),
    )


def _build_model_template(config: HarnessConfig, env: GymnaxCartPoleEnv) -> ModelTemplate:
    num_obs = [config.obs_division, config.obs_division, config.obs_division, config.action_division]
    num_states = [(len(num_obs) + 1) * 2]
    num_controls = [config.action_division]

    A = utils.random_A_matrix(num_obs, num_states)
    A = utils.norm_dist_obj_arr(A * 0 + 1)

    B = utils.random_B_matrix(num_states, num_controls)
    B = utils.norm_dist_obj_arr(B * 0 + 1)

    C = utils.obj_array_zeros(num_obs)
    x = np.linspace(0, config.obs_division - 1, config.obs_division, dtype=np.float64)
    sigma = 1.7 / (env.pole_angle_max * 180 / np.pi) * (num_obs[1] / 2)
    s = sigma * 0.5513 * 0.25
    z = (x - env.goal_pole_angle) / s
    C[1] = -z - np.log(s) - 2.0 * np.logaddexp(0.0, -z)
    for m in range(len(C)):
        C[m] = np.asarray(nn.log_softmax(jnp.asarray(C[m], dtype=jnp.float32)))

    D = utils.obj_array_uniform(num_states)
    a = 4.0 / ((int(num_states[0]) - 1) / 2)
    x_d = np.linspace(0, int(num_states[0]) - 1, int(num_states[0])) - (int(num_states[0]) - 1) / 2
    D[0] = np.asarray(nn.softmax(jnp.log(1.0 / (1.0 + np.exp(-a * x_d)))))

    pA = utils.dirichlet_like(A, scale=1e-2)
    pB = utils.dirichlet_like(B, scale=1e-2)

    pA = [np.asarray(item, dtype=np.float32) for item in pA]
    pB = [np.asarray(item, dtype=np.float32) for item in pB]

    return ModelTemplate(
        num_obs=num_obs,
        num_controls=num_controls,
        A=list(A),
        B=list(B),
        C=list(C),
        D=list(D),
        pA=list(pA),
        pB=list(pB),
    )


def _build_agent(template: ModelTemplate, config: HarnessConfig) -> Agent:
    inference_horizon = config.inference_horizon
    num_iter = max(config.num_iter, inference_horizon or 1)

    A_jax = _to_jax_batched(template.A, config.batch_size)
    B_jax = _to_jax_batched(template.B, config.batch_size)
    C_jax = _to_jax_batched(template.C, config.batch_size)
    D_jax = _to_jax_batched(template.D, config.batch_size)
    pA_jax = _to_jax_batched(template.pA, config.batch_size)
    pB_jax = _to_jax_batched(template.pB, config.batch_size)

    return Agent(
        A=A_jax,
        B=B_jax,
        C=C_jax,
        D=D_jax,
        pA=pA_jax,
        pB=pB_jax,
        learn_A=True,
        learn_B=True,
        learn_C=False,
        learn_D=config.learn_D,
        learn_E=False,
        inference_algo=config.inference_algo,
        inference_horizon=inference_horizon,
        action_selection="stochastic",
        policy_len=config.policy_length,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=True,
        use_inductive=False,
        categorical_obs=True,
        sampling_mode="full",
        num_iter=num_iter,
        batch_size=config.batch_size,
        learning_mode=config.learning_mode,
    )


def _trim_outcomes(obs_tree):
    def _trim(x):
        if x.ndim >= 4 and x.shape[2] == 1:
            x = jnp.squeeze(x, axis=2)
        return x

    return jtu.tree_map(_trim, obs_tree)


def _steps_until_done(info_block, last_block) -> tuple[int, bool]:
    done_pre = np.asarray(info_block["env_state"].done[0], dtype=bool)
    done_last = bool(np.asarray(last_block["env_state"].done[0]))
    done_next = np.concatenate([done_pre[1:], np.asarray([done_last], dtype=bool)])
    if done_next.any():
        return int(np.argmax(done_next)) + 1, True
    return int(done_next.shape[0]), False


def _obs_to_target_and_index(obs_t: np.ndarray, num_obs_m: int) -> tuple[np.ndarray, int]:
    obs_t = np.asarray(obs_t)
    if obs_t.ndim == 0:
        idx = int(obs_t)
        target = np.zeros(num_obs_m, dtype=np.float64)
        target[idx] = 1.0
        return target, idx

    if obs_t.ndim == 1 and obs_t.shape[0] == num_obs_m:
        idx = int(np.argmax(obs_t))
        target = obs_t.astype(np.float64)
        return target, idx

    if obs_t.ndim == 1 and obs_t.shape[0] == 1:
        idx = int(obs_t[0])
        target = np.zeros(num_obs_m, dtype=np.float64)
        target[idx] = 1.0
        return target, idx

    raise ValueError(f"Unsupported observation shape for modality target conversion: {obs_t.shape}")


def _bin_centers(max_abs: float, n_bins: int) -> np.ndarray:
    return np.linspace(-float(max_abs), float(max_abs), int(n_bins), dtype=np.float64)


def _continuous_bin_centers(env: GymnaxCartPoleEnv, num_obs: Sequence[int]) -> Dict[int, np.ndarray]:
    # Modalities 0..2 are continuous CartPole sensors: x_dot, theta, theta_dot.
    return {
        0: _bin_centers(env.cart_velocity_max, num_obs[0]),
        1: _bin_centers(env.pole_angle_max, num_obs[1]),
        2: _bin_centers(env.pole_ang_velocity_max, num_obs[2]),
    }


def _continuous_actuals_from_env_state(env_state_block, valid_steps: int) -> Dict[int, np.ndarray]:
    return {
        0: np.asarray(env_state_block.x_dot[0, :valid_steps], dtype=np.float64),
        1: np.asarray(env_state_block.theta[0, :valid_steps], dtype=np.float64),
        2: np.asarray(env_state_block.theta_dot[0, :valid_steps], dtype=np.float64),
    }


def _prediction_stats_for_block(
    agent_for_prediction: Agent,
    qs_block,
    obs_block,
    num_obs: Sequence[int],
    continuous_bin_centers: Dict[int, np.ndarray] | None = None,
    continuous_actuals: Dict[int, np.ndarray] | None = None,
) -> dict:
    batch_idx = 0
    num_steps = int(obs_block[0].shape[1])
    num_modalities = len(obs_block)

    ce_by_modality = np.zeros((num_steps, num_modalities), dtype=np.float64)
    acc_by_modality = np.zeros((num_steps, num_modalities), dtype=np.float64)
    cont_error_by_modality = np.full((num_steps, num_modalities), np.nan, dtype=np.float64)
    cont_abs_error_by_modality = np.full((num_steps, num_modalities), np.nan, dtype=np.float64)
    cont_diracc_by_modality = np.full((num_steps, num_modalities), np.nan, dtype=np.float64)
    pred_probs: List[List[np.ndarray]] = [[] for _ in range(num_modalities)]
    obs_indices: List[List[int]] = [[] for _ in range(num_modalities)]

    A_batch = jtu.tree_map(lambda x: jnp.take(x, batch_idx, axis=0), agent_for_prediction.A)

    for t in range(num_steps):
        qs_t = jtu.tree_map(lambda x: x[batch_idx, t], qs_block)
        qo_t = compute_expected_obs(qs_t, A_batch, agent_for_prediction.A_dependencies)

        for m in range(num_modalities):
            pred = np.asarray(qo_t[m], dtype=np.float64)
            pred = np.clip(pred, EPS, 1.0)
            pred = pred / pred.sum()

            target, idx = _obs_to_target_and_index(
                np.asarray(obs_block[m][batch_idx, t]),
                num_obs[m],
            )
            ce = float(-(target * np.log(pred)).sum())
            acc = float(np.argmax(pred) == idx)

            ce_by_modality[t, m] = ce
            acc_by_modality[t, m] = acc
            pred_probs[m].append(pred)
            obs_indices[m].append(idx)

            if (
                (continuous_bin_centers is not None)
                and (continuous_actuals is not None)
                and (m in continuous_bin_centers)
                and (m in continuous_actuals)
            ):
                centers = continuous_bin_centers[m]
                if centers.shape[0] != pred.shape[0]:
                    raise ValueError(
                        f"Bin-center size mismatch for modality {m}: "
                        f"centers={centers.shape[0]} pred={pred.shape[0]}"
                    )
                pred_cont = float(np.dot(pred, centers))
                actual_cont = float(continuous_actuals[m][t])
                err = pred_cont - actual_cont
                cont_error_by_modality[t, m] = err
                cont_abs_error_by_modality[t, m] = abs(err)
                cont_diracc_by_modality[t, m] = float(np.sign(pred_cont) == np.sign(actual_cont))

    pred_probs_arr = [np.stack(per_mod, axis=0) for per_mod in pred_probs]
    obs_indices_arr = [np.asarray(per_mod, dtype=np.int32) for per_mod in obs_indices]

    return {
        "ce_by_modality": ce_by_modality,
        "acc_by_modality": acc_by_modality,
        "cont_error_by_modality": cont_error_by_modality,
        "cont_abs_error_by_modality": cont_abs_error_by_modality,
        "cont_diracc_by_modality": cont_diracc_by_modality,
        "pred_probs": pred_probs_arr,
        "obs_indices": obs_indices_arr,
    }


def _run_single_seed(
    seed: int,
    config: HarnessConfig,
    env: GymnaxCartPoleEnv,
    env_params,
    jit_rollout,
    template: ModelTemplate,
    continuous_bin_centers: Dict[int, np.ndarray],
    capture_episodes: set[int],
    verbose: bool = True,
) -> RunResult:
    agent = _build_agent(template, config)
    D_initial = agent.D

    master_key = jr.PRNGKey(seed)

    warmup_steps = min(config.rollout_block_size, config.max_steps_per_episode)
    warmup_steps = max(warmup_steps, 1)
    master_key, warmup_key = jr.split(master_key)
    t_compile0 = time.perf_counter()
    _ = jit_rollout(
        agent,
        env,
        warmup_steps - 1,
        warmup_key,
        initial_carry=None,
        env_params=env_params,
    )
    compile_time_sec = time.perf_counter() - t_compile0

    episode_lengths: List[int] = []
    episode_efe_means: List[float] = []
    episode_pred_ce: List[float] = []
    episode_pred_acc: List[float] = []
    episode_pred_cont_l2: List[float] = []
    episode_pred_cont_diracc: List[float] = []
    episode_pred_ce_by_modality: List[np.ndarray] = []
    episode_pred_acc_by_modality: List[np.ndarray] = []
    episode_pred_cont_abs_error_by_modality: List[np.ndarray] = []
    episode_pred_cont_signed_error_by_modality: List[np.ndarray] = []
    episode_pred_cont_diracc_by_modality: List[np.ndarray] = []
    episode_runtime_sec: List[float] = []
    episode_steps_per_sec: List[float] = []
    trajectories: Dict[int, EpisodeTrajectory] = {}

    pD = [jnp.zeros_like(d) for d in agent.D]
    qs_t0 = [jnp.zeros_like(d) for d in agent.D]

    metric_modality_indices = list(range(len(template.num_obs)))
    if (not config.include_action_modality) and len(metric_modality_indices) > 0:
        metric_modality_indices = metric_modality_indices[:-1]
    continuous_metric_modality_indices = [
        m for m in metric_modality_indices if m in continuous_bin_centers
    ]

    for episode in range(config.num_episodes):
        if verbose:
            print(f"[seed={seed}] episode={episode + 1}/{config.num_episodes}")

        if episode > 0:
            pD = [pd + q0 for pd, q0 in zip(pD, qs_t0)]
            D_learned = [pd / (jnp.sum(pd, axis=1, keepdims=True) + EPS) for pd in pD]
            if config.learn_D:
                agent = tree_at(lambda x: x.D, agent, D_learned)
            else:
                agent = tree_at(lambda x: x.D, agent, D_initial)

        t_episode0 = time.perf_counter()

        block_carry = None
        done = False
        step_count = 0
        first_block = True

        efe_chunks: List[np.ndarray] = []
        ce_per_modality: List[List[float]] = [[] for _ in range(len(template.num_obs))]
        acc_per_modality: List[List[float]] = [[] for _ in range(len(template.num_obs))]
        cont_abs_error_per_modality: List[List[float]] = [[] for _ in range(len(template.num_obs))]
        cont_signed_error_per_modality: List[List[float]] = [[] for _ in range(len(template.num_obs))]
        cont_diracc_per_modality: List[List[float]] = [[] for _ in range(len(template.num_obs))]
        ce_per_step: List[float] = []
        acc_per_step: List[float] = []
        cont_l2_per_step: List[float] = []
        cont_diracc_per_step: List[float] = []

        capture_episode = episode in capture_episodes
        pred_probs_chunks = [[] for _ in range(len(template.num_obs))]
        obs_indices_chunks = [[] for _ in range(len(template.num_obs))]
        action_chunks: List[np.ndarray] = []
        state_chunks: Dict[str, List[np.ndarray]] = {
            "x": [],
            "x_dot": [],
            "theta": [],
            "theta_dot": [],
            "time": [],
            "done": [],
        }

        while (not done) and (step_count < config.max_steps_per_episode):
            block_steps = min(config.rollout_block_size, config.max_steps_per_episode - step_count)
            block_steps = max(block_steps, 1)

            agent_for_prediction = agent
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
            action_block = np.asarray(info_block["action"][0, :valid_steps, 0], dtype=np.int32)
            G_block = np.asarray(info_block["G"][0, :valid_steps], dtype=np.float64)
            env_state_block = info_block["env_state"]
            continuous_actuals = _continuous_actuals_from_env_state(env_state_block, valid_steps)

            pred_source_agent = (
                agent_for_prediction if config.learning_mode == "offline" else agent
            )
            pred_stats = _prediction_stats_for_block(
                pred_source_agent,
                qs_block,
                obs_block,
                template.num_obs,
                continuous_bin_centers=continuous_bin_centers,
                continuous_actuals=continuous_actuals,
            )

            ce_m = pred_stats["ce_by_modality"]
            acc_m = pred_stats["acc_by_modality"]
            cont_err_m = pred_stats["cont_error_by_modality"]
            cont_abs_m = pred_stats["cont_abs_error_by_modality"]
            cont_diracc_m = pred_stats["cont_diracc_by_modality"]
            for m in range(len(template.num_obs)):
                ce_per_modality[m].extend(ce_m[:, m].tolist())
                acc_per_modality[m].extend(acc_m[:, m].tolist())
                cont_abs_error_per_modality[m].extend(cont_abs_m[:, m].tolist())
                cont_signed_error_per_modality[m].extend(cont_err_m[:, m].tolist())
                cont_diracc_per_modality[m].extend(cont_diracc_m[:, m].tolist())

            if metric_modality_indices:
                ce_step_block = ce_m[:, metric_modality_indices].mean(axis=1)
                acc_step_block = acc_m[:, metric_modality_indices].mean(axis=1)
            else:
                ce_step_block = ce_m.mean(axis=1)
                acc_step_block = acc_m.mean(axis=1)
            ce_per_step.extend(ce_step_block.tolist())
            acc_per_step.extend(acc_step_block.tolist())

            if continuous_metric_modality_indices:
                cont_err_selected = cont_err_m[:, continuous_metric_modality_indices]
                cont_l2_step_block = np.sqrt(np.nansum(cont_err_selected**2, axis=1))
                cont_diracc_step_block = np.nanmean(
                    cont_diracc_m[:, continuous_metric_modality_indices],
                    axis=1,
                )
                cont_l2_per_step.extend(cont_l2_step_block.tolist())
                cont_diracc_per_step.extend(cont_diracc_step_block.tolist())

            efe_block = -G_block.mean(axis=-1) if G_block.ndim > 1 else -G_block
            efe_chunks.append(np.asarray(efe_block, dtype=np.float64))

            if first_block:
                qs_t0 = jtu.tree_map(lambda x: x[:, history_start], last_block["qs"])
                first_block = False

            agent = tree_at(
                lambda x: x.D,
                agent,
                jtu.tree_map(lambda x: x[:, history_start], last_block["qs"]),
            )

            if capture_episode:
                for m in range(len(template.num_obs)):
                    pred_probs_chunks[m].append(pred_stats["pred_probs"][m])
                    obs_indices_chunks[m].append(pred_stats["obs_indices"][m])

                action_chunks.append(action_block)
                for key in state_chunks.keys():
                    state_arr = np.asarray(getattr(env_state_block, key)[0, :valid_steps])
                    state_chunks[key].append(state_arr)

            step_count += valid_steps

            if done:
                break

            block_carry = dict(last_block)
            block_carry["agent"] = agent

        elapsed = time.perf_counter() - t_episode0
        episode_lengths.append(step_count)
        episode_runtime_sec.append(elapsed)
        episode_steps_per_sec.append(step_count / max(elapsed, EPS))

        if efe_chunks:
            episode_efe_means.append(float(np.concatenate(efe_chunks).mean()))
        else:
            episode_efe_means.append(0.0)

        episode_pred_ce.append(float(np.mean(ce_per_step)) if ce_per_step else np.nan)
        episode_pred_acc.append(float(np.mean(acc_per_step)) if acc_per_step else np.nan)
        episode_pred_cont_l2.append(float(np.nanmean(cont_l2_per_step)) if cont_l2_per_step else np.nan)
        episode_pred_cont_diracc.append(
            float(np.nanmean(cont_diracc_per_step)) if cont_diracc_per_step else np.nan
        )

        ce_mod = np.array(
            [float(np.mean(mod_vals)) if mod_vals else np.nan for mod_vals in ce_per_modality],
            dtype=np.float64,
        )
        acc_mod = np.array(
            [float(np.mean(mod_vals)) if mod_vals else np.nan for mod_vals in acc_per_modality],
            dtype=np.float64,
        )
        episode_pred_ce_by_modality.append(ce_mod)
        episode_pred_acc_by_modality.append(acc_mod)
        cont_abs_mod = np.array(
            [_finite_mean(mod_vals) if mod_vals else np.nan for mod_vals in cont_abs_error_per_modality],
            dtype=np.float64,
        )
        cont_signed_mod = np.array(
            [_finite_mean(mod_vals) if mod_vals else np.nan for mod_vals in cont_signed_error_per_modality],
            dtype=np.float64,
        )
        cont_diracc_mod = np.array(
            [_finite_mean(mod_vals) if mod_vals else np.nan for mod_vals in cont_diracc_per_modality],
            dtype=np.float64,
        )
        episode_pred_cont_abs_error_by_modality.append(cont_abs_mod)
        episode_pred_cont_signed_error_by_modality.append(cont_signed_mod)
        episode_pred_cont_diracc_by_modality.append(cont_diracc_mod)

        if capture_episode:
            pred_probs = []
            actual_indices = []
            for m in range(len(template.num_obs)):
                pred_probs_m = (
                    np.concatenate(pred_probs_chunks[m], axis=0)
                    if pred_probs_chunks[m]
                    else np.zeros((0, template.num_obs[m]), dtype=np.float64)
                )
                actual_indices_m = (
                    np.concatenate(obs_indices_chunks[m], axis=0)
                    if obs_indices_chunks[m]
                    else np.zeros((0,), dtype=np.int32)
                )
                pred_probs.append(pred_probs_m)
                actual_indices.append(actual_indices_m)

            actions = (
                np.concatenate(action_chunks, axis=0)
                if action_chunks
                else np.zeros((0,), dtype=np.int32)
            )
            states = {
                key: (
                    np.concatenate(chunks, axis=0)
                    if chunks
                    else np.zeros((0,), dtype=np.float64)
                )
                for key, chunks in state_chunks.items()
            }

            trajectories[episode] = EpisodeTrajectory(
                predicted_probs=pred_probs,
                actual_indices=actual_indices,
                actions=actions,
                states=states,
            )

    return RunResult(
        seed=seed,
        compile_time_sec=float(compile_time_sec),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int32),
        episode_efe_means=np.asarray(episode_efe_means, dtype=np.float64),
        episode_pred_ce=np.asarray(episode_pred_ce, dtype=np.float64),
        episode_pred_acc=np.asarray(episode_pred_acc, dtype=np.float64),
        episode_pred_cont_l2=np.asarray(episode_pred_cont_l2, dtype=np.float64),
        episode_pred_cont_diracc=np.asarray(episode_pred_cont_diracc, dtype=np.float64),
        episode_pred_ce_by_modality=np.asarray(episode_pred_ce_by_modality, dtype=np.float64),
        episode_pred_acc_by_modality=np.asarray(episode_pred_acc_by_modality, dtype=np.float64),
        episode_pred_cont_abs_error_by_modality=np.asarray(
            episode_pred_cont_abs_error_by_modality,
            dtype=np.float64,
        ),
        episode_pred_cont_signed_error_by_modality=np.asarray(
            episode_pred_cont_signed_error_by_modality,
            dtype=np.float64,
        ),
        episode_pred_cont_diracc_by_modality=np.asarray(
            episode_pred_cont_diracc_by_modality,
            dtype=np.float64,
        ),
        episode_runtime_sec=np.asarray(episode_runtime_sec, dtype=np.float64),
        episode_steps_per_sec=np.asarray(episode_steps_per_sec, dtype=np.float64),
        trajectories=trajectories,
    )


def _parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    for token in text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token == "last":
            out.append(-1)
        else:
            out.append(int(token))
    return out


def _resolve_episode_indices(raw_indices: Sequence[int], num_episodes: int) -> set[int]:
    resolved: set[int] = set()
    for idx in raw_indices:
        if idx == -1:
            resolved.add(num_episodes - 1)
        elif 0 <= idx < num_episodes:
            resolved.add(idx)
    return resolved


def _moving_average(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window <= 1:
        return np.arange(1, len(x) + 1), x
    if len(x) < window:
        return np.arange(1, len(x) + 1), x
    kernel = np.ones(window, dtype=np.float64) / float(window)
    y = np.convolve(x, kernel, mode="valid")
    t = np.arange(window, len(x) + 1)
    return t, y


def _finite_mean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(arr.mean())


def _finite_std(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(arr.std())


def _plot_metric_over_episodes(
    run_results: Sequence[RunResult],
    metric_name: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    series = np.stack([getattr(r, metric_name) for r in run_results], axis=0)
    episodes = np.arange(1, series.shape[1] + 1)

    mean = np.nanmean(series, axis=0)
    std = np.nanstd(series, axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    for run in series:
        ax.plot(episodes, run, color="tab:blue", alpha=0.25, linewidth=1.0)
    ax.plot(episodes, mean, color="tab:blue", linewidth=2.0, label="Mean across seeds")
    ax.fill_between(episodes, mean - std, mean + std, color="tab:blue", alpha=0.2, label="±1 std")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_tmax_moving_average(
    run_results: Sequence[RunResult],
    window: int,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    for run in run_results:
        t, y = _moving_average(run.episode_lengths.astype(np.float64), window)
        ax.plot(t, y, alpha=0.25, linewidth=1.0)

    mean_series = np.nanmean(
        np.stack([r.episode_lengths.astype(np.float64) for r in run_results], axis=0),
        axis=0,
    )
    t_mean, y_mean = _moving_average(mean_series, window)
    ax.plot(t_mean, y_mean, color="black", linewidth=2.0, label=f"Mean MA (window={window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Tmax (moving average)")
    ax.set_title("Tmax Per Episode (Moving Average)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_modality_metric(
    run_results: Sequence[RunResult],
    metric_name: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    stacked = np.stack([getattr(r, metric_name) for r in run_results], axis=0)  # (R, E, M)
    num_modalities = stacked.shape[-1]
    episodes = np.arange(1, stacked.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    plotted_any = False
    for m in range(num_modalities):
        mean = np.full(stacked.shape[1], np.nan, dtype=np.float64)
        std = np.full(stacked.shape[1], np.nan, dtype=np.float64)
        for ep in range(stacked.shape[1]):
            vals = stacked[:, ep, m]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            mean[ep] = vals.mean()
            std[ep] = vals.std()
        if not np.isfinite(mean).any():
            continue
        plotted_any = True
        label = MODALITY_LABELS.get(m, f"Modality {m}")
        ax.plot(episodes, mean, linewidth=1.8, label=label)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    if plotted_any:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_run_metrics_csv(run_results: Sequence[RunResult], out_path: Path):
    num_modalities = run_results[0].episode_pred_ce_by_modality.shape[1]
    fields = [
        "seed",
        "episode",
        "tmax",
        "efe_mean",
        "pred_ce_mean",
        "pred_acc_mean",
        "pred_cont_l2_mean",
        "pred_cont_diracc_mean",
        "runtime_sec",
        "steps_per_sec",
    ]
    fields.extend([f"pred_ce_modality_{m}" for m in range(num_modalities)])
    fields.extend([f"pred_acc_modality_{m}" for m in range(num_modalities)])
    fields.extend([f"pred_cont_abs_error_modality_{m}" for m in range(num_modalities)])
    fields.extend([f"pred_cont_signed_error_modality_{m}" for m in range(num_modalities)])
    fields.extend([f"pred_cont_diracc_modality_{m}" for m in range(num_modalities)])

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in run_results:
            for ep in range(len(run.episode_lengths)):
                row = {
                    "seed": run.seed,
                    "episode": ep,
                    "tmax": int(run.episode_lengths[ep]),
                    "efe_mean": float(run.episode_efe_means[ep]),
                    "pred_ce_mean": float(run.episode_pred_ce[ep]),
                    "pred_acc_mean": float(run.episode_pred_acc[ep]),
                    "pred_cont_l2_mean": float(run.episode_pred_cont_l2[ep]),
                    "pred_cont_diracc_mean": float(run.episode_pred_cont_diracc[ep]),
                    "runtime_sec": float(run.episode_runtime_sec[ep]),
                    "steps_per_sec": float(run.episode_steps_per_sec[ep]),
                }
                for m in range(num_modalities):
                    row[f"pred_ce_modality_{m}"] = float(run.episode_pred_ce_by_modality[ep, m])
                    row[f"pred_acc_modality_{m}"] = float(run.episode_pred_acc_by_modality[ep, m])
                    row[f"pred_cont_abs_error_modality_{m}"] = float(
                        run.episode_pred_cont_abs_error_by_modality[ep, m]
                    )
                    row[f"pred_cont_signed_error_modality_{m}"] = float(
                        run.episode_pred_cont_signed_error_by_modality[ep, m]
                    )
                    row[f"pred_cont_diracc_modality_{m}"] = float(
                        run.episode_pred_cont_diracc_by_modality[ep, m]
                    )
                writer.writerow(row)


def _save_aggregate_metrics_csv(run_results: Sequence[RunResult], out_path: Path):
    lengths = np.stack([r.episode_lengths for r in run_results], axis=0).astype(np.float64)
    efe = np.stack([r.episode_efe_means for r in run_results], axis=0)
    ce = np.stack([r.episode_pred_ce for r in run_results], axis=0)
    acc = np.stack([r.episode_pred_acc for r in run_results], axis=0)
    cont_l2 = np.stack([r.episode_pred_cont_l2 for r in run_results], axis=0)
    cont_diracc = np.stack([r.episode_pred_cont_diracc for r in run_results], axis=0)
    runtime = np.stack([r.episode_runtime_sec for r in run_results], axis=0)
    sps = np.stack([r.episode_steps_per_sec for r in run_results], axis=0)
    ce_mod = np.stack([r.episode_pred_ce_by_modality for r in run_results], axis=0)
    acc_mod = np.stack([r.episode_pred_acc_by_modality for r in run_results], axis=0)
    cont_abs_mod = np.stack(
        [r.episode_pred_cont_abs_error_by_modality for r in run_results],
        axis=0,
    )
    cont_signed_mod = np.stack(
        [r.episode_pred_cont_signed_error_by_modality for r in run_results],
        axis=0,
    )
    cont_diracc_mod = np.stack(
        [r.episode_pred_cont_diracc_by_modality for r in run_results],
        axis=0,
    )

    num_episodes = lengths.shape[1]
    num_modalities = ce_mod.shape[2]

    fields = [
        "episode",
        "tmax_mean",
        "tmax_std",
        "efe_mean",
        "efe_std",
        "pred_ce_mean",
        "pred_ce_std",
        "pred_acc_mean",
        "pred_acc_std",
        "pred_cont_l2_mean",
        "pred_cont_l2_std",
        "pred_cont_diracc_mean",
        "pred_cont_diracc_std",
        "runtime_sec_mean",
        "runtime_sec_std",
        "steps_per_sec_mean",
        "steps_per_sec_std",
    ]
    fields.extend([f"pred_ce_modality_{m}_mean" for m in range(num_modalities)])
    fields.extend([f"pred_ce_modality_{m}_std" for m in range(num_modalities)])
    fields.extend([f"pred_acc_modality_{m}_mean" for m in range(num_modalities)])
    fields.extend([f"pred_acc_modality_{m}_std" for m in range(num_modalities)])
    fields.extend([f"pred_cont_abs_error_modality_{m}_mean" for m in range(num_modalities)])
    fields.extend([f"pred_cont_abs_error_modality_{m}_std" for m in range(num_modalities)])
    fields.extend([f"pred_cont_signed_error_modality_{m}_mean" for m in range(num_modalities)])
    fields.extend([f"pred_cont_signed_error_modality_{m}_std" for m in range(num_modalities)])
    fields.extend([f"pred_cont_diracc_modality_{m}_mean" for m in range(num_modalities)])
    fields.extend([f"pred_cont_diracc_modality_{m}_std" for m in range(num_modalities)])

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ep in range(num_episodes):
            row = {
                "episode": ep,
                "tmax_mean": float(np.nanmean(lengths[:, ep])),
                "tmax_std": float(np.nanstd(lengths[:, ep])),
                "efe_mean": float(np.nanmean(efe[:, ep])),
                "efe_std": float(np.nanstd(efe[:, ep])),
                "pred_ce_mean": float(np.nanmean(ce[:, ep])),
                "pred_ce_std": float(np.nanstd(ce[:, ep])),
                "pred_acc_mean": float(np.nanmean(acc[:, ep])),
                "pred_acc_std": float(np.nanstd(acc[:, ep])),
                "pred_cont_l2_mean": _finite_mean(cont_l2[:, ep]),
                "pred_cont_l2_std": _finite_std(cont_l2[:, ep]),
                "pred_cont_diracc_mean": _finite_mean(cont_diracc[:, ep]),
                "pred_cont_diracc_std": _finite_std(cont_diracc[:, ep]),
                "runtime_sec_mean": float(np.nanmean(runtime[:, ep])),
                "runtime_sec_std": float(np.nanstd(runtime[:, ep])),
                "steps_per_sec_mean": float(np.nanmean(sps[:, ep])),
                "steps_per_sec_std": float(np.nanstd(sps[:, ep])),
            }
            for m in range(num_modalities):
                row[f"pred_ce_modality_{m}_mean"] = float(np.nanmean(ce_mod[:, ep, m]))
                row[f"pred_ce_modality_{m}_std"] = float(np.nanstd(ce_mod[:, ep, m]))
                row[f"pred_acc_modality_{m}_mean"] = float(np.nanmean(acc_mod[:, ep, m]))
                row[f"pred_acc_modality_{m}_std"] = float(np.nanstd(acc_mod[:, ep, m]))
                row[f"pred_cont_abs_error_modality_{m}_mean"] = _finite_mean(cont_abs_mod[:, ep, m])
                row[f"pred_cont_abs_error_modality_{m}_std"] = _finite_std(cont_abs_mod[:, ep, m])
                row[f"pred_cont_signed_error_modality_{m}_mean"] = _finite_mean(
                    cont_signed_mod[:, ep, m]
                )
                row[f"pred_cont_signed_error_modality_{m}_std"] = _finite_std(
                    cont_signed_mod[:, ep, m]
                )
                row[f"pred_cont_diracc_modality_{m}_mean"] = _finite_mean(cont_diracc_mod[:, ep, m])
                row[f"pred_cont_diracc_modality_{m}_std"] = _finite_std(cont_diracc_mod[:, ep, m])
            writer.writerow(row)


def _save_prediction_heatmap(
    predicted_probs: np.ndarray,
    actual_indices: np.ndarray,
    out_path: Path,
    title: str,
):
    if predicted_probs.size == 0:
        return
    qo_arr = predicted_probs.T  # (obs_dim, T)
    t = np.arange(qo_arr.shape[1])

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(
        qo_arr,
        aspect="auto",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
    )
    fig.colorbar(im, ax=ax, label="Predicted probability")
    ax.scatter(t, actual_indices, c="tab:blue", s=10, label="Actual observations")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Observation index")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _render_cartpole_frame(state: Dict[str, float], action: int, step: int) -> Image.Image:
    width, height = 640, 360
    bg = (248, 248, 248)
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    ground_y = int(height * 0.78)
    draw.line([(0, ground_y), (width, ground_y)], fill=(80, 80, 80), width=3)

    x_threshold = 2.4
    track_half_px = int(width * 0.33)
    x = float(np.clip(state["x"], -x_threshold, x_threshold))
    theta = float(state["theta"])

    cart_x = width // 2 + int((x / x_threshold) * track_half_px)
    cart_w, cart_h = 80, 36
    cart_y = ground_y - 14
    cart_box = [
        cart_x - cart_w // 2,
        cart_y - cart_h // 2,
        cart_x + cart_w // 2,
        cart_y + cart_h // 2,
    ]
    draw.rectangle(cart_box, fill=(61, 106, 171), outline=(30, 30, 30), width=2)

    pole_len = 130
    pivot_x = cart_x
    pivot_y = cart_y - cart_h // 2 + 2
    end_x = pivot_x + int(pole_len * np.sin(theta))
    end_y = pivot_y - int(pole_len * np.cos(theta))
    draw.line([(pivot_x, pivot_y), (end_x, end_y)], fill=(196, 64, 56), width=8)
    draw.ellipse(
        [pivot_x - 5, pivot_y - 5, pivot_x + 5, pivot_y + 5],
        fill=(25, 25, 25),
    )

    text = f"t={step}  action={action}  x={x:.2f}  theta={theta:.2f}"
    draw.text((12, 12), text, fill=(10, 10, 10))
    return img


def _save_episode_gif(
    trajectory: EpisodeTrajectory,
    out_path: Path,
    fps: int,
    max_frames: int,
) -> bool:
    if trajectory.actions.size == 0:
        return False
    n_frames = int(min(len(trajectory.actions), max_frames))
    if n_frames <= 0:
        return False

    frames: List[Image.Image] = []
    for t in range(n_frames):
        state_t = {key: float(trajectory.states[key][t]) for key in trajectory.states.keys()}
        frame = _render_cartpole_frame(state_t, int(trajectory.actions[t]), t)
        frames.append(frame)

    duration_ms = int(1000 / max(fps, 1))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return True


def _save_trajectory_artifacts(
    run_result: RunResult,
    episodes: Sequence[int],
    modalities: Sequence[int],
    out_dir: Path,
):
    run_dir = out_dir / f"seed_{run_result.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for episode in episodes:
        traj = run_result.trajectories.get(episode)
        if traj is None:
            continue

        ep_dir = run_dir / f"episode_{episode:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        np.save(ep_dir / "actions.npy", traj.actions)
        np.savez(
            ep_dir / "states.npz",
            **{k: np.asarray(v) for k, v in traj.states.items()},
        )

        for m in modalities:
            if m < 0 or m >= len(traj.predicted_probs):
                continue
            pred = traj.predicted_probs[m]
            obs_idx = traj.actual_indices[m]
            np.save(ep_dir / f"predicted_probs_modality_{m}.npy", pred)
            np.save(ep_dir / f"actual_indices_modality_{m}.npy", obs_idx)
            _save_prediction_heatmap(
                pred,
                obs_idx,
                ep_dir / f"pred_vs_actual_modality_{m}.png",
                title=(
                    f"Seed {run_result.seed} Episode {episode} "
                    f"{MODALITY_LABELS.get(m, f'Modality {m}')}"
                ),
            )


def _make_output_dir(path_arg: str | None) -> Path:
    if path_arg:
        out_dir = Path(path_arg).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (REPO_ROOT / "artifacts" / f"cartpole_eval_{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _summary_stats(run_results: Sequence[RunResult]) -> dict:
    tmax = np.stack([r.episode_lengths for r in run_results], axis=0).astype(np.float64)
    ce = np.stack([r.episode_pred_ce for r in run_results], axis=0)
    acc = np.stack([r.episode_pred_acc for r in run_results], axis=0)
    cont_l2 = np.stack([r.episode_pred_cont_l2 for r in run_results], axis=0)
    cont_diracc = np.stack([r.episode_pred_cont_diracc for r in run_results], axis=0)
    sps = np.stack([r.episode_steps_per_sec for r in run_results], axis=0)

    tail = min(25, tmax.shape[1])
    return {
        "num_runs": int(tmax.shape[0]),
        "num_episodes": int(tmax.shape[1]),
        "tmax_mean_all": float(np.nanmean(tmax)),
        "tmax_std_all": float(np.nanstd(tmax)),
        "tmax_mean_last_window": float(np.nanmean(tmax[:, -tail:])),
        "pred_ce_mean_all": float(np.nanmean(ce)),
        "pred_acc_mean_all": float(np.nanmean(acc)),
        "pred_cont_l2_mean_all": float(np.nanmean(cont_l2)),
        "pred_cont_diracc_mean_all": float(np.nanmean(cont_diracc)),
        "steps_per_sec_mean_all": float(np.nanmean(sps)),
        "compile_time_sec_per_seed": {str(r.seed): float(r.compile_time_sec) for r in run_results},
    }


def main():
    parser = argparse.ArgumentParser(description="Gymnax CartPole evaluation harness for PyMDP agents.")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--obs-division", type=int, default=64)
    parser.add_argument("--action-division", type=int, default=2)
    parser.add_argument("--policy-length", type=int, default=2)
    parser.add_argument("--inference-horizon", type=int, default=8, help="Set 0 to disable finite horizon.")
    parser.add_argument(
        "--rollout-block-size",
        type=int,
        default=0,
        help="Block size used in rollout loop. If 0, uses inference_horizon or max-steps-per-episode.",
    )
    parser.add_argument("--inference-algo", type=str, default="mmp", choices=["fpi", "vmp", "mmp", "ovf", "exact"])
    parser.add_argument("--learning-mode", type=str, default="offline", choices=["offline", "online"])
    parser.add_argument("--num-iter", type=int, default=16)

    parser.add_argument("--learn-D", dest="learn_D", action="store_true")
    parser.add_argument("--no-learn-D", dest="learn_D", action="store_false")
    parser.set_defaults(learn_D=True)

    parser.add_argument(
        "--include-action-modality",
        dest="include_action_modality",
        action="store_true",
        help="Include action modality in aggregated inference-quality metrics (default excludes it).",
    )
    parser.add_argument(
        "--exclude-action-modality",
        dest="include_action_modality",
        action="store_false",
    )
    parser.set_defaults(include_action_modality=False)

    parser.add_argument("--moving-average-window", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--save-heatmaps", action="store_true")
    parser.add_argument("--heatmap-run-index", type=int, default=0)
    parser.add_argument("--heatmap-episodes", type=str, default="0,last")
    parser.add_argument("--heatmap-modalities", type=str, default="0,1,2,3")

    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--gif-run-index", type=int, default=0)
    parser.add_argument("--gif-episode", type=str, default="last")
    parser.add_argument("--gif-fps", type=int, default=30)
    parser.add_argument("--gif-max-frames", type=int, default=500)

    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise SystemExit("No seeds provided.")

    inference_horizon = None if args.inference_horizon <= 0 else args.inference_horizon
    rollout_block_size = args.rollout_block_size
    if rollout_block_size <= 0:
        rollout_block_size = inference_horizon if inference_horizon is not None else args.max_steps_per_episode
    rollout_block_size = max(rollout_block_size, 1)

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
        include_action_modality=args.include_action_modality,
        num_iter=args.num_iter,
    )

    out_dir = _make_output_dir(args.output_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir = out_dir / "heatmaps"
    gifs_dir = out_dir / "gifs"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir.mkdir(parents=True, exist_ok=True)

    env = GymnaxCartPoleEnv(
        num_obs=[config.obs_division, config.obs_division, config.obs_division, config.action_division],
        action_division=config.action_division,
    )
    env_params = env.generate_env_params(batch_size=config.batch_size)
    template = _build_model_template(config, env)
    continuous_bin_centers = _continuous_bin_centers(env, template.num_obs)

    run_rollout = jit(rollout, static_argnums=(1, 2))

    if args.save_heatmaps and not (0 <= args.heatmap_run_index < len(seeds)):
        raise SystemExit(f"--heatmap-run-index must be in [0, {len(seeds) - 1}]")
    if args.save_gif and not (0 <= args.gif_run_index < len(seeds)):
        raise SystemExit(f"--gif-run-index must be in [0, {len(seeds) - 1}]")

    heatmap_episode_indices = _resolve_episode_indices(
        _parse_csv_ints(args.heatmap_episodes),
        config.num_episodes,
    )
    gif_episode_indices = _resolve_episode_indices(
        _parse_csv_ints(args.gif_episode),
        config.num_episodes,
    )
    gif_episode = next(iter(sorted(gif_episode_indices)), config.num_episodes - 1)
    heatmap_modalities = _parse_csv_ints(args.heatmap_modalities)
    if not heatmap_modalities:
        heatmap_modalities = list(range(len(template.num_obs)))

    capture_by_seed: Dict[int, set[int]] = {seed: set() for seed in seeds}
    if args.save_heatmaps:
        capture_by_seed[seeds[args.heatmap_run_index]].update(heatmap_episode_indices)
    if args.save_gif:
        capture_by_seed[seeds[args.gif_run_index]].add(gif_episode)

    run_results: List[RunResult] = []
    t0 = time.perf_counter()
    for seed in seeds:
        result = _run_single_seed(
            seed=seed,
            config=config,
            env=env,
            env_params=env_params,
            jit_rollout=run_rollout,
            template=template,
            continuous_bin_centers=continuous_bin_centers,
            capture_episodes=capture_by_seed.get(seed, set()),
            verbose=not args.quiet,
        )
        run_results.append(result)
    total_runtime = time.perf_counter() - t0

    _plot_metric_over_episodes(
        run_results,
        metric_name="episode_lengths",
        ylabel="Tmax per episode",
        title="CartPole Control Performance (Tmax)",
        out_path=plots_dir / "tmax_per_episode.png",
    )
    _plot_tmax_moving_average(
        run_results,
        window=args.moving_average_window,
        out_path=plots_dir / "tmax_moving_average.png",
    )
    _plot_metric_over_episodes(
        run_results,
        metric_name="episode_pred_ce",
        ylabel="Cross-entropy",
        title="Inference Quality (Prediction Cross-Entropy)",
        out_path=plots_dir / "prediction_cross_entropy_per_episode.png",
    )
    _plot_metric_over_episodes(
        run_results,
        metric_name="episode_pred_acc",
        ylabel="Top-1 accuracy",
        title="Inference Quality (Prediction Accuracy)",
        out_path=plots_dir / "prediction_accuracy_per_episode.png",
    )
    _plot_metric_over_episodes(
        run_results,
        metric_name="episode_pred_cont_l2",
        ylabel="L2 error",
        title="Inference Quality (Continuous Prediction L2 Error)",
        out_path=plots_dir / "prediction_continuous_l2_error_per_episode.png",
    )
    _plot_metric_over_episodes(
        run_results,
        metric_name="episode_pred_cont_diracc",
        ylabel="Sign-match accuracy",
        title="Inference Quality (Continuous Prediction Direction Accuracy)",
        out_path=plots_dir / "prediction_continuous_direction_accuracy_per_episode.png",
    )
    _plot_metric_over_episodes(
        run_results,
        metric_name="episode_steps_per_sec",
        ylabel="Steps / second",
        title="Rollout Throughput",
        out_path=plots_dir / "steps_per_second_per_episode.png",
    )
    _plot_modality_metric(
        run_results,
        metric_name="episode_pred_ce_by_modality",
        ylabel="Cross-entropy",
        title="Prediction Cross-Entropy By Modality",
        out_path=plots_dir / "prediction_cross_entropy_by_modality.png",
    )
    _plot_modality_metric(
        run_results,
        metric_name="episode_pred_acc_by_modality",
        ylabel="Top-1 accuracy",
        title="Prediction Accuracy By Modality",
        out_path=plots_dir / "prediction_accuracy_by_modality.png",
    )
    _plot_modality_metric(
        run_results,
        metric_name="episode_pred_cont_abs_error_by_modality",
        ylabel="Absolute error",
        title="Continuous Prediction Absolute Error By Modality",
        out_path=plots_dir / "prediction_continuous_abs_error_by_modality.png",
    )
    _plot_modality_metric(
        run_results,
        metric_name="episode_pred_cont_signed_error_by_modality",
        ylabel="Signed error",
        title="Continuous Prediction Signed Error By Modality",
        out_path=plots_dir / "prediction_continuous_signed_error_by_modality.png",
    )
    _plot_modality_metric(
        run_results,
        metric_name="episode_pred_cont_diracc_by_modality",
        ylabel="Sign-match accuracy",
        title="Continuous Prediction Direction Accuracy By Modality",
        out_path=plots_dir / "prediction_continuous_direction_accuracy_by_modality.png",
    )

    _save_run_metrics_csv(run_results, out_dir / "run_metrics.csv")
    _save_aggregate_metrics_csv(run_results, out_dir / "aggregate_metrics.csv")

    if args.save_heatmaps:
        seed_for_heatmaps = seeds[args.heatmap_run_index]
        run_for_heatmaps = next(r for r in run_results if r.seed == seed_for_heatmaps)
        _save_trajectory_artifacts(
            run_result=run_for_heatmaps,
            episodes=sorted(heatmap_episode_indices),
            modalities=heatmap_modalities,
            out_dir=heatmaps_dir,
        )

    if args.save_gif:
        seed_for_gif = seeds[args.gif_run_index]
        run_for_gif = next(r for r in run_results if r.seed == seed_for_gif)
        gif_traj = run_for_gif.trajectories.get(gif_episode)
        if gif_traj is not None:
            gif_path = gifs_dir / f"cartpole_seed_{seed_for_gif}_episode_{gif_episode:04d}.gif"
            _save_episode_gif(
                trajectory=gif_traj,
                out_path=gif_path,
                fps=args.gif_fps,
                max_frames=args.gif_max_frames,
            )

    summary = {
        "config": asdict(config),
        "seeds": seeds,
        "output_dir": str(out_dir),
        "total_runtime_sec": float(total_runtime),
        "summary": _summary_stats(run_results),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CartPole Evaluation Harness Complete ===")
    print(f"output_dir={out_dir}")
    print(f"num_runs={len(run_results)}")
    print(f"num_episodes={config.num_episodes}")
    print(f"mean_tmax={summary['summary']['tmax_mean_all']:.3f}")
    print(f"mean_prediction_ce={summary['summary']['pred_ce_mean_all']:.6f}")
    print(f"mean_prediction_acc={summary['summary']['pred_acc_mean_all']:.6f}")
    print(f"mean_prediction_cont_l2={summary['summary']['pred_cont_l2_mean_all']:.6f}")
    print(f"mean_prediction_cont_diracc={summary['summary']['pred_cont_diracc_mean_all']:.6f}")
    print(f"mean_steps_per_sec={summary['summary']['steps_per_sec_mean_all']:.3f}")


if __name__ == "__main__":
    main()
