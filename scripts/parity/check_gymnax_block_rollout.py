#!/usr/bin/env python3
"""Smoke/parity checks for block-wise offline rollout with Gymnax CartPole.

Checks performed:
1) No non-finite values in G on valid (pre-terminal) timesteps
2) One learning update per block, matching ceil(episode_len / inference_horizon)
3) Episode boundaries are preserved (non-auto-reset wrapper, done-aware trimming)
"""

from __future__ import annotations

import argparse
import math
from functools import partial
from typing import NamedTuple

from equinox import tree_at
import gymnax
from gymnax.environments.classic_control.cartpole import EnvState as GymnaxEnvState
from jax import jit, lax, nn
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np

from pymdp.agent import Agent
from pymdp.envs.env import Env
from pymdp.envs.rollout import rollout
from pymdp.legacy import utils


class GymnaxCartPoleState(NamedTuple):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: jnp.ndarray
    done: jnp.ndarray


class GymnaxCartPoleEnv(Env):
    def __init__(self, num_obs, action_division=2):
        self.num_obs = num_obs
        self.num_actions = action_division
        self.env, self._default_env_params = gymnax.make("CartPole-v1")

        self.cart_velocity_max = 2.5 * 2
        self.pole_angle_max = 0.418 / 2 * 2
        self.pole_ang_velocity_max = 3.0 * 2

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


def _to_jax_batched(obj_arr, batch_size):
    return jtu.tree_map(
        lambda x: jnp.broadcast_to(jnp.asarray(x), (batch_size,) + x.shape), list(obj_arr)
    )


def _build_agent(obs_division, inference_horizon, batch_size):
    action_division = 2
    num_obs = [obs_division, obs_division, obs_division, action_division]
    num_states = [(len(num_obs) + 1) * 2]
    num_controls = [action_division]

    A = utils.random_A_matrix(num_obs, num_states)
    A = utils.norm_dist_obj_arr(A * 0 + 1)
    B = utils.random_B_matrix(num_states, num_controls)
    B = utils.norm_dist_obj_arr(B * 0 + 1)
    C = utils.obj_array_zeros(num_obs)
    D = utils.obj_array_uniform(num_states)
    pA = utils.dirichlet_like(A, scale=1e-2)
    pB = utils.dirichlet_like(B, scale=1e-2)

    A_jax = _to_jax_batched(A, batch_size)
    B_jax = _to_jax_batched(B, batch_size)
    C_jax = _to_jax_batched(C, batch_size)
    D_jax = _to_jax_batched(D, batch_size)
    pA_jax = _to_jax_batched(pA, batch_size)
    pB_jax = _to_jax_batched(pB, batch_size)

    agent = Agent(
        A=A_jax,
        B=B_jax,
        C=C_jax,
        D=D_jax,
        pA=pA_jax,
        pB=pB_jax,
        learn_A=True,
        learn_B=True,
        learn_C=False,
        learn_D=False,
        learn_E=False,
        inference_algo="mmp",
        inference_horizon=inference_horizon,
        action_selection="stochastic",
        policy_len=2,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=True,
        use_inductive=False,
        categorical_obs=True,
        sampling_mode="full",
        num_iter=max(16, inference_horizon),
        batch_size=batch_size,
        learning_mode="offline",
    )
    return agent, num_obs


def _steps_until_done(info_block, last_block):
    done_pre = np.asarray(info_block["env_state"].done[0], dtype=bool)
    done_last = bool(np.asarray(last_block["env_state"].done[0]))
    done_next = np.concatenate([done_pre[1:], np.asarray([done_last], dtype=bool)])
    if done_next.any():
        return int(np.argmax(done_next)) + 1, True
    return int(done_next.shape[0]), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--obs-division", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    batch_size = 1
    agent, num_obs = _build_agent(args.obs_division, args.horizon, batch_size)
    env = GymnaxCartPoleEnv(num_obs)
    env_params = env.generate_env_params(batch_size=batch_size)

    run_rollout = jit(rollout, static_argnums=(1, 2))
    key = jr.PRNGKey(args.seed)

    lengths = []
    block_counts = []
    nonfinite_blocks = 0

    for episode in range(args.episodes):
        carry = None
        done = False
        steps = 0
        blocks = 0

        while (not done) and (steps < args.max_steps):
            block_len = min(args.horizon, args.max_steps - steps)
            key, key_block = jr.split(key)

            last, info = run_rollout(
                agent,
                env,
                block_len - 1,
                key_block,
                initial_carry=carry,
                env_params=env_params,
            )
            blocks += 1

            valid_steps, done = _steps_until_done(info, last)
            g_block = np.asarray(info["G"][0, :valid_steps])
            if not np.isfinite(g_block).all():
                nonfinite_blocks += 1

            steps += valid_steps

            # Mirror finite-horizon D update used in the notebook loop.
            agent = tree_at(lambda x: x.D, last["agent"], jtu.tree_map(lambda x: x[:, 0], last["qs"]))

            if done:
                break

            carry = dict(last)
            carry["agent"] = agent

        lengths.append(steps)
        block_counts.append(blocks)

    expected_blocks = [math.ceil(length / args.horizon) for length in lengths]
    block_parity_ok = block_counts == expected_blocks

    print("=== Gymnax Block Rollout Summary ===")
    print(f"episodes={args.episodes}")
    print(f"lengths={lengths}")
    print(f"blocks={block_counts}")
    print(f"expected_blocks={expected_blocks}")
    print(f"block_parity_ok={block_parity_ok}")
    print(f"nonfinite_blocks={nonfinite_blocks}")
    print(f"all_finite_G={nonfinite_blocks == 0}")

    if not block_parity_ok or nonfinite_blocks > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
