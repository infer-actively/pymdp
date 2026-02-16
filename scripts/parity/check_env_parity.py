#!/usr/bin/env python3
"""Compare Gymnasium and Gymnax CartPole one-step transitions.

This script checks parity at two levels:
1) Raw continuous next observation from identical (state, action)
2) Discretized observations used by the CartPole notebooks
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium as gym
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from gymnax.environments.classic_control.cartpole import CartPole as GymnaxCartPole
from gymnax.environments.classic_control.cartpole import EnvState as GymnaxState


@dataclass(frozen=True)
class Discretizer:
    num_obs: tuple[int, int, int, int]
    cart_velocity_max: float = 2.5 * 2
    pole_angle_max: float = 0.418 / 2 * 2
    pole_ang_velocity_max: float = 3.0 * 2

    def _disc(self, value: float, max_abs: float, n_bins: int) -> int:
        bin_size = (2 * max_abs) / (n_bins - 1)
        idx = int(np.rint((value + max_abs) / bin_size))
        return int(np.clip(idx, 0, n_bins - 1))

    def from_obs(self, obs: np.ndarray, action: int) -> tuple[int, int, int, int]:
        cart_velocity = self._disc(obs[1], self.cart_velocity_max, self.num_obs[0])
        pole_angle = self._disc(obs[2], self.pole_angle_max, self.num_obs[1])
        pole_ang_velocity = self._disc(obs[3], self.pole_ang_velocity_max, self.num_obs[2])
        return (cart_velocity, pole_angle, pole_ang_velocity, int(action))


def gymnasium_step_from_state(
    env: gym.Env, state: np.ndarray, action: int
) -> tuple[np.ndarray, bool]:
    unwrapped = env.unwrapped
    # Match underlying environment state directly; avoids reset-time randomness.
    unwrapped.state = np.asarray(state, dtype=np.float32)
    unwrapped.steps_beyond_terminated = None
    obs, _, terminated, truncated, _ = unwrapped.step(int(action))
    return np.asarray(obs, dtype=np.float32), bool(terminated or truncated)


def gymnax_step_from_state(
    env: GymnaxCartPole, state: np.ndarray, action: int, time: int = 0
) -> tuple[np.ndarray, bool]:
    params = env.default_params
    env_state = GymnaxState(
        x=jnp.asarray(state[0], dtype=jnp.float32),
        x_dot=jnp.asarray(state[1], dtype=jnp.float32),
        theta=jnp.asarray(state[2], dtype=jnp.float32),
        theta_dot=jnp.asarray(state[3], dtype=jnp.float32),
        time=int(time),
    )
    obs, _, _, done, _ = env.step_env(jr.PRNGKey(0), env_state, int(action), params)
    return np.asarray(obs, dtype=np.float32), bool(done)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-obs", type=int, default=64)
    parser.add_argument("--num-cases", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    num_obs = (args.num_obs, args.num_obs, args.num_obs, 2)
    discretizer = Discretizer(num_obs=num_obs)

    gym_env = gym.make("CartPole-v1")
    gymnax_env = GymnaxCartPole()

    # Sample states in a stable range and add a few edge-ish cases.
    sampled = rng.uniform(
        low=np.array([-0.5, -1.5, -0.12, -1.5], dtype=np.float32),
        high=np.array([0.5, 1.5, 0.12, 1.5], dtype=np.float32),
        size=(args.num_cases, 4),
    ).astype(np.float32)
    edge_cases = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.05, -0.8],
            [-0.2, -0.8, -0.05, 0.8],
            [2.2, 0.0, 0.18, 0.0],
        ],
        dtype=np.float32,
    )
    states = np.concatenate([sampled, edge_cases], axis=0)

    worst_raw_diff = 0.0
    raw_failures = 0
    disc_failures = 0

    for idx, state in enumerate(states):
        for action in (0, 1):
            obs_gym, done_gym = gymnasium_step_from_state(gym_env, state, action)
            obs_gymnax, done_gymnax = gymnax_step_from_state(gymnax_env, state, action)

            raw_diff = np.max(np.abs(obs_gym - obs_gymnax))
            worst_raw_diff = max(worst_raw_diff, float(raw_diff))

            if raw_diff > 1e-5 or done_gym != done_gymnax:
                raw_failures += 1
                print(
                    f"[RAW MISMATCH] case={idx} action={action} "
                    f"max_abs_diff={raw_diff:.6g} done(gym,gymnax)=({done_gym},{done_gymnax})"
                )

            disc_gym = discretizer.from_obs(obs_gym, action)
            disc_gymnax = discretizer.from_obs(obs_gymnax, action)
            if disc_gym != disc_gymnax:
                disc_failures += 1
                print(
                    f"[DISC MISMATCH] case={idx} action={action} "
                    f"gym={disc_gym} gymnax={disc_gymnax}"
                )

    print("=== Environment Parity Summary ===")
    print(f"total_cases={len(states) * 2}")
    print(f"raw_mismatches={raw_failures}")
    print(f"discretized_mismatches={disc_failures}")
    print(f"worst_raw_max_abs_diff={worst_raw_diff:.6g}")


if __name__ == "__main__":
    main()

