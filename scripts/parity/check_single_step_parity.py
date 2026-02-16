#!/usr/bin/env python3
"""Single-step parity harness for old notebook loop vs rollout().

This script compares one MMP inference/action/learning step between:
1) Old notebook-style update loop
2) rollout()-based step

The primary parity target is the default rollout carry initialization.
An additional run with explicit uniform history initialization is kept as a
regression check for history-buffer handling.
"""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from equinox import tree_at
from jax import jit, nn

from pymdp.agent import Agent
from pymdp.envs.env import Env
from pymdp.envs.rollout import rollout
from pymdp.legacy import utils
from pymdp.legacy.maths import softmax


def _to_jax_batched(obj_arr, batch_size):
    return jtu.tree_map(lambda x: jnp.broadcast_to(jnp.asarray(x), (batch_size,) + x.shape), list(obj_arr))


def _max_abs_diff_tree(a, b):
    leaves_a = jtu.tree_leaves(a)
    leaves_b = jtu.tree_leaves(b)
    return max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(leaves_a, leaves_b))


class ConstantObsEnv(Env):
    """Jittable environment returning a fixed observation each step."""

    def __init__(self, obs_per_sample):
        # obs_per_sample: list of modality arrays shaped (1, O)
        self.obs_per_sample = [jnp.asarray(o) for o in obs_per_sample]

    def reset(self, key, state=None, env_params=None):
        if state is None:
            state = jnp.array(0, dtype=jnp.int32)
        return self.obs_per_sample, state

    def step(self, key, state, action, env_params=None):
        return self.obs_per_sample, state


def build_setup(obs_division=64, batch_size=1, inference_horizon=3, seed=0, use_notebook_c=True):
    action_division = 2
    num_obs = [obs_division, obs_division, obs_division, action_division]
    num_states = [(len(num_obs) + 1) * 2]
    num_controls = [action_division]

    # Keep A/B matched to notebook style (uniform then normalized).
    A = utils.random_A_matrix(num_obs, num_states)
    A = utils.norm_dist_obj_arr(A * 0 + 1)

    B = utils.random_B_matrix(num_states, num_controls)
    B = utils.norm_dist_obj_arr(B * 0 + 1)

    # Preferences C: default to the same construction as the notebook.
    C = utils.obj_array_zeros(num_obs)
    if use_notebook_c:
        pole_angle_max = 0.418 / 2 * 2
        goal_pole_angle = (0 + pole_angle_max) / (2 * pole_angle_max / (num_obs[1] - 1))
        x = np.linspace(0, obs_division - 1, obs_division)
        sigma = 1.7 / (pole_angle_max * 180 / 3.14159) * (num_obs[1] / 2)
        s = sigma * 0.5513 * 0.25
        z = (x - goal_pole_angle) / s
        C[1] = np.log(np.exp(-z) / (s * (1 + np.exp(-z)) ** 2))
    else:
        x = np.linspace(0, obs_division - 1, obs_division)
        center = (obs_division - 1) / 2.0
        sigma = max(1.0, obs_division / 12.0)
        pref = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        pref = softmax(pref)
        C[1] = np.log(pref + 1e-16)
    for m in range(len(C)):
        C[m] = np.log(softmax(C[m]) + 1e-16)

    D = utils.obj_array_uniform(num_states)
    pA = utils.dirichlet_like(A, scale=1e-2)
    pB = utils.dirichlet_like(B, scale=1e-2)

    A_jax = _to_jax_batched(A, batch_size)
    B_jax = _to_jax_batched(B, batch_size)
    C_jax = _to_jax_batched(C, batch_size)
    D_jax = _to_jax_batched(D, batch_size)
    pA_jax = _to_jax_batched(pA, batch_size)
    pB_jax = _to_jax_batched(pB, batch_size)

    # Fixed observation index tuple (same shape semantics as notebooks).
    obs_idx = [obs_division // 2, obs_division // 2, obs_division // 2, 0]
    obs_per_sample = [nn.one_hot(jnp.array(i, dtype=jnp.int32), num_obs[m])[None, :] for m, i in enumerate(obs_idx)]

    agent_kwargs = dict(
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
    )

    return {
        "num_obs": num_obs,
        "obs_per_sample": obs_per_sample,
        "obs_hist_single": jtu.tree_map(lambda o: jnp.expand_dims(o, axis=1), obs_per_sample),  # (B,1,O)
        "agent_kwargs": agent_kwargs,
        "inference_horizon": inference_horizon,
        "batch_size": batch_size,
    }


def old_loop_single_step(agent: Agent, obs_hist_single, key, inference_horizon, learn_now):
    """Replicates T=0 notebook-style step."""
    outcomes = obs_hist_single
    actions = None
    infer_prior = agent.D

    key, key_action = jr.split(key)
    beliefs = agent.infer_states(outcomes, infer_prior, past_actions=actions, qs_hist=None)
    qpi, G = agent.infer_policies(beliefs)
    action = agent.sample_action(qpi, rng_key=jr.split(key_action, agent.batch_size))

    actions = jnp.expand_dims(action, axis=1)  # (B,1,F)
    outcomes = jtu.tree_map(lambda x: x[:, -inference_horizon:], outcomes)
    beliefs = jtu.tree_map(lambda x: x[:, -inference_horizon:], beliefs)
    actions = actions[:, -inference_horizon:, :]

    # Notebook behavior: update D from start of the retained horizon.
    agent = tree_at(lambda x: x.D, agent, jtu.tree_map(lambda x: x[:, 0], beliefs))
    infer_args = agent.update_empirical_prior(action, beliefs)

    if learn_now:
        agent = agent.infer_parameters(infer_args[1], outcomes, actions, beliefs_B=infer_args[1], lr_pA=1.0, lr_pB=1.0)

    return {
        "agent": agent,
        "qpi": qpi,
        "G": G,
        "qs": beliefs,
        "action": action,
        "outcomes": outcomes,
        "actions_hist": actions,
        "infer_prior": infer_args[0],
    }


def build_rollout_initial_carry(agent, obs0, env_state, rng_key, inference_horizon, use_uniform_padding):
    """Override rollout's default carry to control history initialization."""
    batch_size = agent.batch_size
    num_controls = agent.policies.policy_arr.shape[-1]
    qs_0 = jtu.tree_map(
        lambda x: jnp.broadcast_to(jnp.expand_dims(x, axis=1), (x.shape[0], inference_horizon, x.shape[-1])),
        agent.D,
    )
    action_0 = -jnp.ones((batch_size, num_controls), dtype=jnp.int32)
    action_hist = -jnp.ones((batch_size, max(inference_horizon - 1, 0), num_controls), dtype=jnp.int32)

    def _obs_hist(o, n_o):
        # o shape: (B,1,O)
        if use_uniform_padding:
            hist = jnp.ones((o.shape[0], inference_horizon, o.shape[-1]), dtype=o.dtype) / float(n_o)
        else:
            hist = jnp.zeros((o.shape[0], inference_horizon, o.shape[-1]), dtype=o.dtype)
        hist = hist.at[:, -1:, :].set(o[:, :1, :])
        return hist

    observation_hist = [_obs_hist(o, agent.num_obs[m]) for m, o in enumerate(obs0)]
    return {
        "qs": qs_0,
        "action": action_0,
        "observation": obs0,
        "observation_hist": observation_hist,
        "action_hist": action_hist,
        "valid_steps": jnp.array(1, dtype=jnp.int32),
        "empirical_prior": agent.D,
        "env_state": env_state,
        "agent": agent,
        "rng_key": rng_key,
    }


def rollout_single_step(agent: Agent, env: Env, key, initial_carry=None):
    run_rollout = jit(rollout, static_argnums=(1, 2))
    last, info = run_rollout(agent, env, 0, key, initial_carry=initial_carry, env_params=None)
    return last, info


def summarize(label, old, rl_last, rl_info):
    rl_G = rl_info["G"][:, 0, :]
    rl_qpi = rl_info["qpi"][:, 0, :]
    rl_qs = jtu.tree_map(lambda x: x[:, 0, :], rl_info["qs"])
    rl_action = rl_info["action"][:, 0, :]

    any_nonfinite_G = bool(~np.isfinite(np.array(rl_G)).all())
    diff_G = _max_abs_diff_tree(old["G"], rl_G)
    diff_qpi = _max_abs_diff_tree(old["qpi"], rl_qpi)
    diff_qs = _max_abs_diff_tree(old["qs"], rl_qs)
    action_equal = bool(np.array_equal(np.array(old["action"]), np.array(rl_action)))
    diff_pA = _max_abs_diff_tree(old["agent"].pA, rl_last["agent"].pA)
    diff_pB = _max_abs_diff_tree(old["agent"].pB, rl_last["agent"].pB)

    print(f"\n=== {label} ===")
    print(f"any_nonfinite_G={any_nonfinite_G}")
    print(f"max_abs_diff_G={diff_G:.6g}")
    print(f"max_abs_diff_qpi={diff_qpi:.6g}")
    print(f"max_abs_diff_qs={diff_qs:.6g}")
    print(f"action_equal={action_equal}")
    print(f"max_abs_diff_pA={diff_pA:.6g}")
    print(f"max_abs_diff_pB={diff_pB:.6g}")

    return {
        "any_nonfinite_G": any_nonfinite_G,
        "diff_G": diff_G,
        "diff_qpi": diff_qpi,
        "diff_qs": diff_qs,
        "action_equal": action_equal,
        "diff_pA": diff_pA,
        "diff_pB": diff_pB,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-division", type=int, default=64)
    parser.add_argument("--inference-horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--simple-c", action="store_true", help="Use a simple Gaussian preference instead of notebook C.")
    args = parser.parse_args()

    setup = build_setup(
        obs_division=args.obs_division,
        batch_size=1,
        inference_horizon=args.inference_horizon,
        seed=args.seed,
        use_notebook_c=not args.simple_c,
    )
    env = ConstantObsEnv(setup["obs_per_sample"])

    # Old notebook behavior at T=0: learning runs because T % inference_horizon == 0.
    agent_old = Agent(**setup["agent_kwargs"], learning_mode="online")
    old = old_loop_single_step(
        agent_old,
        setup["obs_hist_single"],
        jr.PRNGKey(args.seed),
        setup["inference_horizon"],
        learn_now=True,
    )

    # rollout default behavior (online learning, default history initialization).
    agent_rl_default = Agent(**setup["agent_kwargs"], learning_mode="online")
    last_default, info_default = rollout_single_step(agent_rl_default, env, jr.PRNGKey(args.seed))
    default_metrics = summarize("rollout_default", old, last_default, info_default)

    # rollout with explicit uniform history padding to neutralize unobserved timepoints.
    agent_rl_uniform = Agent(**setup["agent_kwargs"], learning_mode="online")
    obs0, env_state0 = env.reset(jr.PRNGKey(123))
    obs0_batched = jtu.tree_map(lambda x: jnp.expand_dims(x, axis=0), obs0)  # (B,1,O)
    env_state_batched = jnp.expand_dims(env_state0, axis=0)
    carry_uniform = build_rollout_initial_carry(
        agent_rl_uniform,
        obs0_batched,
        env_state_batched,
        jr.PRNGKey(args.seed),
        setup["inference_horizon"],
        use_uniform_padding=True,
    )
    last_uniform, info_uniform = rollout_single_step(
        agent_rl_uniform, env, jr.PRNGKey(args.seed), initial_carry=carry_uniform
    )
    uniform_metrics = summarize("rollout_uniform_init", old, last_uniform, info_uniform)

    tol = 1e-8
    checks = [("default", default_metrics), ("uniform_init", uniform_metrics)]
    failures = []
    for name, metrics in checks:
        if metrics["any_nonfinite_G"]:
            failures.append(f"{name}: non-finite G")
        if metrics["diff_G"] > tol:
            failures.append(f"{name}: diff_G={metrics['diff_G']:.3g}")
        if metrics["diff_qpi"] > tol:
            failures.append(f"{name}: diff_qpi={metrics['diff_qpi']:.3g}")
        if metrics["diff_qs"] > tol:
            failures.append(f"{name}: diff_qs={metrics['diff_qs']:.3g}")
        if metrics["diff_pA"] > tol:
            failures.append(f"{name}: diff_pA={metrics['diff_pA']:.3g}")
        if metrics["diff_pB"] > tol:
            failures.append(f"{name}: diff_pB={metrics['diff_pB']:.3g}")

    print("\n=== Parity Check Summary ===")
    print(f"passed={len(failures) == 0}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
