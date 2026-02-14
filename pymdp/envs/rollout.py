"""Utilities for running active-inference loops against environment dynamics.

The two primary public entry points are:
- :func:`infer_and_plan` for one-step inference/planning/action selection
- :func:`rollout` for multi-step scanned execution with optional online learning
"""

import warnings
from functools import partial
from typing import List

from jax import lax, vmap
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from pymdp import control
from pymdp.agent import Agent
from pymdp.envs.env import Env
from pymdp.inference import SEQUENCE_METHODS, SMOOTHING_METHODS


MAX_WINDOWED_HISTORY_WITHOUT_HORIZON = 32


def _append_to_window(window, value):
    if window.shape[1] == 0:
        return window

    if value.ndim == window.ndim - 1:
        value = jnp.expand_dims(value, axis=1)

    value = value[:, -1:, ...]
    return jnp.concatenate([window[:, 1:, ...], value], axis=1)


def _resolve_history_len(agent, num_timesteps, use_windowing):
    if not use_windowing:
        return 1

    if agent.inference_horizon is not None:
        return agent.inference_horizon

    uncapped_history_len = num_timesteps + 1
    if uncapped_history_len > MAX_WINDOWED_HISTORY_WITHOUT_HORIZON:
        warnings.warn(
            "No `inference_horizon` provided for a windowed inference mode; "
            f"capping rollout history to {MAX_WINDOWED_HISTORY_WITHOUT_HORIZON} steps "
            "to keep fixed-size rollout buffers bounded. "
            "Set `inference_horizon` explicitly to override this cap.",
            UserWarning,
            stacklevel=2,
        )
    return min(uncapped_history_len, MAX_WINDOWED_HISTORY_WITHOUT_HORIZON)


def default_policy_search(agent, qs, rng_key):
    qpi, G = agent.infer_policies(
        qs
    )  # infer_policies computes posterior over policies using EFE
    return qpi, {"G": G}


def _resolve_empirical_prior(agent, qs_prev, action_prev, empirical_prior):
    if empirical_prior is not None:
        return empirical_prior

    if action_prev is None:
        return agent.D

    return lax.cond(
        jnp.any(action_prev < 0),
        lambda _: agent.D,  # no valid previous action available
        lambda _: agent.update_empirical_prior(action_prev, qs_prev),
        operand=None,
    )


def update_parameters_online(
    agent,
    qs_prev,
    qs,
    observation,
    action_prev,
    *,
    learning_observations=None,
    learning_actions=None,
    learning_beliefs=None,
):
    # `qs_prev` must remain available here: we need it to construct temporal
    # belief pairs for transition learning and smoothing windows.
    if agent.inference_algo in SEQUENCE_METHODS:
        observations_for_learning = (
            observation if learning_observations is None else learning_observations
        )
        if agent.learn_B:
            if learning_actions is None:
                actions_for_learning = jnp.expand_dims(action_prev, 1)
                beliefs_B = jtu.tree_map(
                    lambda x, y: jnp.concatenate([x, y], axis=1), qs_prev, qs
                )
            else:
                actions_for_learning = learning_actions
                beliefs_B = qs
        else:
            actions_for_learning = action_prev
            beliefs_B = None

        return agent.infer_parameters(
            qs,
            observations_for_learning,
            actions_for_learning,
            beliefs_B=beliefs_B,
        )

    if (agent.inference_algo in SMOOTHING_METHODS) and (learning_beliefs is not None):
        observations_for_learning = (
            observation if learning_observations is None else learning_observations
        )
        beliefs_for_learning = jtu.tree_map(
            _append_to_window,
            learning_beliefs,
            qs,
        )

        if agent.learn_B:
            actions_for_learning = (
                learning_actions
                if learning_actions is not None
                else jnp.expand_dims(action_prev, 1)
            )
            beliefs_B = beliefs_for_learning
        else:
            actions_for_learning = (
                learning_actions
                if learning_actions is not None
                else action_prev
            )
            beliefs_B = None

        return agent.infer_parameters(
            beliefs_for_learning,
            observations_for_learning,
            actions_for_learning,
            beliefs_B=beliefs_B,
        )

    if agent.learn_B:
        # Stack beliefs so B-learning can build t->t+1 transitions.
        beliefs_B = jtu.tree_map(lambda x, y: jnp.concatenate([x, y], axis=1), qs_prev, qs)
        action_B = jnp.expand_dims(action_prev, 1)
    else:
        beliefs_B = None
        action_B = action_prev

    return agent.infer_parameters(
        qs,
        observation,
        action_B if agent.learn_B else action_prev,
        beliefs_B=beliefs_B,
    )


def _compute_sequence_empirical_prior_next(
    updated_agent,
    qs_window,
    action_hist,
    empirical_prior,
    valid_steps,
    history_len,
):
    def _shift_empirical_prior(_):
        qs_window_start = jtu.tree_map(lambda x: x[:, 0, ...], qs_window)
        action_window_start = action_hist[:, 0, :]
        propagate = partial(
            control.compute_expected_state,
            B_dependencies=updated_agent.B_dependencies,
        )
        return vmap(propagate)(qs_window_start, updated_agent.B, action_window_start)

    return lax.cond(
        (valid_steps == history_len) & (history_len > 1),
        _shift_empirical_prior,
        lambda _: empirical_prior,
        operand=None,
    )


def _run_sequence_fixed_window_step(
    agent,
    qs_prev,
    observation_hist,
    action_hist,
    action_prev,
    empirical_prior,
    valid_steps,
    history_len,
    rng_key,
    policy_search,
):
    updated_agent, action_next, qs, xtra = infer_and_plan(
        agent,
        qs_prev,
        observation=observation_hist,
        action_prev=action_prev,
        rng_key=rng_key,
        policy_search=policy_search,
        past_actions=action_hist,
        empirical_prior=empirical_prior,
        learning_observations=observation_hist,
        learning_actions=action_hist,
        valid_steps=valid_steps,
    )
    qs_carry = qs
    qs_latest = jtu.tree_map(lambda x: x[:, -1, ...], qs)
    empirical_prior_next = _compute_sequence_empirical_prior_next(
        updated_agent,
        qs,
        action_hist,
        empirical_prior,
        valid_steps,
        history_len,
    )
    return updated_agent, action_next, qs, xtra, qs_carry, qs_latest, empirical_prior_next


def _run_smoothing_fixed_window_step(
    agent,
    qs_prev,
    observation,
    observation_hist,
    action_hist,
    action_prev,
    rng_key,
    policy_search,
):
    updated_agent, action_next, qs, xtra = infer_and_plan(
        agent,
        qs_prev,
        observation,
        action_prev,
        rng_key,
        policy_search=policy_search,
        learning_observations=observation_hist,
        learning_actions=action_hist,
        learning_beliefs=qs_prev,
    )
    qs_carry = jtu.tree_map(_append_to_window, qs_prev, qs)
    qs_latest = jtu.tree_map(lambda x: x[:, -1, ...], qs)
    return updated_agent, action_next, qs, xtra, qs_carry, qs_latest


def _run_non_window_step(agent, qs_prev, observation, action_prev, rng_key, policy_search):
    updated_agent, action_next, qs, xtra = infer_and_plan(
        agent,
        qs_prev,
        observation,
        action_prev,
        rng_key,
        policy_search=policy_search,
    )
    qs_latest = jtu.tree_map(lambda x: x[:, -1, ...], qs)
    qs_carry = jtu.tree_map(lambda x: x[:, -1:, ...], qs)
    return updated_agent, action_next, qs, xtra, qs_carry, qs_latest


def _update_window_buffers(
    observation_hist,
    action_hist,
    valid_steps,
    observation_next,
    action_next,
    history_len,
):
    observation_hist_next = jtu.tree_map(_append_to_window, observation_hist, observation_next)
    action_hist_next = _append_to_window(action_hist, action_next)
    valid_steps_next = jnp.minimum(valid_steps + 1, history_len)
    return observation_hist_next, action_hist_next, valid_steps_next


def _init_observation_history(obs, history_len, categorical_obs):
    if obs.ndim == 1:
        obs = jnp.expand_dims(obs, axis=1)
    history_shape = (obs.shape[0], history_len) + obs.shape[2:]
    if categorical_obs and obs.shape[-1] > 0:
        # Padded categorical timesteps should contribute no evidence.
        hist = jnp.zeros(history_shape, dtype=obs.dtype)
    else:
        hist = jnp.full(history_shape, -1, dtype=obs.dtype)
    start_idx = (0, history_len - 1) + (0,) * (hist.ndim - 2)
    return lax.dynamic_update_slice(hist, obs[:, :1, ...], start_idx)


def _init_windowed_carry(agent, observation_0, env_state, rng_key, history_len, action_history_len, is_sequence_method):
    action_0 = -jnp.ones((agent.batch_size, agent.policies.policy_arr.shape[-1]), dtype=jnp.int32)

    # Initialize windowed beliefs with D across all slots. Left padded slots are
    # intentionally prior-filled (not neutral-padded): sequence inference/learning
    # uses `valid_steps` and action-validity masks to ignore them until they become
    # part of the valid suffix, and the rolling window then overwrites them over time.
    qs_0 = jtu.tree_map(
        lambda x: jnp.broadcast_to(
            jnp.expand_dims(x, axis=1),
            (x.shape[0], history_len, x.shape[-1]),
        ),
        agent.D,
    )

    observation_hist_0 = jtu.tree_map(
        lambda obs: _init_observation_history(obs, history_len, agent.categorical_obs),
        observation_0,
    )
    action_hist_0 = -jnp.ones(
        (agent.batch_size, action_history_len, agent.policies.policy_arr.shape[-1]),
        dtype=jnp.int32,
    )

    carry = {
        "qs": qs_0,
        "action": action_0,
        "observation": observation_0,
        "observation_hist": observation_hist_0,
        "action_hist": action_hist_0,
        "valid_steps": jnp.array(1, dtype=jnp.int32),
        "env_state": env_state,
        "agent": agent,
        "rng_key": rng_key,
    }
    if is_sequence_method:
        carry["empirical_prior"] = agent.D
    return carry


def _init_non_windowed_carry(agent, observation_0, env_state, rng_key):
    action_0 = -jnp.ones((agent.batch_size, agent.policies.policy_arr.shape[-1]), dtype=jnp.int32)
    qs_0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)
    return {
        "qs": qs_0,
        "action": action_0,
        "observation": observation_0,
        "env_state": env_state,
        "agent": agent,
        "rng_key": rng_key,
    }


def infer_and_plan(
    agent: Agent,
    qs_prev: List,
    observation: List,
    action_prev: jnp.array = None,
    rng_key: jr.PRNGKey = None,
    policy_search=None,
    past_actions=None,
    empirical_prior=None,
    learning_observations=None,
    learning_actions=None,
    learning_beliefs=None,
    valid_steps=None,
):
    """Run one active-inference step (state update, policy inference, action sample).

    Parameters
    ----------
    agent : Agent
        Active inference agent instance.
    qs_prev : list[jax.Array]
        Previous posterior beliefs over hidden states.
    observation : list[jax.Array] | list[int]
        Current environment observation.
    action_prev : jax.Array | None, optional
        Previous action. If ``None``, ``agent.D`` is used as empirical prior.
    rng_key : jax.Array
        PRNG key used by policy search and action sampling.
    policy_search : callable | None, optional
        Optional custom policy-search function. Defaults to expected-free-energy
        policy inference.
    past_actions : jax.Array | None, optional
        Optional action history for sequence inference methods.
    empirical_prior : list[jax.Array] | None, optional
        Optional override for the empirical prior.
    learning_observations : optional
        Optional learning observation buffer; defaults to current observation.
    learning_actions : optional
        Optional learning action buffer.
    learning_beliefs : optional
        Optional learning belief buffer for smoothing-based updates.
    valid_steps : int | jax.Array | None, optional
        Number of valid timesteps in padded fixed windows.

    Returns
    -------
    tuple
        ``(updated_agent, action, qs, info)`` where ``info`` contains policy
        posterior and additional policy-search diagnostics.
    """
    if policy_search is None:
        policy_search = default_policy_search

    empirical_prior = _resolve_empirical_prior(
        agent, qs_prev, action_prev, empirical_prior
    )

    # infer states
    qs = agent.infer_states(
        observations=observation,
        empirical_prior=empirical_prior,
        past_actions=past_actions,
        valid_steps=valid_steps,
    )

    # get posterior over policies
    rng_key, key = jr.split(rng_key)
    qpi, xtra = policy_search(
        agent, qs, key
    )  # compute policy posterior using EFE - uses C to consider preferred outcomes

    # for learning A and/or B
    if (
        (action_prev is not None)
        and (agent.learning_mode == "online")
        and (agent.learn_A or agent.learn_B)
    ):
        agent = update_parameters_online(
            agent,
            qs_prev,
            qs,
            observation,
            action_prev,
            learning_observations=learning_observations,
            learning_actions=learning_actions,
            learning_beliefs=learning_beliefs,
        )

    # sample action from policy distribution
    keys = jr.split(rng_key, agent.batch_size + 1)
    rng_key = keys[0]
    action = agent.sample_action(qpi, rng_key=keys[1:])

    info = {"empirical_prior": empirical_prior, "qpi": qpi}
    info.update(xtra)  # add extra information from policy search
    return agent, action, qs, info


def rollout(
    agent: Agent,
    env: Env,
    num_timesteps: int,
    rng_key: jr.PRNGKey,
    initial_carry=None,
    policy_search=None,
    env_params=None,
):
    """Roll out an active-inference agent/environment loop for ``num_timesteps``.

    Parameters
    ----------
    agent : Agent
        Active inference agent.
    env : Env
        Environment implementing ``reset`` and ``step``.
    num_timesteps : int
        Number of timesteps to simulate.
    rng_key : jax.Array
        Root PRNG key; internally split per-step and per-batch.
    initial_carry : dict | None, optional
        Optional carry overrides for warm-starting from existing state.
    policy_search : callable | None, optional
        Optional custom policy-search routine.
    env_params : pytree | None, optional
        Optional batched environment parameters.

    Returns
    ----------
    last : dict
        Final carry state after the final timestep.
    info : dict
        Time-indexed rollout traces (actions, observations, beliefs, etc.).
    """

    # get the batch_size of the agent
    batch_size = agent.batch_size
    is_sequence_method = agent.inference_algo in SEQUENCE_METHODS
    use_smoothing_online_windows = (
        (agent.inference_algo in SMOOTHING_METHODS)
        and (agent.learning_mode == "online")
        and (agent.learn_A or agent.learn_B)
    )
    use_windowing = is_sequence_method or use_smoothing_online_windows
    history_len = _resolve_history_len(agent, num_timesteps, use_windowing)
    action_history_len = max(history_len - 1, 0)

    # default policy search just uses standard active inference policy selection
    if policy_search is None:
        policy_search = default_policy_search

    def step_fn(carry, t):
        # Carry current action/observation/beliefs/state and RNG through scan.
        # We keep `qs_prev` explicitly because smoothing and transition learning
        # require temporal belief context beyond the empirical prior alone.
        action = carry["action"]
        observation = carry["observation"]
        qs_prev = carry["qs"]
        env_state = carry["env_state"]
        agent = carry["agent"]
        rng_key = carry["rng_key"]

        keys = jr.split(rng_key, batch_size + 2)
        rng_key = keys[0]  # carry first key

        use_fixed_window = is_sequence_method or use_smoothing_online_windows
        if use_fixed_window:
            observation_hist = carry["observation_hist"]
            action_hist = carry["action_hist"]
            valid_steps = carry["valid_steps"]

            if is_sequence_method:
                empirical_prior = carry["empirical_prior"]
                (
                    updated_agent,
                    action_next,
                    qs,
                    xtra,
                    qs_carry,
                    qs_latest,
                    empirical_prior_next,
                ) = _run_sequence_fixed_window_step(
                    agent,
                    qs_prev,
                    observation_hist,
                    action_hist,
                    action,
                    empirical_prior,
                    valid_steps,
                    history_len,
                    keys[1],
                    policy_search,
                )
            else:
                updated_agent, action_next, qs, xtra, qs_carry, qs_latest = _run_smoothing_fixed_window_step(
                    agent,
                    qs_prev,
                    observation,
                    observation_hist,
                    action_hist,
                    action,
                    keys[1],
                    policy_search,
                )
                empirical_prior_next = None
        else:
            # infer next action and beliefs using the agent's inference and planning methods
            updated_agent, action_next, qs, xtra, qs_carry, qs_latest = _run_non_window_step(
                agent,
                qs_prev,
                observation,
                action,
                keys[1],
                policy_search,
            )

        observation_next, env_state_next = vmap(env.step)(
            keys[2:], env_state, action_next, env_params=env_params
        )
        if updated_agent.learn_A:
            xtra["A"] = updated_agent.A
            xtra["pA"] = updated_agent.pA
        if updated_agent.learn_B:
            xtra["B"] = updated_agent.B
            xtra["pB"] = updated_agent.pB

        if use_fixed_window:
            observation_hist = carry["observation_hist"]
            action_hist = carry["action_hist"]
            valid_steps = carry["valid_steps"]
            (
                observation_hist_next,
                action_hist_next,
                valid_steps_next,
            ) = _update_window_buffers(
                observation_hist,
                action_hist,
                valid_steps,
                observation_next,
                action_next,
                history_len,
            )

            carry = {
                "action": action_next,
                "observation": observation_next,
                "observation_hist": observation_hist_next,
                "action_hist": action_hist_next,
                "valid_steps": valid_steps_next,
                "qs": qs_carry,
                "env_state": env_state_next,
                "agent": updated_agent,
                "rng_key": rng_key,
            }
            if is_sequence_method:
                carry["empirical_prior"] = empirical_prior_next
        else:
            carry = {
                "action": action_next,
                "observation": observation_next,
                "qs": qs_carry,
                "env_state": env_state_next,
                "agent": updated_agent,
                "rng_key": rng_key,
            }

        info = {
            "qs": qs_latest,
            "env_state": env_state,
            "observation": observation,
            "action": action_next,
        }
        info.update(xtra)  # add extra information from inference and planning

        return carry, info

    # initialise first observation from environment
    keys = jr.split(rng_key, batch_size + 1)
    rng_key = keys[0]
    observation_0, env_state = vmap(env.reset)(keys[1:], env_params=env_params)

    if is_sequence_method or use_smoothing_online_windows:
        built_carry = _init_windowed_carry(
            agent,
            observation_0,
            env_state,
            rng_key,
            history_len,
            action_history_len,
            is_sequence_method,
        )
    else:
        built_carry = _init_non_windowed_carry(
            agent,
            observation_0,
            env_state,
            rng_key,
        )

    if initial_carry is not None:
        built_carry = {**built_carry, **initial_carry}

    initial_carry = built_carry
    
    # run the active inference loop for num_timesteps using lax.scan
    last, info = lax.scan(step_fn, initial_carry, jnp.arange(num_timesteps + 1))

    info = jtu.tree_map(
        lambda x: x.transpose((1, 0) + tuple(range(2, x.ndim))), info
    )  # transpose to have timesteps as first dimension

    if agent.learning_mode == "offline":
        def _format_offline_observations(x):
            # Handle env wrappers that emit either (B, T, 1, O) or (B, T, O)
            # for categorical observations, and either (B, T, 1) or (B, T) for
            # discrete observations.
            if agent.categorical_obs:
                if x.ndim >= 4 and x.shape[2] == 1:
                    return x[:, :, 0, ...]
                return x
            if x.ndim >= 3 and x.shape[-1] == 1:
                return x[..., 0]
            return x

        observations_for_learning = jtu.tree_map(_format_offline_observations, info["observation"])
        last["agent"] = last["agent"].infer_parameters(
            info["qs"],
            observations_for_learning,
            info["action"],
        )

    return last, info
