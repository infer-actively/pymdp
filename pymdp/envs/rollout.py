from typing import List

import jax
from jax import lax, vmap
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from pymdp.agent import Agent
from pymdp.envs.env import Env


def default_policy_search(agent, qs, rng_key):
    qpi, G = agent.infer_policies(
        qs
    )  # infer_policies computes posterior over policies using EFE
    return qpi, {"G": G}


def infer_and_plan(
    agent: Agent,
    qs_prev: List,
    observation: List,
    action_prev: jnp.array = None,
    rng_key: jr.PRNGKey = None,
    policy_search=None,
):
    """
    Perform a single inference and planning step for an agent in an environment.

    Parameters
    ----------
    agent: active inference agent
    qs_prev: previous posterior beliefs about hidden states
    observation: current observation from the environment
    action_prev: optional last action taken by the agent, if None agent.D is used as empirical prior
    rng_key: random key in cased the policy_search function needs to sample
    policy_search: optional custom policy inference function such as sophisticated inference
    """
    if policy_search is None:
        policy_search = default_policy_search

    empirical_prior = jax.lax.cond(
        jnp.any(action_prev < 0),
        lambda: agent.D,  # if no action is provided, use the agent's D as empirical prior
        lambda: agent.update_empirical_prior(action_prev, qs_prev)[0],
    )

    # infer states
    qs = agent.infer_states(
        observations=observation,
        empirical_prior=empirical_prior,
    )

    # get posterior over policies
    rng_key, key = jr.split(rng_key)
    qpi, xtra = policy_search(
        agent, qs, key
    )  # compute policy posterior using EFE - uses C to consider preferred outcomes

    # for learning A and/or B
    if action_prev is not None and (agent.learning_mode == "online"):
        if agent.learn_A or agent.learn_B:
            if agent.learn_B:
                # stacking beliefs for B learning
                beliefs_B = jtu.tree_map(
                    lambda x, y: jnp.concatenate([x, y], axis=1), qs_prev, qs
                )
                # reshaping action to match the stacked beliefs
                action_B = jnp.expand_dims(action_prev, 1)  # adding time dimension
            else:
                beliefs_B = None
                action_B = action_prev

            agent = agent.infer_parameters(
                qs,
                observation,
                action_B if agent.learn_B else action_prev,
                beliefs_B=beliefs_B,
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
    env_params_batched=False,
):
    """
    Rollout an agent in an environment for a number of timesteps.

    Parameters
    ----------
    agent: active inference agent
    env: environment that can step forward and return observations
    num_timesteps: how many timesteps to simulate
    rng_key: random key for sampling
    initial_carry: optional initial carry state to start the rollout from, if None it will be initialized from an environment reset
    policy_search: optional custom policy inference function such as sophisticated inference
    env_params: optional pytree of environment parameters
    env_params_batched: optional Boolean flag to indicate whether env_params carries a leading batch dim equal to agent.batch_size

    Returns
    ----------
    last: ``dict``
        dictionary from the last timestep about the rollout, i.e., the final action, observation, beliefs, etc.
    info: ``dict``
        dictionary containing information about the rollout, i.e. executed actions, observations, beliefs, etc.
    env: ``Env``
        environment state after the rollout
    """

    # get the batch_size of the agent
    batch_size = agent.batch_size

    # default policy search just uses standard active inference policy selection
    if policy_search is None:
        policy_search = default_policy_search

    if env_params_batched and env_params is None:
        raise ValueError("env_params_batched=True requires env_params to be provided")

    if env_params is None:
        env_reset = vmap(env.reset)
        env_step = vmap(env.step)
    else:
        reset_in_axes = (0, 0) if env_params_batched else (0, None)
        step_in_axes = (0, 0, 0, 0) if env_params_batched else (0, 0, 0, None)
        env_reset = vmap(lambda k, ep: env.reset(k, env_params=ep), in_axes=reset_in_axes)
        env_step = vmap(lambda k, s, a, ep: env.step(k, s, a, env_params=ep), in_axes=step_in_axes)

    def step_fn(carry, t):
        # carrying the current timestep's action, observation, beliefs, empirical prior, environment state, and random key
        action = carry["action"]
        observation = carry["observation"]
        qs_prev = carry["qs"]
        env_state = carry["env_state"]
        agent = carry["agent"]
        rng_key = carry["rng_key"]

        keys = jr.split(rng_key, batch_size + 2)
        rng_key = keys[0]  # carry first key

        # infer next action and beliefs using the agent's inference and planning methods
        # first timestep we don't have a prev action yet, so then we call with action=None
        updated_agent, action_next, qs, xtra = infer_and_plan(
            agent, qs_prev, observation, action, keys[1], policy_search=policy_search
        )

        # step the environment forward with the chosen action
        observation_next, env_state_next = env_step(keys[2:], env_state, action_next, env_params)

        # carrying the next timestep's action, observation, beliefs, empirical prior, environment state, and random key
        carry = {
            "action": action_next,
            "observation": observation_next,
            "qs": jtu.tree_map(lambda x: x[:, -1:, ...], qs),  # keep only latest belief
            "env_state": env_state_next,
            "agent": updated_agent,
            "rng_key": rng_key,
        }
        info = {
            "qs": jtu.tree_map(lambda x: x[:, 0, ...], qs),
            "env_state": env_state,
            "observation": observation,
            "action": action_next,
        }

        if updated_agent.learn_A:
            xtra["A"] = updated_agent.A
            xtra["pA"] = updated_agent.pA
        if updated_agent.learn_B:
            xtra["B"] = updated_agent.B
            xtra["pB"] = updated_agent.pB
            
        info.update(xtra)  # add extra information from inference and planning

        return carry, info

    if initial_carry is None:
        # initialise first observation from environment
        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        observation_0, env_state = env_reset(keys[1:], env_params)

        # specify initial beliefs using D
        qs_0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)

        # put action to -1 to indicate no action taken yet
        action_0 = -jnp.ones((agent.batch_size, agent.policies.shape[-1]), dtype=jnp.int32)

        # set up initial state to carry through timesteps
        initial_carry = {
            "qs": qs_0,
            "action": action_0,
            "observation": observation_0,
            "env_state": env_state,
            "agent": agent,
            "rng_key": rng_key,
        }

    # run the active inference loop for num_timesteps using lax.scan
    last, info = lax.scan(step_fn, initial_carry, jnp.arange(num_timesteps + 1))

    info = jtu.tree_map(
        lambda x: x.transpose((1, 0) + tuple(range(2, x.ndim))), info
    )  # transpose to have timesteps as first dimension

    if agent.learning_mode == "offline":
        outcomes = jtu.tree_map(lambda x: x.squeeze(-1), info['observation']) if num_timesteps > 1 else info['observation']
        last["agent"] = last["agent"].infer_parameters(info['qs'], outcomes, info['action'])

    return last, info
