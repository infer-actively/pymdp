import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.lax

from pymdp.jax.agent import Agent
from pymdp.jax.envs.env import PyMDPEnv


def rollout(agent: Agent, env: PyMDPEnv, num_timesteps: int, rng_key: jr.PRNGKey, policy_search = None):
    """
    Rollout an agent in an environment for a number of timesteps.

    Parameters
    ----------
    agent: ``Agent``
        Agent to interact with the environment
    env: ``PyMDPEnv`
        Environment to interact with
    num_timesteps: ``int``
        Number of timesteps to rollout for
    rng_key: ``PRNGKey``
        Random key to use for sampling actions
    policy_search: ``callable``
        Function to use for policy search (optional)
        Calls policy_search(agent, beliefs, rng_key) and expects q_pi, info back.
        If none, agent.infer_policies will be used.

    Returns
    ----------
    last: ``dict``
        Carry dictionary from the last timestep
    info: ``dict``
        Dictionary containing information about the rollout, i.e. executed actions, observations, beliefs, etc.
    env: ``PyMDPEnv``
        Environment state after the rollout
    """
    # get the batch_size of the agent
    batch_size = agent.batch_size

    if policy_search is None:
        def default_policy_search(agent, qs, rng_key):
            qpi, _ = agent.infer_policies(qs)
            return qpi, None
        policy_search = default_policy_search

    def step_fn(carry, x):
        action_t = carry["action_t"]
        observation_t = carry["observation_t"]
        qs = carry["qs"]
        empirical_prior = carry["empirical_prior"]
        env = carry["env"]
        rng_key = carry["rng_key"]

        # We infer the posterior using FPI
        # so we don't need past actions or qs_hist
        qs = agent.infer_states(
            observations=observation_t,
            past_actions=None,
            empirical_prior=empirical_prior,
            qs_hist=None,
        )
      
        rng_key, key = jr.split(rng_key)
        qpi, _ = policy_search(agent, qs, key)

        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        action_t = agent.sample_action(qpi, rng_key=keys[1:])

        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        observation_t, env = env.step(rng_key=keys[1:], actions=action_t)

        empirical_prior, qs = agent.infer_empirical_prior(action_t, qs)

        carry = {
            "action_t": action_t,
            "observation_t": observation_t,
            "qs": jtu.tree_map(lambda x: x[:, -1:, ...], qs),
            "empirical_prior": empirical_prior,
            "env": env,
            "rng_key": rng_key,
        }
        info = {
            "qpi": qpi,
            "qs": jtu.tree_map(lambda x: x[:, 0, ...], qs),
            "env": env,
            "observation": observation_t,
            "action": action_t,
        }

        return carry, info

    # generate initial observation
    keys = jr.split(rng_key, batch_size + 1)
    rng_key = keys[0]
    observation_0, env = env.step(keys[1:])

    # initial belief
    qs_0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)

    # infer initial action to get the right shape
    qpi_0, _ = agent.infer_policies(qs_0)
    keys = jr.split(rng_key, batch_size + 1)
    rng_key = keys[0]
    action_t = agent.sample_action(qpi_0, rng_key=keys[1:])
    # but set it to zeros
    action_t *= 0

    initial_carry = {
        "qs": qs_0,
        "action_t": action_t,
        "observation_t": observation_0,
        "empirical_prior": agent.D,
        "env": env,
        "rng_key": rng_key,
    }

    # Scan over time dimension (axis 1)
    last, info = jax.lax.scan(step_fn, initial_carry, jnp.arange(num_timesteps))

    info = jtu.tree_map(lambda x: jnp.swapaxes(x, 0, 1), info)
    return last, info, env
