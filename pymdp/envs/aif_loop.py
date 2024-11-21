import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.lax
from .env import Env
from pymdp.agent import Agent


def aif_loop(agent: Agent, env: Env, num_timesteps: int, rng_key: jr.PRNGKey):
    """
    Rollout an agent in an environment for a number of timesteps.

    Parameters
    ----------
    agent: ``Agent``
        Agent to interact with the environment
    env: ``Env`
        Environment to interact with
    num_timesteps: ``int``
        Number of timesteps to rollout for
    rng_key: ``PRNGKey``
        Random key to use for sampling actions

    Returns
    ----------
    last: ``dict``
        Carry dictionary from the last timestep
    info: ``dict``
        Dictionary containing information about the rollout, i.e. executed actions, observations, beliefs, etc.
    env: ``Env``
        Environment state after the rollout
    """
    # get the batch_size of the agent (number of parallel environments)
    batch_size = agent.batch_size

    def default_policy_search(agent, qs, rng_key):
        qpi, _ = agent.infer_policies(qs)
        return qpi, None

    policy_search = default_policy_search # default policy inference that returns action probabilities and no additional info

    def step_fn(carry, x):
        action_t = carry["action_t"]
        observation_t = carry["observation_t"]
        qs = carry["qs"]
        empirical_prior = carry["empirical_prior"]
        env = carry["env"]
        rng_key = carry["rng_key"]

        # we infer the posterior using FPI so we don't need past actions or qs_hist
        qs = agent.infer_states(
            observations=observation_t,
            empirical_prior=empirical_prior,
        )

        rng_key, key = jr.split(rng_key)
        qpi, _ = policy_search(agent, qs, key)

        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        action_t = agent.sample_action(qpi, rng_key=keys[1:])

        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        observation_t, env = env.step(rng_key=keys[1:], actions=action_t)

        empirical_prior, qs = agent.update_empirical_prior(action_t, qs)

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
    observation_0, env = env.reset(keys[1:])

    # initial belief
    qs_0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)

    # infer initial action to get the right shape
    qpi_0, _ = agent.infer_policies(qs_0)
    keys = jr.split(rng_key, batch_size + 1)
    rng_key = keys[0]
    action_t = agent.sample_action(qpi_0, rng_key=keys[1:])
    action_t *= 0

    initial_carry = {
        "qs": qs_0,
        "action_t": action_t,
        "observation_t": observation_0,
        "empirical_prior": agent.D,
        "env": env,
        "rng_key": rng_key,
    }

    # scan over time dimension
    last, info = jax.lax.scan(step_fn, initial_carry, jnp.arange(num_timesteps))

    initial_info = {
        "action": jnp.expand_dims(action_t, 0),
        "observation": [jnp.expand_dims(o, 0) for o in observation_0],  
        "qs": jtu.tree_map(lambda x: jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim))), qs_0), 
        "qpi": jnp.expand_dims(qpi_0, 0),  
        "env": env
    }
    
    def concat_or_pass(init, steps):
        if isinstance(init, list):
            return [jnp.concatenate([i, s], axis=0) for i, s in zip(init, steps)]
        elif isinstance(init, jnp.ndarray):
            if init.ndim < steps.ndim:
                init = jnp.expand_dims(init, 0)
            elif init.shape[1:] != steps.shape[1:]: 
                init = jnp.transpose(init, (1, 0) + tuple(range(2, init.ndim)))
            return jnp.concatenate([init, steps], axis=0)
        return steps

    info = jtu.tree_map(concat_or_pass, initial_info, info)

    return last, info, env