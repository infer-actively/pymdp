import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.lax

from pymdp.agent import Agent
from pymdp.envs.env import Env


def rollout(agent: Agent, env: Env, num_timesteps: int, rng_key: jr.PRNGKey, policy_search=None):
    """
    Rollout an agent in an environment for a number of timesteps.

    Parameters
    ----------
    agent: active inference agent 
    env: environment that can step forward and return observations
    num_timesteps: how many timesteps to simulate
    rng_key: random key for sampling
    policy_search: optional custom policy inference function such as sophisticated inference
    
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

        def default_policy_search(agent, qs, rng_key):
            qpi, _ = agent.infer_policies(qs) # infer_policies computes posterior over policies using EFE
            return qpi, None

        policy_search = default_policy_search

    def step_fn(carry, x):
        # carrying the current timestep's action, observation, beliefs, empirical prior, environment state, and random key
        action_t = carry["action_t"]
        observation_t = carry["observation_t"]
        qs = carry["qs"]
        empirical_prior = carry["empirical_prior"]
        env = carry["env"]
        agent = carry["agent"]
        rng_key = carry["rng_key"]

        # perform state inference using variational inference (FPI) - uses A matrix to map between hidden states and observations
        qs = agent.infer_states(
            observations=observation_t,
            empirical_prior=empirical_prior,
        )

        rng_key, key = jr.split(rng_key)
        qpi, _ = policy_search(agent, qs, key) # compute policy posterior using EFE - uses C to consider preferred outcomes

        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        action_t = agent.sample_action(qpi, rng_key=keys[1:]) # sample action from policy distribution

        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        observation_t, env = env.step(rng_key=keys[1:], actions=action_t) # step environment forward with chosen action

        empirical_prior, qs = agent.update_empirical_prior(action_t, qs) # updating the prior over hidden states and using B matrix to predict next state given action

        # carrying the next timestep's action, observation, beliefs, empirical prior, environment state, and random key
        carry = {
            "action_t": action_t,
            "observation_t": observation_t,
            "qs": jtu.tree_map(lambda x: x[:, -1:, ...], qs), # keep only latest belief
            "empirical_prior": empirical_prior,
            "env": env,
            "agent": agent,
            "rng_key": rng_key,
        }
        info = {
            "qpi": qpi,
            "qs": jtu.tree_map(lambda x: x[:, 0, ...], qs),
            "env": env,
            "agent": agent,
            "observation": observation_t,
            "action": action_t,
        }

        return carry, info

    # initialise first observation from environment
    keys = jr.split(rng_key, batch_size + 1)
    rng_key = keys[0]
    observation_0, env = env.reset(keys[1:])

    # specify initial beliefs using D 
    qs_0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)

    # get initial policy and action distribution - unused, just for shape matching
    qpi_0, _ = agent.infer_policies(qs_0)
    keys = jr.split(rng_key, batch_size + 1)
    rng_key = keys[0]
    action_t = agent.sample_action(qpi_0, rng_key=keys[1:])
    action_t *= 0 # zero out initial action as no action taken yet

    # set up initial state to carry through timesteps
    initial_carry = {
        "qs": qs_0,
        "action_t": action_t,
        "observation_t": observation_0,
        "empirical_prior": agent.D,
        "env": env,
        "agent": agent,
        "rng_key": rng_key,
    }

    # run the active inference loop for num_timesteps using jax.lax.scan
    last, info = jax.lax.scan(step_fn, initial_carry, jnp.arange(num_timesteps))

    # prepare initial info to concatenate with trajectory
    initial_info = {
        "action": jnp.expand_dims(action_t, 0),
        "observation": [jnp.expand_dims(o, 0) for o in observation_0],  
        "qs": jtu.tree_map(lambda x: jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim))), qs_0), 
        "qpi": jnp.expand_dims(qpi_0, 0),  
        "env": env,
        "agent": agent,
    }
    
    # helper function to concatenate initial state with trajectory by dealing with different shapes and data types
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

    # combine initial info with trajectory info
    info = jtu.tree_map(concat_or_pass, initial_info, info)

    return last, info, env
