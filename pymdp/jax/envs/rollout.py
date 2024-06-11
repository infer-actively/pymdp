import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.lax

from pymdp.jax.agent import Agent
from pymdp.jax.envs.env import PyMDPEnv


def rollout(agent: Agent, env: PyMDPEnv, num_timesteps: int, rng_key: jr.PRNGKey):
    """
    Rollout an agent in an environment for a number of timesteps.

        agent: pymdp agent to generate actions
        env: pymdp environment to interact with
        num_timesteps: number of timesteps to rollout for
        rng_key: random key to use for sampling actions
    """
    # get the batch_size of the agent
    batch_size = agent.A[0].shape[0]

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
        qpi, nefe = agent.infer_policies(qs)

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
    observation_0, env = env.step(keys[1:])

    # initial belief
    qs_0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)

    # initial action
    # TODO better fill with zeros?
    qpi_0, _ = agent.infer_policies(qs_0)
    keys = jr.split(rng_key, batch_size + 1)
    rng_key = keys[0]
    action_t = agent.sample_action(qpi_0, rng_key=keys[1:])

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
