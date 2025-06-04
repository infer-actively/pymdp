from functools import partial
from jax import vmap, nn, random as jr, tree_util as jtu, lax
from pymdp.control import (
    compute_expected_state,
    compute_expected_obs,
    compute_info_gain,
    compute_expected_utility,
)

import mctx
import jax.numpy as jnp


def mcts_policy_search(
    search_algo=mctx.gumbel_muzero_policy, max_depth=6, num_simulations=4096
):

    def si_policy(agent, beliefs, rng_key):

        # remove time dimension
        embedding = jtu.tree_map(lambda x: x[:, 0], beliefs)
        root = mctx.RootFnOutput(
            prior_logits=jnp.log(agent.E),
            value=jnp.zeros((agent.batch_size)),
            embedding=embedding,
        )

        recurrent_fn = make_aif_recurrent_fn()

        policy_output = search_algo(
            agent,
            rng_key,
            root,
            recurrent_fn,
            num_simulations=num_simulations,
            max_depth=max_depth,
        )

        return policy_output.action_weights, policy_output

    return si_policy


@partial(vmap, in_axes=(0, 0, 0, None, 0, None, 0, None, None))
def compute_neg_efe(qs, action, A, A_dependencies, B, B_dependencies, C, use_states_info_gain=True, use_utility=True):
    qs_next_pi = compute_expected_state(
        qs, B, action, B_dependencies=B_dependencies
    )
    qo_next_pi = compute_expected_obs(qs_next_pi, A, A_dependencies)
    if use_states_info_gain:
        exp_info_gain = compute_info_gain(
            qs_next_pi, qo_next_pi, A, A_dependencies
        )
    else:
        exp_info_gain = 0.0

    if use_utility:
        exp_utility = compute_expected_utility(qo_next_pi, C)
    else:
        exp_utility = 0.0

    return exp_utility + exp_info_gain, qs_next_pi, qo_next_pi


@partial(vmap, in_axes=(0, 0, None))
def get_prob_single_modality(o_m, po_m, distr_obs):
    """Compute observation likelihood for a single modality (observation and likelihood)"""
    return jnp.inner(o_m, po_m) if distr_obs else po_m[o_m]


def make_aif_recurrent_fn():
    """Returns a recurrent_fn for an AIF agent."""

    def recurrent_fn(agent, rng_key, action, embedding):
        multi_action = agent.policies[action, 0]
        qs = embedding
        neg_efe, qs_next_pi, qo_next_pi = compute_neg_efe(qs,
                                                          multi_action,
                                                          agent.A,
                                                          agent.A_dependencies,
                                                          agent.B,
                                                          agent.B_dependencies,
                                                          agent.C,
                                                          agent.use_states_info_gain,
                                                          agent.use_utility)

        # recursively branch the policy + outcome tree
        choice = lambda key, po: jr.categorical(key, logits=jnp.log(po))
        if agent.onehot_obs:
            sample = lambda key, po, no: nn.one_hot(choice(key, po), no)
        else:
            sample = lambda key, po, no: choice(key, po)

        # set discount to outcome probabilities
        discount = 1.0
        obs = []
        for no_m, qo_m in zip(agent.num_obs, qo_next_pi):
            rng_key, key = jr.split(rng_key)
            o_m = sample(key, qo_m, no_m)
            discount *= get_prob_single_modality(o_m, qo_m, agent.onehot_obs)
            obs.append(jnp.expand_dims(o_m, 1))

        qs_next_posterior = agent.infer_states(obs, qs_next_pi)
        # remove time dimension
        # TODO: update infer_states to not expand along time dimension when needed
        qs_next_posterior = jtu.tree_map(lambda x: x.squeeze(1), qs_next_posterior)
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=neg_efe,
            discount=discount,
            prior_logits=jnp.log(agent.E),
            value=jnp.zeros_like(neg_efe),
        )

        return recurrent_fn_output, qs_next_posterior

    return recurrent_fn


# custom rollout function for mcts
def rollout(policy_search, agent, env, num_timesteps, rng_key):
    # get the batch_size of the agent
    batch_size = agent.batch_size

    def step_fn(carry, x):
        observation_t = carry["observation_t"]
        prior = carry["empirical_prior"]
        env = carry["env"]
        rng_key = carry["rng_key"]

        # We infer the posterior using FPI
        # so we don't need past actions or qs_hist
        qs = agent.infer_states(observation_t, prior)
        rng_key, key = jr.split(rng_key)
        qpi, _ = policy_search(key, agent, qs)

        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        action_t = agent.sample_action(qpi, rng_key=keys[1:])

        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        observation_t, env = env.step(rng_key=keys[1:], actions=action_t)

        prior, _ = agent.infer_empirical_prior(action_t, qs)

        carry = {
            "observation_t": observation_t,
            "empirical_prior": prior,
            "env": env,
            "rng_key": rng_key,
        }
        info = {
            "qpi": qpi,
            "qs": jtu.tree_map(lambda x: x[:, 0], qs),
            "env": env,
            "observation": observation_t,
            "action": action_t,
        }

        return carry, info

    # generate initial observation
    keys = jr.split(rng_key, batch_size + 1)
    rng_key = keys[0]
    observation_0, env = env.step(keys[1:])

    initial_carry = {
        "observation_t": observation_0,
        "empirical_prior": agent.D,
        "env": env,
        "rng_key": rng_key,
    }

    # Scan over time dimension (axis 1)
    last, info = lax.scan(step_fn, initial_carry, jnp.arange(num_timesteps))

    info = jtu.tree_map(lambda x: jnp.swapaxes(x, 0, 1), info)
    return last, info, env
