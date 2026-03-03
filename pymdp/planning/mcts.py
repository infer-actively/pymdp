from __future__ import annotations

from functools import partial
from typing import Any, Callable
from jax import vmap, nn, random as jr, tree_util as jtu, lax
from pymdp.control import (
    compute_expected_state,
    compute_expected_obs,
    compute_info_gain,
    compute_expected_utility,
)

import jax.numpy as jnp

try:
    import mctx
except ImportError:  # pragma: no cover - handled by runtime guard
    mctx = None


def _require_mctx() -> None:
    """Raise a helpful error if optional `mctx` is not installed."""
    if mctx is None:
        raise ImportError(
            "`pymdp.planning.mcts` requires the optional dependency `mctx`. "
            "Install it to use MCTS policy search."
        )


def mcts_policy_search(
    search_algo: Callable | None = None,
    max_depth: int = 6,
    num_simulations: int = 4096,
) -> Callable[[Any, list[jnp.ndarray], jnp.ndarray], tuple[jnp.ndarray, Any]]:
    """Build an MCTS-based policy-search callable for `Agent` planning.

    Parameters
    ----------
    search_algo : Callable, optional
        MCTS search routine from `mctx` used to evaluate policy actions.
    max_depth : int, default=6
        Maximum planning depth for the tree search.
    num_simulations : int, default=4096
        Number of MCTS simulations per planning call.

    Returns
    -------
    Callable[[Any, list[jnp.ndarray], jnp.ndarray], tuple[jnp.ndarray, Any]]
        Function with signature `(agent, beliefs, rng_key) -> (q_pi, info)`
        returning policy weights and raw search output.
    """

    if search_algo is None:
        _require_mctx()
        search_algo = mctx.gumbel_muzero_policy

    def si_policy(agent: Any, beliefs: list[jnp.ndarray], rng_key: jnp.ndarray) -> tuple[jnp.ndarray, Any]:

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
def compute_neg_efe(
    qs: list[jnp.ndarray],
    action: jnp.ndarray,
    A: list[jnp.ndarray],
    A_dependencies: list[list[int]],
    B: list[jnp.ndarray],
    B_dependencies: list[list[int]],
    C: list[jnp.ndarray],
    use_states_info_gain: bool = True,
    use_utility: bool = True,
) -> tuple[jnp.ndarray, list[jnp.ndarray], list[jnp.ndarray]]:
    """Compute one-step negative expected free energy under an action.

    Parameters
    ----------
    qs : list[jnp.ndarray]
        Current posterior beliefs over hidden-state factors.
    action : jnp.ndarray
        Candidate action (multi-action index vector per batch element).
    A : list[jnp.ndarray]
        Observation model tensors.
    A_dependencies : list[list[int]]
        Modality-to-factor dependencies for `A`.
    B : list[jnp.ndarray]
        Transition model tensors.
    B_dependencies : list[list[int]]
        Factor-to-factor transition dependencies for `B`.
    C : list[jnp.ndarray]
        Preferences over observations.
    use_states_info_gain : bool, default=True
        Whether to include state information gain in EFE.
    use_utility : bool, default=True
        Whether to include expected utility in EFE.

    Returns
    -------
    tuple[jnp.ndarray, list[jnp.ndarray], list[jnp.ndarray]]
        Negative EFE, predicted next-state beliefs, and predicted
        next-observation beliefs.
    """
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
def get_prob_single_modality(o_m: jnp.ndarray, po_m: jnp.ndarray, distr_obs: bool) -> jnp.ndarray:
    """Compute observation likelihood for a single modality (observation and likelihood)"""
    return jnp.inner(o_m, po_m) if distr_obs else po_m[o_m]


def make_aif_recurrent_fn() -> Callable[[Any, jnp.ndarray, jnp.ndarray, Any], tuple[mctx.RecurrentFnOutput, Any]]:
    """Returns a recurrent_fn for an AIF agent."""
    _require_mctx()

    def recurrent_fn(
        agent: Any, rng_key: jnp.ndarray, action: jnp.ndarray, embedding: Any
    ) -> tuple[mctx.RecurrentFnOutput, Any]:
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
        if agent.categorical_obs:
            sample = lambda key, po, no: nn.one_hot(choice(key, po), no)
        else:
            sample = lambda key, po, no: choice(key, po)

        # set discount to outcome probabilities
        discount = 1.0
        obs = []
        for no_m, qo_m in zip(agent.num_obs, qo_next_pi):
            rng_key, key = jr.split(rng_key)
            o_m = sample(key, qo_m, no_m)
            discount *= get_prob_single_modality(o_m, qo_m, agent.categorical_obs)
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
def rollout(
    policy_search: Callable,
    agent: Any,
    env: Any,
    num_timesteps: int,
    rng_key: jnp.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Run a policy-search rollout loop for MCTS-based planning.

    Parameters
    ----------
    policy_search : Callable
        Planning callable that maps `(rng_key, agent, qs)` to policy weights.
    agent : Any
        Active inference agent.
    env : Any
        Environment exposing `step(...)` for batched transitions.
    num_timesteps : int
        Number of rollout steps.
    rng_key : jnp.ndarray
        Root JAX PRNG key.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], Any]
        Final carry dictionary, per-step rollout traces, and final environment.
    """
    # get the batch_size of the agent
    batch_size = agent.batch_size

    def step_fn(carry: dict[str, Any], x: jnp.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
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
