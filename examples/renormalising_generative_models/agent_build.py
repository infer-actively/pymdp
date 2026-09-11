"""Shared agent construction and inference primitives.

Low-level helpers for building batched pymdp Agents with identity transitions
and uniform priors, and for running inference to extract MAP states or full
posteriors.  Used by all levels of the RGM hierarchy.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

from pymdp.agent import Agent
from pymdp.control import Policies

from state_stats import StateStats


# ---------------------------------------------------------------------------
# Shared agent construction and inference
# ---------------------------------------------------------------------------


class RGMLevel(NamedTuple):
    """One level of the RGM hierarchy."""

    agent: Agent
    valid_mask: jnp.ndarray
    stats: StateStats


def _build_agent(
    A: list[jnp.ndarray],
    n_patches: int,
    max_states: int,
    n_mod: int,
    valid_mask: jnp.ndarray,
    learn_A: bool = False,
    pA: list[jnp.ndarray] | None = None,
    categorical_obs: bool = False,
) -> Agent:
    """Build a batched pymdp Agent with identity transitions and uniform prior.

    Args:
        A: list of n_mod observation likelihood arrays, each (n_patches, n_obs, max_states)
        n_patches: batch size
        max_states: hidden state dimension
        n_mod: number of observation modalities
        valid_mask: (n_patches, max_states) boolean mask
        learn_A: if True, enable Dirichlet learning for A matrices
        pA: Dirichlet concentration parameters (required when learn_A=True)
        categorical_obs: if True, observations are probability vectors (n_obs,) rather
            than integer indices; required for passing soft beliefs between levels

    Returns:
        Configured Agent
    """
    B_eye = jnp.broadcast_to(
        jnp.eye(max_states)[None, :, :, None],
        (n_patches, max_states, max_states, 1),
    )

    D_vals = jnp.where(valid_mask, 1.0, 0.0)
    D_vals = D_vals / D_vals.sum(axis=1, keepdims=True)

    return Agent(
        A=A,
        B=[B_eye],
        D=[D_vals],
        pA=pA,
        A_dependencies=[[0]] * n_mod,
        B_dependencies=[[0]],
        policies=Policies(jnp.zeros((1, 1, 1), dtype=jnp.int32)),
        num_controls=[1],
        batch_size=n_patches,
        inference_algo="fpi",
        num_iter=16,
        learn_A=learn_A,
        categorical_obs=categorical_obs,
    )


def _infer_map_states(
    agent: Agent,
    valid_mask: jnp.ndarray,
    obs_list: list[jnp.ndarray],
    output_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Run inference and return MAP state indices.

    Args:
        agent: Batched Agent
        valid_mask: (n_patches, max_states) boolean mask
        obs_list: list of n_mod arrays, each (n_patches,) integer observations
        output_shape: shape to reshape the result into

    Returns:
        MAP state indices reshaped to output_shape
    """
    qs = agent.infer_states(obs_list, agent.D)
    beliefs = qs[0][:, 0, :]
    masked = jnp.where(valid_mask, beliefs, -jnp.inf)
    return jnp.argmax(masked, axis=1).reshape(output_shape)


def _infer_beliefs(
    agent: Agent,
    valid_mask: jnp.ndarray,
    obs_list: list[jnp.ndarray],
) -> jnp.ndarray:
    """Run inference and return the full posterior (masked).

    Args:
        agent: Batched Agent
        valid_mask: (n_patches, max_states) boolean mask
        obs_list: list of n_mod arrays, each (n_patches,) integer observations

    Returns:
        (n_patches, max_states) posterior beliefs with padded states zeroed
    """
    qs = agent.infer_states(obs_list, agent.D)
    beliefs = qs[0][:, 0, :]
    return jnp.where(valid_mask, beliefs, 0.0)
