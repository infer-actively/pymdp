"""Classification level: top-level states → digit classes.

Builds a classification agent that maps top-level hierarchical states to
digit classes via a soft likelihood learned from exemplar label counts.
Also provides per-image bidirectional classify helpers used by RGMHierarchy.
"""

import jax
import jax.numpy as jnp
import numpy as np

from pymdp.agent import Agent

from state_stats import StateStats
from agent_build import _build_agent, _infer_map_states
from hierarchical import _bottom_up_pass, _top_down_refinement_pass


# ---------------------------------------------------------------------------
# Classification level: top-level states → digit classes
# ---------------------------------------------------------------------------


def create_classification_agent(
    top_stats: StateStats,
    y_exemplars: np.ndarray,
    n_classes: int = 10,
) -> tuple[Agent, jnp.ndarray]:
    """Build an agent mapping top-level states to digit classes.

    The A matrix encodes P(top_state | digit), learned from exemplar label
    counts. Ambiguous top states (shared by multiple digits) naturally spread
    posterior mass, capturing similarity structure (e.g. 1 and 7 share states).

    Args:
        top_stats: StateStats from the final hierarchical level (1x1 grid)
        y_exemplars: (N,) digit labels for the exemplar images
        n_classes: number of output classes

    Returns:
        (agent, valid_mask) where valid_mask is (1, n_classes) all-True
    """
    n_top_states = int(top_stats.num_states[0, 0])

    # A shape: (1 batch, n_top_states obs, n_classes hidden)
    # Count how many exemplars of each digit land in each top state
    A_np = np.zeros((1, n_top_states, n_classes))
    for s in range(n_top_states):
        for idx in top_stats.state_to_images[0][0][s]:
            A_np[0, s, int(y_exemplars[idx])] += 1

    # Normalize columns: P(top_state | digit)
    col_sums = A_np[0].sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1, col_sums)
    A_np[0] = A_np[0] / col_sums

    A = [jnp.array(A_np)]
    valid_mask = jnp.ones((1, n_classes), dtype=bool)
    # categorical_obs=True: observation is a soft distribution over top-level states
    agent = _build_agent(
        A, n_patches=1, max_states=n_classes, n_mod=1, valid_mask=valid_mask,
        categorical_obs=True,
    )
    return agent, valid_mask


def infer_classification(
    agent: Agent,
    valid_mask: jnp.ndarray,
    top_state: jnp.ndarray,
) -> jnp.ndarray:
    """Run classification inference: top-level state → digit posterior.

    Args:
        agent: Agent from create_classification_agent()
        valid_mask: (1, n_classes) boolean mask
        top_state: scalar top-level MAP state index

    Returns:
        (1, 1) MAP digit class index
    """
    n_top_states = agent.num_obs[0]
    obs_list = [jax.nn.one_hot(jnp.array([top_state]), n_top_states)]
    return _infer_map_states(agent, valid_mask, obs_list, (1, 1))


# ---------------------------------------------------------------------------
# Compiled per-image classify helpers (used by RGMHierarchy.classify)
# ---------------------------------------------------------------------------


def _classify_one(
    obs_i: jnp.ndarray,
    level_agents: tuple,
    level_masks: tuple,
    cls_agent,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-image bidirectional inference, safe for use under jax.lax.map.

    Runs one bottom-up pass, one initial classification, then one top-down
    refinement pass, and a final re-classification with the refined top beliefs.

    Args:
        obs_i: (n_groups, n_groups, max_components) encoded observations.
        level_agents: tuple of Agent objects for the hierarchical levels
            (L1 patch agent first, then L2+ agents; excludes cls agent).
        level_masks: tuple of (n_patches, max_states) boolean masks,
            one per hierarchical level.
        cls_agent: classification Agent.

    Returns:
        (predicted_digit, top_state, cls_beliefs) as on-device scalars/arrays.
    """
    n_hier = len(level_agents)

    # === Bottom-up pass ===
    level_soft_beliefs, level_obs_lists = _bottom_up_pass(level_agents, level_masks, obs_i)
    child_beliefs = level_soft_beliefs[-1]

    # === Initial classification ===
    qs_cls = cls_agent.infer_states([child_beliefs], cls_agent.D)
    q_cls = qs_cls[0][:, 0, :]  # (1, n_classes)

    # === Top-down refinement pass ===
    refined_soft = _top_down_refinement_pass(
        level_agents, level_masks, cls_agent, q_cls, level_soft_beliefs, level_obs_lists
    )

    refined_top = refined_soft[n_hier - 1]
    qs_cls_final = cls_agent.infer_states([refined_top], cls_agent.D)
    cls_beliefs = qs_cls_final[0][:, 0, :]  # (1, n_classes)

    predicted_digit = jnp.argmax(cls_beliefs[0])
    top_state = jnp.argmax(refined_top[0])
    return predicted_digit, top_state, cls_beliefs[0]


@jax.jit
def _classify_batch(
    level_agents: tuple,
    level_masks: tuple,
    cls_agent,
    obs_batch: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Map _classify_one over N images in a single compiled program.

    Decorated with @jax.jit so the first call compiles once and subsequent
    calls (even after training updates the agent weights) reuse the same XLA
    program — agent arrays flow through as dynamic inputs, not compile-time
    constants.  jax.lax.map keeps peak memory comparable to single-image
    inference by executing the body sequentially on-device.

    Args:
        level_agents: tuple of Agent objects (L1 … Ltop, no cls).
        level_masks: tuple of valid-state boolean masks, one per level.
        cls_agent: classification Agent.
        obs_batch: (N, n_groups, n_groups, max_components) encoded images.

    Returns:
        (predicted_digits, top_states, digit_beliefs) as JAX arrays.
    """
    def classify_one(obs_i):
        return _classify_one(obs_i, level_agents, level_masks, cls_agent)

    return jax.lax.map(classify_one, obs_batch)
