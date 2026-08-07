"""Mutual information metrics for RGM agents.

Computes I(observations; states) from Dirichlet parameters, matching
SPM's spm_MI from DEM_MNIST_RGM.m.  Used to gate Dirichlet updates during
training and to track learning progress.
"""

from jax import numpy as jnp

from pymdp.agent import Agent


# ---------------------------------------------------------------------------
# Mutual information helpers
# ---------------------------------------------------------------------------


def _mi_per_mapping_from_pA(pA_list: list[jnp.ndarray]) -> list[jnp.ndarray]:
    """Per-mapping MI I(obs; states) from a list of Dirichlet parameter arrays.

    For each modality array (n_patches, n_obs, n_states), each (patch, modality)
    slice is treated as one independent likelihood mapping and its
    MI = H(o) + H(s) - H(o,s) is returned *without* aggregating across patches or
    modalities. SPM gates each likelihood mapping independently rather than once
    across a whole level, so the MI gate needs this un-summed form.

    Returns a list (one entry per modality) of (n_patches,) MI arrays. JIT-safe.
    """
    per_mod = []
    for A_m in pA_list:
        # A_m shape: (n_patches, n_obs, n_states)
        totals = A_m.sum(axis=(1, 2), keepdims=True)  # (n_patches, 1, 1)
        A_norm = A_m / jnp.clip(totals, 1e-16, None)  # (n_patches, n_obs, n_states)
        log_A = jnp.log(jnp.clip(A_norm, 1e-16, None))
        joint = jnp.sum(A_norm * log_A, axis=(1, 2))  # (n_patches,)
        p_s = A_norm.sum(axis=1)  # (n_patches, n_states)
        p_o = A_norm.sum(axis=2)  # (n_patches, n_obs)
        h_s = jnp.sum(p_s * jnp.log(jnp.clip(p_s, 1e-16, None)), axis=1)
        h_o = jnp.sum(p_o * jnp.log(jnp.clip(p_o, 1e-16, None)), axis=1)
        per_mod.append(joint - h_s - h_o)  # (n_patches,)
    return per_mod


def _mi_from_pA(pA_list: list[jnp.ndarray]) -> jnp.ndarray:
    """Total MI across all patches and modalities (scalar; for reporting).

    Sums the per-mapping MI so the training MI-history plot keeps a single
    scalar per level. Matches spm_MI from SPM's DEM_MNIST_RGM.m.

    Returns a scalar jnp.ndarray (JIT-safe; no device→host sync).
    """
    return jnp.stack([m.sum() for m in _mi_per_mapping_from_pA(pA_list)]).sum()


def compute_level_mi(agent: Agent) -> float:
    """Compute MI I(observations; states) for an agent level.

    Uses pA (Dirichlet parameters) if available, otherwise falls back to A.

    Args:
        agent: Agent whose pA (or A) matrices define the joint distribution

    Returns:
        Mutual information in nats (Python float)
    """
    src = agent.pA if agent.pA is not None else agent.A
    return float(_mi_from_pA(src))
