#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""State inference and smoothing utilities for modern JAX-based pymdp agents.

This module provides:
- one-step posterior updates (`fpi`, `exact`, `ovf`),
- sequence-based inference (`mmp`, `vmp`),
- backward smoothing utilities for transition/posterior learning.

All public functions operate on JAX arrays and pytrees and are designed to
work with batched agent execution (`vmap`) and fixed-window sequence buffers.
"""

import jax.numpy as jnp
from typing import Any, TypedDict
from pymdp.algos import (
    run_factorized_fpi,
    run_mmp,
    run_vmp,
    hmm_smoother_from_filtered_colstoch,
)
from pymdp.maths import calc_vfe
from jax import tree_util as jtu, lax
from jax.experimental.sparse._base import JAXSparse
from jax.experimental import sparse
from jaxtyping import Array, ArrayLike

eps = jnp.finfo('float').eps

EXACT_METHOD = "exact"
ONE_STEP_METHODS = {"fpi", "ovf", EXACT_METHOD}
SEQUENCE_METHODS = {"mmp", "vmp"}
SMOOTHING_METHODS = {"ovf", EXACT_METHOD}


class VFEInfo(TypedDict):
    vfe_t: ArrayLike
    vfe: ArrayLike
    vfe_components: dict[str, ArrayLike]


def _select_current_obs(obs: list[Array] | Array, distr_obs: bool) -> list[Array] | Array:
    def _select_leaf(x: Array) -> Array:
        if x.ndim == 0:
            return x
        if distr_obs:
            # Distributional observations use the last axis for observations, so
            # 1D leaves are already a single-time-step observation.
            return x if x.ndim == 1 else x[-1]
        return x[-1]

    return jtu.tree_map(_select_leaf, obs)


def _truncate_for_horizon(
    obs: list[Array], past_actions: Array | None, inference_horizon: int | None
) -> tuple[list[Array], Array | None]:
    if inference_horizon is None:
        return obs, past_actions

    if inference_horizon < 1:
        raise ValueError("`inference_horizon` must be >= 1 when provided")

    obs = jtu.tree_map(lambda x: x[-inference_horizon:], obs)
    if past_actions is None:
        return obs, None

    action_horizon = max(inference_horizon - 1, 0)
    if action_horizon == 0:
        return obs, past_actions[:0]

    return obs, past_actions[-action_horizon:]


def _ensure_action_history_shape(past_actions: Array | None, num_factors: int) -> Array | None:
    if past_actions is None:
        return None

    if past_actions.ndim == 1:
        if num_factors == 1:
            return jnp.expand_dims(past_actions, -1)
        raise ValueError(
            "1D `past_actions` is only supported for single-factor action "
            "histories; multi-factor action histories must have shape "
            "(T-1, num_factors)"
        )

    if past_actions.ndim != 2:
        raise ValueError("`past_actions` must have shape (T-1, num_factors) per batch sample")

    if past_actions.shape[1] != num_factors:
        raise ValueError(
            f"`past_actions` has second dimension {past_actions.shape[1]}, expected {num_factors}"
        )

    return past_actions


def _build_sequence_validity_masks(
    obs: list[Array], past_actions: Array | None, valid_steps: int | Array | None
) -> tuple[Array, Array]:
    T = obs[0].shape[0]

    if valid_steps is None:
        obs_valid = jnp.ones((T,), dtype=bool)
    else:
        valid_steps = jnp.asarray(valid_steps, dtype=jnp.int32)
        k = jnp.clip(valid_steps, 1, T)
        obs_valid = jnp.arange(T) >= (T - k)

    if T <= 1:
        trans_valid = jnp.zeros((0,), dtype=bool)
    else:
        trans_valid = obs_valid[:-1] & obs_valid[1:]
        if past_actions is not None:
            action_valid = jnp.all(past_actions >= 0, axis=-1)
            trans_valid = trans_valid & action_valid

    return obs_valid, trans_valid


def _condition_transitions_on_actions(
    B: list[ArrayLike],
    past_actions: Array,
    invalid_action_mode: str = "neutral",
) -> list[Array]:
    nf = len(B)
    past_actions = _ensure_action_history_shape(past_actions, nf)
    actions_tree = [past_actions[:, i] for i in range(nf)]

    def _select_transitions_for_actions(b: ArrayLike, action_idx: Array) -> Array:
        if isinstance(b, JAXSparse):
            b = sparse.todense(b)

        action_idx = action_idx.astype(jnp.int32)
        safe_idx = jnp.where(action_idx < 0, 0, action_idx)
        selected = jnp.moveaxis(jnp.take(b, safe_idx, axis=-1), -1, 0)

        if invalid_action_mode == "neutral":
            invalid_transition = jnp.ones_like(b[..., 0], dtype=b.dtype)
            invalid_transition = invalid_transition / jnp.clip(
                invalid_transition.sum(axis=0, keepdims=True), min=eps
            )
        elif invalid_action_mode == "identity":
            transition = b[..., 0]
            if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
                raise ValueError(
                    "`invalid_action_mode='identity'` requires square 2D transitions "
                    "after action selection"
                )
            invalid_transition = jnp.eye(transition.shape[0], dtype=transition.dtype)
        else:
            raise ValueError(
                f"Unsupported invalid_action_mode `{invalid_action_mode}`. "
                "Expected one of {'neutral', 'identity'}."
            )

        invalid_transition = jnp.broadcast_to(invalid_transition, selected.shape)

        invalid = (action_idx < 0).reshape((action_idx.shape[0],) + (1,) * (selected.ndim - 1))
        return jnp.where(invalid, invalid_transition, selected)

    return [
        _select_transitions_for_actions(b, action_idx)
        for b, action_idx in zip(B, actions_tree)
    ]


def _run_one_step_inference(
    A: list[Array],
    obs: list[Array],
    prior: list[Array] | None,
    A_dependencies: list[list[int]] | None,
    *,
    method: str,
    num_iter: int,
    distr_obs: bool,
) -> tuple[list[Array], list[Array] | Array]:
    curr_obs = _select_current_obs(obs, distr_obs)
    fpi_num_iter = 1 if method == EXACT_METHOD else num_iter
    qs = run_factorized_fpi(
        A,
        curr_obs,
        prior,
        A_dependencies,
        num_iter=fpi_num_iter,
        distr_obs=distr_obs,
    )
    return qs, curr_obs


def _run_sequence_inference(
    A: list[Array],
    B: list[Array] | None,
    obs: list[Array],
    past_actions: Array | None,
    prior: list[Array] | None,
    A_dependencies: list[list[int]] | None,
    B_dependencies: list[list[int]] | None,
    *,
    method: str,
    num_iter: int,
    distr_obs: bool,
    valid_steps: int | Array | None,
) -> tuple[list[Array], Array | None, Array | None]:
    if B is None:
        raise ValueError(f"Sequence inference method `{method}` requires `B`.")

    obs_valid_mask = None
    transition_valid_mask = None
    history_len = max(obs[0].shape[0] - 1, 0)

    if past_actions is not None:
        past_actions = _ensure_action_history_shape(past_actions, len(B))
        if past_actions.shape[0] != history_len:
            raise ValueError(
                "`past_actions` has leading dimension "
                f"{past_actions.shape[0]}, expected {history_len}"
            )

    if valid_steps is not None:
        obs_valid_mask, transition_valid_mask = _build_sequence_validity_masks(
            obs, past_actions, valid_steps
        )

    conditioned_B = None
    if past_actions is not None:
        conditioned_B = _condition_transitions_on_actions(B, past_actions)
    elif history_len > 0:
        conditioned_B = []
        for b_f in B:
            if b_f.shape[-1] != 1:
                raise ValueError(
                    f"Sequence inference method `{method}` requires `past_actions` "
                    "when transition tensors have more than one control state"
                )
            single_transition = b_f[..., 0]
            conditioned_B.append(
                jnp.broadcast_to(single_transition, (history_len,) + single_transition.shape)
            )

    if method == "vmp":
        qs = run_vmp(
            A,
            conditioned_B,
            obs,
            prior,
            A_dependencies,
            B_dependencies,
            num_iter=num_iter,
            distr_obs=distr_obs,
            obs_valid_mask=obs_valid_mask,
            transition_valid_mask=transition_valid_mask,
        )
    else:
        qs = run_mmp(
            A,
            conditioned_B,
            obs,
            prior,
            A_dependencies,
            B_dependencies,
            num_iter=num_iter,
            distr_obs=distr_obs,
            obs_valid_mask=obs_valid_mask,
            transition_valid_mask=transition_valid_mask,
        )

    return qs, obs_valid_mask, transition_valid_mask


def _update_qs_history(
    qs_hist: list[Array] | None,
    qs: list[Array],
    *,
    method: str,
    inference_horizon: int | None,
) -> list[Array]:
    if qs_hist is not None:
        if method in ONE_STEP_METHODS:
            qs_hist = jtu.tree_map(
                lambda x, y: jnp.concatenate([x, jnp.expand_dims(y, 0)], 0),
                qs_hist,
                qs,
            )
            if inference_horizon is not None:
                qs_hist = jtu.tree_map(lambda x: x[-inference_horizon:], qs_hist)
            return qs_hist
        return qs

    if method in ONE_STEP_METHODS:
        return jtu.tree_map(lambda x: jnp.expand_dims(x, 0), qs)
    return qs


def _assemble_vfe_kwargs(
    *,
    method: str,
    qs: list[Array],
    prior: list[Array] | None,
    A: list[Array],
    B_for_vfe: list[Array] | None,
    curr_obs: list[Array] | Array | None,
    obs: list[Array],
    past_actions: Array | None,
    A_dependencies: list[list[int]] | None,
    B_dependencies: list[list[int]] | None,
    obs_valid_mask: Array | None,
    transition_valid_mask: Array | None,
    distr_obs: bool,
) -> dict[str, Any]:
    if method in ONE_STEP_METHODS:
        return {
            "qs": qs,
            "prior": prior,
            "obs": curr_obs,
            "A": A,
            "A_dependencies": A_dependencies,
            "distr_obs": distr_obs,
        }

    return {
        "qs": qs,
        "prior": prior,
        "obs": obs,
        "A": A,
        "B": B_for_vfe,
        "past_actions": past_actions,
        "A_dependencies": A_dependencies,
        "B_dependencies": B_dependencies,
        "obs_valid_mask": obs_valid_mask,
        "transition_valid_mask": transition_valid_mask,
        "distr_obs": distr_obs,
    }

def update_posterior_states(
    A: list[Array],
    B: list[Array] | None,
    obs: list[Array],
    past_actions: Array | None,
    prior: list[Array] | None = None,
    qs_hist: list[Array] | None = None,
    A_dependencies: list[list[int]] | None = None,
    B_dependencies: list[list[int]] | None = None,
    num_iter: int = 16,
    method: str = "fpi",
    distr_obs: bool = True,
    inference_horizon: int | None = None,
    valid_steps: int | Array | None = None,
    return_info: bool = False,
) -> list[Array] | tuple[list[Array], VFEInfo]:
    """Infer posterior beliefs over hidden states from observations.

    Parameters
    ----------
    A : list[Array]
        Observation likelihood tensors per modality.
    B : list[Array] | None
        Transition model tensors per hidden-state factor. For one-step methods,
        this can be provided unchanged. For sequence methods and non-`None`
        `past_actions`, transitions are conditioned per timestep.
    obs : pytree
        Observation sequence or single-step observation in distributional form
        (for example, one-hot vectors per modality).
    past_actions : Array | None
        Action history with shape `(T-1, num_factors)` for sequence methods.
        Can be `None` when no valid history is available.
    prior : list[Array], optional
        Prior beliefs over hidden states. Required when `return_info=True`
        so canonical VFE diagnostics can be computed.
    qs_hist : list[Array], optional
        Existing posterior history buffer. If provided, one-step updates append
        to this history.
    A_dependencies : list[list[int]], optional
        Sparse modality-to-factor dependency mapping.
    B_dependencies : list[list[int]], optional
        Sparse transition-factor dependency mapping.
    num_iter : int, default=16
        Number of variational update iterations.
    method : {"fpi", "ovf", "mmp", "vmp", "exact"}, default="fpi"
        Inference routine to execute.
    distr_obs : bool, default=True
        Whether observations are already distributional.
    inference_horizon : int | None, optional
        Optional truncation horizon for sequence inference.
    valid_steps : int | Array | None, optional
        Number of valid (unpadded) timesteps for fixed-window sequence inputs.
    return_info : bool, default=False
        If `True`, also return an info dictionary containing canonical VFE
        diagnostics (`vfe_t`, `vfe`, and component terms). For forward-only
        methods (`fpi`, `ovf`, `exact`) this reports the current posterior /
        history returned by the inference call; if you need a full smoothed
        sequence VFE for `ovf` or `exact`, first run `smoothing_ovf(...)` or
        `smoothing_exact(...)` and then call `pymdp.maths.calc_vfe(...,
        joint_qs=...)`.

    Returns
    -------
    list[Array] or tuple[list[Array], VFEInfo]
        Posterior state beliefs, with shape semantics depending on `method`:
        one-step methods return/append a time axis, sequence methods return full
        sequence posteriors. If `return_info=True`, a second return value
        contains canonical VFE diagnostics.
    """
    B_for_vfe = B
    info = None
    curr_obs = None
    obs_valid_mask = None
    transition_valid_mask = None

    if return_info and prior is None:
        raise ValueError("`prior` must be provided when `return_info=True`")

    if method in SEQUENCE_METHODS:
        obs, past_actions = _truncate_for_horizon(obs, past_actions, inference_horizon)

    if method in ONE_STEP_METHODS:
        qs, curr_obs = _run_one_step_inference(
            A,
            obs,
            prior,
            A_dependencies,
            method=method,
            num_iter=num_iter,
            distr_obs=distr_obs,
        )
    elif method in SEQUENCE_METHODS:
        qs, obs_valid_mask, transition_valid_mask = _run_sequence_inference(
            A,
            B,
            obs,
            past_actions,
            prior,
            A_dependencies,
            B_dependencies,
            method=method,
            num_iter=num_iter,
            distr_obs=distr_obs,
            valid_steps=valid_steps,
        )
    else:
        raise ValueError(
            f"Unsupported inference method `{method}`. "
            "Expected one of {'fpi', 'ovf', 'mmp', 'vmp', 'exact'}."
        )

    qs_hist = _update_qs_history(
        qs_hist,
        qs,
        method=method,
        inference_horizon=inference_horizon,
    )

    if return_info:
        vfe_kwargs = _assemble_vfe_kwargs(
            method=method,
            qs=qs,
            prior=prior,
            A=A,
            B_for_vfe=B_for_vfe,
            curr_obs=curr_obs,
            obs=obs,
            past_actions=past_actions,
            A_dependencies=A_dependencies,
            B_dependencies=B_dependencies,
            obs_valid_mask=obs_valid_mask,
            transition_valid_mask=transition_valid_mask,
            distr_obs=distr_obs,
        )
        vfe_t, vfe, decomposition = calc_vfe(
            **vfe_kwargs,
            return_decomposition=True,
        )

        info = {
            "vfe_t": vfe_t,
            "vfe": vfe,
            "vfe_components": decomposition,
        }

    if return_info:
        return qs_hist, info
    return qs_hist

def _joint_dist_factor(
    conditioned_transitions: ArrayLike, filtered_qs: list[Array]
) -> tuple[Array, Array]:
    qs_last = filtered_qs[-1]
    qs_filter = filtered_qs[:-1]
    n_transitions = conditioned_transitions.shape[0]

    def step_fn(qs_smooth: Array, xs: tuple[Array, Array]) -> tuple[Array, tuple[Array, Array]]:
        qs_f, time_b = xs
        qs_j = time_b * qs_f
        norm = qs_j.sum(-1, keepdims=True)
        if isinstance(norm, JAXSparse):
            norm = sparse.todense(norm)
        norm = jnp.where(norm == 0, eps, norm)
        qs_backward_cond = qs_j / norm
        qs_joint = qs_backward_cond * jnp.expand_dims(qs_smooth, -1)
        qs_smooth = qs_joint.sum(-2)
        if isinstance(qs_smooth, JAXSparse):
            qs_smooth = sparse.todense(qs_smooth)

        return qs_smooth, (qs_smooth, qs_joint)

    _, seq_qs = lax.scan(
        step_fn,
        qs_last,
        (qs_filter, conditioned_transitions),
        reverse=True,
        unroll=2
    )

    qs_smooth_all = jnp.concatenate([seq_qs[0], jnp.expand_dims(qs_last, 0)], 0)
    qs_joint_all = seq_qs[1]
    if isinstance(qs_joint_all, JAXSparse):
        qs_joint_all.shape = (n_transitions,) + qs_joint_all.shape
    return qs_smooth_all, qs_joint_all


def joint_dist_factor(
    b: ArrayLike,
    filtered_qs: list[Array],
    actions: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Compute smoothed marginals and pairwise joints for one factor.

    Parameters
    ----------
    b : Array
        Either an action-conditioned transition sequence
        `(T-1, K_next, K_curr)` or a transition tensor with action axis
        `(..., K_next, K_curr, n_actions)`.
    filtered_qs : list[Array]
        Filtered posterior sequence for this factor with leading time axis.
    actions : Array | None, optional
        Optional action sequence to select transitions from `b`.

    Returns
    -------
    tuple[Array, Array]
        `(smoothed_marginals, pairwise_joints)` where:
        - smoothed marginals have shape `(T, K)`
        - joints have shape `(T-1, K_next, K_curr)`
    """
    if actions is None:
        conditioned_transitions = b
    else:
        action_idx = jnp.asarray(actions).astype(jnp.int32)
        conditioned_transitions = jnp.moveaxis(
            jnp.take(b, action_idx, axis=-1), -1, 0
        )
    return _joint_dist_factor(conditioned_transitions, filtered_qs)


def smoothing_ovf(
    filtered_post: list[Array], B: list[Array], past_actions: Array | None
) -> tuple[list[Array], list[Array]]:
    """Run backward smoothing for factorized online variational filtering history.

    Parameters
    ----------
    filtered_post : list[Array]
        Filtering posteriors per factor, each with shape `(T, K_f)` (or
        equivalent leading-time layout for a batch element).
    B : list[Array]
        Transition tensors per factor.
    past_actions : Array
        Action history with shape `(T-1, num_factors)`.

    Returns
    -------
    tuple[list[Array], list[Array]]
        `(marginals, joints)` per factor.
    """
    assert len(filtered_post) == len(B)
    nf = len(B)  # number of factors

    if past_actions is not None:
        past_actions = _ensure_action_history_shape(past_actions, nf)
    B_seq = _condition_transitions_on_actions(B, past_actions, invalid_action_mode="identity")

    marginals_and_joints = ([], [])
    for b_seq, qs in zip(B_seq, filtered_post):
        marginals, joints = joint_dist_factor(b_seq, qs, actions=None)
        marginals_and_joints[0].append(marginals)
        marginals_and_joints[1].append(joints)

    return marginals_and_joints


def smoothing_exact(
    filtered_post: list[Array], B: list[Array], past_actions: Array | None
) -> tuple[list[Array], list[Array]]:
    """
    Exact single-factor HMM backward smoothing from online filtering history.

    Parameters
    ----------
    filtered_post:
        List containing one `(T, K)` array of filtering marginals.
    B:
        List containing one transition tensor in pymdp column-stochastic orientation.
    past_actions:
        `(T-1, 1)` or `(T-1,)` action history used to select transitions.

    Returns
    -------
    (marginals, joints):
        marginals: list with one `(T, K)` smoothed marginal array.
        joints: list with one `(T-1, K_next, K_curr)` pairwise posterior array.
    """
    if len(filtered_post) != 1 or len(B) != 1:
        raise ValueError("smoothing_exact currently supports only one hidden-state factor")

    filtered = filtered_post[0]
    B_seq = _condition_transitions_on_actions(
        B,
        past_actions,
        invalid_action_mode="identity",
    )[0]
    

    smoothed, joint_next_curr, _ = hmm_smoother_from_filtered_colstoch(
        filtered,
        B_seq,
        return_trans_probs=False,
    )

    return [smoothed], [joint_next_curr]


    
