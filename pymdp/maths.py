import jax.numpy as jnp

from functools import partial
from typing import Optional, Tuple, Sequence
from jax import tree_util, nn, jit, vmap, lax
from jax.scipy.special import xlogy, digamma, gammaln
from opt_einsum import contract
from multimethod import multimethod
from jaxtyping import ArrayLike
from jax.experimental import sparse
from jax.experimental.sparse._base import JAXSparse
import jax.experimental.sparse as jsparse
from pymdp.utils import resolve_a_dependencies, resolve_b_dependencies

MINVAL = jnp.finfo(float).eps

def stable_xlogx(x: ArrayLike) -> ArrayLike:
    """Compute `x log(x)` with non-zero clipping.

    Parameters
    ----------
    x: ArrayLike
        Input tensor.

    Returns
    -------
    ArrayLike
        Elementwise `x * log(x)`.
    """
    return xlogy(x, jnp.clip(x, MINVAL))

def stable_entropy(x: ArrayLike) -> ArrayLike:
    """Compute entropy of a probability-like tensor.

    Parameters
    ----------
    x: ArrayLike
        Tensor of probabilities.

    Returns
    -------
    ArrayLike
        Entropy value.
    """
    return - stable_xlogx(x).sum()

def stable_cross_entropy(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """Compute cross-entropy between two tensors.

    Parameters
    ----------
    x: ArrayLike
        Source distribution tensor.
    y: ArrayLike
        Target distribution tensor.

    Returns
    -------
    ArrayLike
        Cross-entropy value.
    """
    return - xlogy(x, y).sum()

def log_stable(x: ArrayLike) -> ArrayLike:
    """Compute stable logarithm with minimum clipping.

    Parameters
    ----------
    x: ArrayLike
        Input tensor.

    Returns
    -------
    ArrayLike
        Log-transformed tensor with sparse support handled.
    """
    if isinstance(x, jsparse.BCOO):
        x = jsparse.todense(x)
    return jnp.log(jnp.clip(x, min=MINVAL))

@multimethod
@partial(jit, static_argnames=["keep_dims"])
def factor_dot(
    M: ArrayLike, xs: list[ArrayLike], keep_dims: Optional[tuple[int]] = None
) -> ArrayLike:
    """Dot product of a multidimensional array with `x`.

    Parameters
    ----------
    M: ArrayLike
        Input tensor to be contracted.
    xs: list[ArrayLike]
        Factors to contract against `M`.
    keep_dims: tuple[int] | None, optional
        Axes retained in the output.

    Returns
    -------
    ArrayLike
        Contracted tensor.
    """
    d = len(keep_dims) if keep_dims is not None else 0
    assert M.ndim == len(xs) + d
    keep_dims = () if keep_dims is None else keep_dims
    dims = tuple((i,) for i in range(M.ndim) if i not in keep_dims)
    return factor_dot_flex(M, xs, dims, keep_dims=keep_dims)


@multimethod
def factor_dot(
    M: JAXSparse, xs: list[ArrayLike], keep_dims: Optional[tuple[int]] = None
) -> ArrayLike:
    """Dot product of a sparse array with a list of factors.

    Parameters
    ----------
    M: JAXSparse
        Sparse input tensor.
    xs: list[ArrayLike]
        Factors to contract against `M`.
    keep_dims: tuple[int] | None, optional
        Axes retained in the output.

    Returns
    -------
    ArrayLike
        Contracted result.
    """
    d = len(keep_dims) if keep_dims is not None else 0
    assert M.ndim == len(xs) + d
    keep_dims = () if keep_dims is None else keep_dims
    dims = tuple((i,) for i in range(M.ndim) if i not in keep_dims)
    return spm_dot_sparse(M, xs, dims, keep_dims=keep_dims)


def spm_dot_sparse(
    X: JAXSparse,
    x: list[ArrayLike],
    dims: Optional[list[tuple[int]]],
    keep_dims: Optional[list[tuple[int]]],
) -> ArrayLike:
    """Sparse contraction helper used by :func:`factor_dot`.

    Parameters
    ----------
    X: JAXSparse
        Sparse tensor to contract.
    x: list[ArrayLike]
        Factors to contract against `X`.
    dims: list[tuple[int]] | None
        Input axes in `X` aligned to each entry in `x`.
    keep_dims: list[tuple[int]] | None
        Axes preserved in the output.

    Returns
    -------
    ArrayLike
        Contraction result.
    """
    if dims is None:
        dims = (jnp.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    dims = jnp.array(dims).flatten()

    if keep_dims is not None:
        for d in keep_dims:
            if d in dims:
                dims = jnp.delete(dims, jnp.argwhere(dims == d))

    for d in range(len(x)):
        s = jnp.ones(jnp.ndim(X), dtype=int)
        s = s.at[dims[d]].set(jnp.shape(x[d])[0])
        X = X * x[d].reshape(tuple(s))

    sparse_sum = sparse.sparsify(jnp.sum)
    Y = sparse_sum(X, axis=tuple(dims))
    return Y


@partial(jit, static_argnames=["dims", "keep_dims"])
def factor_dot_flex(
    M: ArrayLike,
    xs: list[ArrayLike],
    dims: list[tuple[int]],
    keep_dims: Optional[Tuple[int]] = None,
) -> ArrayLike:
    """Dot product of a multidimensional array with `x`.

    Parameters
    ----------
    M: Array
        Tensor to be contracted.
    xs: list[ArrayLike]
        Factors to contract against `M`.
    dims: list[tuple[int]]
        Axes in `M` aligned to each tensor in `xs`.
    keep_dims: tuple[int], optional
        Axes to retain in the output even if listed in `dims`.

    Returns
    -------
    Array
        Result of the contracted dot product.
    """
    all_dims = tuple(range(M.ndim))
    matrix = [[xs[f], dims[f]] for f in range(len(xs))]
    args = [M, all_dims]
    for row in matrix:
        args.extend(row)

    args += [keep_dims]
    return contract(*args, backend="jax")


def compute_log_likelihood_single_modality(
    o_m: ArrayLike, A_m: ArrayLike, distr_obs: bool = True
) -> ArrayLike:
    """Compute observation log-likelihood for a single modality.

    Parameters
    ----------
    o_m: ArrayLike
        Observation for one modality.
    A_m: ArrayLike
        Likelihood tensor for one modality.
    distr_obs: bool, default=True
        Interpret `o_m` as distribution if `True`, otherwise as discrete index.

    Returns
    -------
    ArrayLike
        Log-likelihood for this modality.
    """
    if distr_obs:
        expanded_obs = jnp.expand_dims(o_m, tuple(range(1, A_m.ndim)))
        log_likelihood = xlogy(expanded_obs, jnp.clip(A_m, min=MINVAL)).sum(axis=0)
    else:
        log_likelihood = log_stable(A_m[o_m])
    return log_likelihood


def compute_log_likelihood(
    obs: list[ArrayLike], A: list[ArrayLike], distr_obs: bool = True
) -> ArrayLike:
    """Compute likelihood over hidden states across observations from different modalities.

    Parameters
    ----------
    obs: list[ArrayLike]
        Observations for each modality.
    A: list[ArrayLike]
        Likelihood tensors for each modality.
    distr_obs: bool, default=True
        Interpret observations as distributions if `True`.

    Returns
    -------
    ArrayLike
        Combined log-likelihood over hidden states.
    """
    result = tree_util.tree_map(lambda o, a: compute_log_likelihood_single_modality(o, a, distr_obs=distr_obs), obs, A)
    ll = jnp.sum(jnp.stack(result), 0)

    return ll


def compute_log_likelihood_per_modality(
    obs: list[ArrayLike], A: list[ArrayLike], distr_obs: bool = True
) -> list[ArrayLike]:
    """Compute likelihood over hidden states per modality.

    Parameters
    ----------
    obs: list[ArrayLike]
        Observations for each modality.
    A: list[ArrayLike]
        Likelihood tensors for each modality.
    distr_obs: bool, default=True
        Interpret observations as distributions if `True`.

    Returns
    -------
    list[ArrayLike]
        Per-modality log-likelihood tensors.
    """
    ll_all = tree_util.tree_map(lambda o, a: compute_log_likelihood_single_modality(o, a, distr_obs=distr_obs), obs, A)

    return ll_all


def _to_dense_if_sparse(x: ArrayLike) -> ArrayLike:
    if isinstance(x, jsparse.BCOO):
        return jsparse.todense(x)
    return x


def _expected_log_prob(log_prob: ArrayLike, marginals: list[ArrayLike]) -> ArrayLike:
    if len(marginals) == 0:
        return jnp.asarray(_to_dense_if_sparse(log_prob)).sum()
    return factor_dot(log_prob, marginals)


def _expected_log_prob_tensor(log_prob: ArrayLike, belief: ArrayLike) -> ArrayLike:
    log_prob = jnp.asarray(_to_dense_if_sparse(log_prob))
    belief = jnp.asarray(_to_dense_if_sparse(belief))
    return factor_dot_flex(
        log_prob,
        [belief],
        dims=(tuple(range(log_prob.ndim)),),
    )


def _pad_sequence_with_initial_zeros(x: ArrayLike) -> ArrayLike:
    x = jnp.asarray(_to_dense_if_sparse(x))
    pad = jnp.zeros((1,) + x.shape[1:], dtype=x.dtype)
    return jnp.concatenate([pad, x], axis=0)


def _ensure_vfe_action_history_shape(
    past_actions: ArrayLike | None,
    num_factors: int,
) -> ArrayLike | None:
    if past_actions is None:
        return None

    past_actions = jnp.asarray(past_actions)
    if past_actions.ndim == 1:
        if num_factors == 1:
            return jnp.expand_dims(past_actions, -1)
        raise ValueError(
            "1D `past_actions` is only supported for single-factor action "
            "histories; multi-factor action histories must have shape "
            "(T-1, num_factors)"
        )

    if past_actions.ndim != 2:
        raise ValueError("`past_actions` must have shape (T-1, num_factors)")

    if past_actions.shape[1] != num_factors:
        raise ValueError(
            f"`past_actions` has second dimension {past_actions.shape[1]}, "
            f"expected {num_factors}"
        )

    return past_actions


def _sum_dirichlet_kl(
    q_dir: list[ArrayLike] | None,
    p_dir: list[ArrayLike] | None,
    *,
    event_dim: int = 0,
) -> ArrayLike:
    if q_dir is None and p_dir is None:
        return jnp.array(0.0)
    if q_dir is None or p_dir is None:
        raise ValueError("Dirichlet KL terms require both posterior and prior parameters")
    if len(q_dir) != len(p_dir):
        raise ValueError("Dirichlet posterior/prior lists must have the same length")

    total = jnp.array(0.0)
    for q_arr, p_arr in zip(q_dir, p_dir):
        if q_arr is None or p_arr is None:
            continue
        total = total + dirichlet_kl_divergence(q_arr, p_arr, event_dim=event_dim)
    return total


def compute_accuracy(
    qs: list[ArrayLike],
    obs: list[ArrayLike],
    A: list[ArrayLike],
    A_dependencies: list[list[int]] | None = None,
    distr_obs: bool = True,
) -> ArrayLike:
    """Compute the accuracy portion of variational free energy.

    Parameters
    ----------
    qs: list[ArrayLike]
        Marginal state beliefs.
    obs: list[ArrayLike]
        Observations for each modality.
    A: list[ArrayLike]
        Likelihood tensors for each modality.
    A_dependencies: list[list[int]] | None, optional
        Sparse modality-to-factor dependency mapping.
    distr_obs: bool, default=True
        Whether observations are already categorical distributions.

    Returns
    -------
    ArrayLike
        Expected log-likelihood term.
    """
    A_dependencies = resolve_a_dependencies(len(qs), len(A), A_dependencies)
    log_likelihoods = compute_log_likelihood_per_modality(obs, A, distr_obs=distr_obs)
    accuracy = 0.0

    for ll_m, deps_m in zip(log_likelihoods, A_dependencies):
        accuracy += _expected_log_prob(ll_m, [qs[f] for f in deps_m])

    return accuracy


def dirichlet_kl_divergence(
    q_dir: ArrayLike,
    p_dir: ArrayLike,
    event_dim: int = 0,
) -> ArrayLike:
    """Compute KL divergence between two Dirichlet distributions.

    Parameters
    ----------
    q_dir: ArrayLike
        Posterior Dirichlet concentration parameters.
    p_dir: ArrayLike
        Prior Dirichlet concentration parameters.
    event_dim: int, default=0
        Axis containing the categorical event dimension.

    Returns
    -------
    ArrayLike
        Scalar KL divergence summed over all conditional contexts.
    """
    q_dir = jnp.clip(_to_dense_if_sparse(q_dir), min=MINVAL)
    p_dir = jnp.clip(_to_dense_if_sparse(p_dir), min=MINVAL)

    event_dim = event_dim % q_dir.ndim
    q_sum = q_dir.sum(axis=event_dim)
    p_sum = p_dir.sum(axis=event_dim)
    digamma_q_sum = jnp.expand_dims(digamma(q_sum), axis=event_dim)

    kl = (
        gammaln(q_sum)
        - gammaln(p_sum)
        - gammaln(q_dir).sum(axis=event_dim)
        + gammaln(p_dir).sum(axis=event_dim)
        + ((q_dir - p_dir) * (digamma(q_dir) - digamma_q_sum)).sum(axis=event_dim)
    )

    return kl.sum()


def calc_vfe(
    qs: list[ArrayLike],
    prior: list[ArrayLike],
    *,
    obs: list[ArrayLike] | None = None,
    A: list[ArrayLike] | None = None,
    B: list[ArrayLike] | None = None,
    past_actions: ArrayLike | None = None,
    A_dependencies: list[list[int]] | None = None,
    B_dependencies: list[list[int]] | None = None,
    joint_qs: list[ArrayLike] | None = None,
    qA: list[ArrayLike] | None = None,
    pA: list[ArrayLike] | None = None,
    qB: list[ArrayLike] | None = None,
    pB: list[ArrayLike] | None = None,
    obs_valid_mask: ArrayLike | None = None,
    transition_valid_mask: ArrayLike | None = None,
    distr_obs: bool = True,
    return_decomposition: bool = False,
) -> tuple[ArrayLike, ArrayLike] | tuple[ArrayLike, ArrayLike, dict[str, ArrayLike]]:
    """Compute canonical variational free energy from a model/posterior pair.

    This function supports both:
    - single-step posteriors, where each `qs[f]` has shape `(num_states_f,)`, and
    - sequence posteriors, where each `qs[f]` has shape `(T, num_states_f)`.

    In the sequence case, transition contributions are assigned to the timestep
    they terminate at, so `vfe_t[t]` contains the `q(s_t)` entropy, the
    observation accuracy for `o_t`, and either the initial-prior term (at
    sequence starts) or the transition-model cross-entropy from `t-1 -> t`.

    Parameters
    ----------
    qs: list[ArrayLike]
        Posterior state marginals.
    prior: list[ArrayLike]
        Prior over initial hidden states (or single-step empirical prior).
    obs: list[ArrayLike] | None, optional
        Observations in distributional or discrete-index form.
    A: list[ArrayLike] | None, optional
        Likelihood tensors.
    B: list[ArrayLike] | None, optional
        Transition tensors.
    past_actions: ArrayLike | None, optional
        Action history with shape `(T-1, num_factors)` for sequence VFE.
    A_dependencies: list[list[int]] | None, optional
        Sparse modality-to-factor dependency mapping.
    B_dependencies: list[list[int]] | None, optional
        Sparse transition-factor dependency mapping.
    joint_qs: list[ArrayLike] | None, optional
        Optional pairwise posterior beliefs for sequence models. Each
        `joint_qs[f]` should have shape
        `(T-1, num_states[f], *[num_states[d] for d in B_dependencies[f]])`.
        When provided, the state-dependent terms of sequence VFE are computed
        from the full smoothed chain posterior rather than the mean-field
        product of adjacent marginals.
    qA, pA, qB, pB: list[ArrayLike] | None, optional
        Optional posterior/prior Dirichlet parameter pairs. When provided, the
        corresponding KL terms are added to the total `vfe`.
    obs_valid_mask: ArrayLike | None, optional
        Validity mask for padded observation windows.
    transition_valid_mask: ArrayLike | None, optional
        Validity mask for transitions in padded sequence windows.
    distr_obs: bool, default=True
        Whether observations are already categorical distributions.
    return_decomposition: bool, default=False
        If `True`, also return a dictionary of component terms.

    Returns
    -------
    tuple
        `(vfe_t, vfe)` by default. `vfe_t` has shape `(T,)` for sequences or
        scalar shape `()` for single-step posteriors. `vfe` is always scalar.
        When `return_decomposition=True`, a third element is returned with
        component arrays and optional parameter KL terms.
    """
    num_factors = len(qs)
    if len(prior) != num_factors:
        raise ValueError("`prior` must have one entry per hidden-state factor")

    if A is not None and obs is None:
        raise ValueError("`obs` must be provided when `A` is provided")
    if obs is not None and A is None:
        raise ValueError("`A` must be provided when `obs` is provided")

    A_dependencies = (
        None if A is None else resolve_a_dependencies(num_factors, len(A), A_dependencies)
    )
    B_dependencies = (
        None if B is None else resolve_b_dependencies(num_factors, B_dependencies)
    )

    is_sequence = qs[0].ndim > 1

    if joint_qs is not None:
        if not is_sequence:
            raise ValueError("`joint_qs` is only valid when `qs` is a sequence posterior")
        if len(joint_qs) != num_factors:
            raise ValueError("`joint_qs` must have one entry per hidden-state factor")

    def _state_prior_term(qs_t: list[ArrayLike]) -> ArrayLike:
        value = 0.0
        for q_f, p_f in zip(qs_t, prior):
            value += stable_cross_entropy(q_f, p_f)
        return value

    def _neg_entropy_term(qs_t: list[ArrayLike]) -> ArrayLike:
        value = 0.0
        for q_f in qs_t:
            value += -stable_entropy(q_f)
        return value

    def _accuracy_term(qs_t: list[ArrayLike], obs_t: list[ArrayLike] | None) -> ArrayLike:
        if A is None or obs_t is None:
            return jnp.array(0.0)
        return compute_accuracy(
            qs_t,
            obs_t,
            A,
            A_dependencies=A_dependencies,
            distr_obs=distr_obs,
        )

    if not is_sequence:
        neg_entropy_t = _neg_entropy_term(qs)
        prior_cross_entropy_t = _state_prior_term(qs)
        accuracy_t = _accuracy_term(qs, obs)
        transition_cross_entropy_t = jnp.array(0.0)
        vfe_t = neg_entropy_t + prior_cross_entropy_t - accuracy_t
    else:
        T = qs[0].shape[0]
        past_actions = _ensure_vfe_action_history_shape(past_actions, num_factors)
        if B is not None and past_actions is not None:
            expected_history_len = max(T - 1, 0)
            if past_actions.shape[0] != expected_history_len:
                raise ValueError(
                    "`past_actions` has leading dimension "
                    f"{past_actions.shape[0]}, expected {expected_history_len}"
                )

        if obs_valid_mask is None:
            obs_valid_mask = jnp.ones((T,), dtype=bool)
        else:
            obs_valid_mask = jnp.asarray(obs_valid_mask, dtype=bool)

        if transition_valid_mask is None:
            if T <= 1:
                transition_valid_mask = jnp.zeros((0,), dtype=bool)
            else:
                transition_valid_mask = obs_valid_mask[:-1] & obs_valid_mask[1:]
                if past_actions is not None:
                    transition_valid_mask = transition_valid_mask & jnp.all(
                        past_actions >= 0, axis=-1
                    )
        else:
            transition_valid_mask = jnp.asarray(transition_valid_mask, dtype=bool)

        start_mask = obs_valid_mask
        if B is not None:
            start_mask = obs_valid_mask & jnp.concatenate(
                [jnp.ones((1,), dtype=bool), jnp.logical_not(transition_valid_mask)],
                axis=0,
            )

        if joint_qs is not None:
            for f, joint_f in enumerate(joint_qs):
                if joint_f.shape[0] != max(T - 1, 0):
                    raise ValueError(
                        f"`joint_qs[{f}]` has leading dimension {joint_f.shape[0]}, "
                        f"expected {max(T - 1, 0)}"
                    )

        if B is not None and past_actions is None and T > 1:
            for B_f in B:
                if B_f.shape[-1] != 1:
                    raise ValueError(
                        "`past_actions` is required to compute sequence VFE "
                        "when transition tensors have more than one control state"
                    )

        padded_past_actions = (
            None if past_actions is None else _pad_sequence_with_initial_zeros(past_actions)
        )
        padded_joint_qs = (
            None
            if joint_qs is None
            else tree_util.tree_map(_pad_sequence_with_initial_zeros, joint_qs)
        )
        padded_prev_qs = tree_util.tree_map(
            lambda q: _pad_sequence_with_initial_zeros(q[:-1]),
            qs,
        )
        padded_transition_valid_mask = jnp.concatenate(
            [jnp.zeros((1,), dtype=bool), transition_valid_mask],
            axis=0,
        )

        def _scan_step(
            carry: None, t: ArrayLike
        ) -> tuple[
            None,
            tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike],
        ]:
            qs_t = tree_util.tree_map(lambda q: q[t], qs)
            obs_t = None if obs is None else tree_util.tree_map(lambda o: o[t], obs)

            valid_scale = obs_valid_mask[t].astype(jnp.result_type(*qs_t))
            start_scale = start_mask[t].astype(valid_scale.dtype)
            prior_term_t = _state_prior_term(qs_t) * valid_scale
            accuracy_term_t = _accuracy_term(qs_t, obs_t) * valid_scale

            neg_entropy_marginal_t = _neg_entropy_term(qs_t) * valid_scale
            if padded_joint_qs is None:
                neg_entropy_term_t = neg_entropy_marginal_t
            else:
                joint_qs_t = tree_util.tree_map(lambda q: q[t], padded_joint_qs)
                neg_entropy_cond_t = 0.0
                for joint_f_t in joint_qs_t:
                    parent_marginal_t = joint_f_t.sum(axis=0)
                    neg_entropy_cond_t += (
                        _expected_log_prob_tensor(log_stable(joint_f_t), joint_f_t)
                        - _expected_log_prob_tensor(
                            log_stable(parent_marginal_t), parent_marginal_t
                        )
                    )
                neg_entropy_cond_t = neg_entropy_cond_t * valid_scale
                neg_entropy_term_t = (
                    start_scale * neg_entropy_marginal_t
                    + (1.0 - start_scale) * neg_entropy_cond_t
                )

            if B is None:
                transition_term_t = jnp.array(0.0, dtype=neg_entropy_term_t.dtype)
            else:
                transition_term_t = 0.0
                actions_t = None if padded_past_actions is None else padded_past_actions[t]
                for f, deps_f in enumerate(B_dependencies):
                    B_f = _to_dense_if_sparse(B[f])
                    if actions_t is None:
                        B_f_t = B_f[..., 0]
                    else:
                        action_f = actions_t[f].astype(jnp.int32)
                        safe_action_f = jnp.where(action_f < 0, 0, action_f)
                        B_f_t = jnp.take(B_f, safe_action_f, axis=-1)

                    if padded_joint_qs is None:
                        transition_term_t += -_expected_log_prob(
                            log_stable(B_f_t),
                            [qs_t[f]] + [padded_prev_qs[d][t] for d in deps_f],
                        )
                    else:
                        joint_f_t = _to_dense_if_sparse(padded_joint_qs[f][t])
                        if joint_f_t.shape != B_f_t.shape:
                            raise ValueError(
                                f"`joint_qs[{f}]` has per-timestep shape {joint_f_t.shape}, "
                                f"expected {B_f_t.shape} to match the conditioned transition tensor"
                            )
                        transition_term_t += -_expected_log_prob_tensor(
                            log_stable(B_f_t), joint_f_t
                        )

                transition_term_t = (
                    transition_term_t
                    * padded_transition_valid_mask[t].astype(neg_entropy_term_t.dtype)
                    * (1.0 - start_scale)
                )

            if B is None:
                prior_cross_entropy_term_t = prior_term_t
                dynamics_term_t = prior_term_t
            else:
                prior_cross_entropy_term_t = prior_term_t * start_scale
                dynamics_term_t = prior_cross_entropy_term_t + transition_term_t

            vfe_term_t = neg_entropy_term_t + dynamics_term_t - accuracy_term_t

            return None, (
                neg_entropy_term_t,
                prior_cross_entropy_term_t,
                transition_term_t,
                accuracy_term_t,
                vfe_term_t,
            )

        _, scan_outputs = lax.scan(_scan_step, None, jnp.arange(T))
        (
            neg_entropy_t,
            prior_cross_entropy_t,
            transition_cross_entropy_t,
            accuracy_t,
            vfe_t,
        ) = scan_outputs

    parameter_kl_A = _sum_dirichlet_kl(qA, pA, event_dim=0)
    parameter_kl_B = _sum_dirichlet_kl(qB, pB, event_dim=0)
    parameter_kl = parameter_kl_A + parameter_kl_B
    vfe = vfe_t.sum() + parameter_kl

    if not return_decomposition:
        return vfe_t, vfe

    decomposition = {
        "neg_entropy_t": neg_entropy_t,
        "prior_cross_entropy_t": prior_cross_entropy_t,
        "transition_cross_entropy_t": transition_cross_entropy_t,
        "accuracy_t": accuracy_t,
        "state_vfe": vfe_t.sum(),
        "parameter_kl_A": parameter_kl_A,
        "parameter_kl_B": parameter_kl_B,
        "parameter_kl": parameter_kl,
    }
    return vfe_t, vfe, decomposition


def multidimensional_outer(arrs: list[ArrayLike]) -> ArrayLike:
    """Compute the outer product of a list of arrays by iterative expansion.

    Parameters
    ----------
    arrs: list[ArrayLike]
        List of arrays to combine.

    Returns
    -------
    ArrayLike
        Outer product tensor.
    """

    x = arrs[0]
    for q in arrs[1:]:
        x = jnp.expand_dims(x, -1) * q

    return x

def _exact_wnorm(A: ArrayLike) -> ArrayLike:
    """
    Implements (-1) * eq. (D.15) in Da Costa et al. ‘Active inference on discrete state-spaces: A synthesis’, Journal of Mathematical Psychology, 2020.

    Note: Like the legacy SPM implementation this function clips A for numerical stability. However note that if some values of Aare set to zero e.g. by Bayesian model reduction, these are non-zeroed in this calculation, and thus contribute a large amount to the information gain unless these are zeroed when multiplying by beliefs about states and expected observations. In principle, this should be the case.

    Parameters
    ----------
    A: ArrayLike
        Dirichlet-like array.

    Returns
    -------
    ArrayLike
        Exact parameter information-gain weight matrix.
    """
    # Clip once and reuse for numerical stability
    safe_A = jnp.clip(A, MINVAL)
    safe_sumA = jnp.clip(safe_A.sum(axis=0), MINVAL)

    wA = (
        jnp.log(safe_sumA) - jnp.log(safe_A)
        + 1. / safe_A - 1. / safe_sumA
        + digamma(safe_A) - digamma(safe_sumA)
    )

    return -wA # TODO: minus sign here gives negative info gain for backward compatibility with spm implementation. Later will need to remove minus sign here to get positive info gain and adjust function documentation accordingly.

def spm_wnorm(A: ArrayLike, exact_param_info_gain: bool = True) -> ArrayLike:
    """
    Returns the weight matrix used in PyMDP's parameter information-gain term.

    Historically this was the heuristic `1/Σα − 1/α`. If `exact_param_info_gain` is set to *True* we instead return the exact value of
    the weight matrix used in the info gain computation defined in _exact_wnorm 
    while keeping the original function signature so that the rest of the codebase remains unchanged.

    Parameters
    ----------
    A: ArrayLike
        Dirichlet concentration-like array.
    exact_param_info_gain: bool, default=True
        Choose exact (`True`) or legacy heuristic (`False`) form.

    Returns
    -------
    ArrayLike
        Parameter information-gain weight matrix.
    """

    if exact_param_info_gain:
        return _exact_wnorm(A)

    """spm legacy heuristic for computing information-gain over parameters:
    Implements (-2) * second line of eq. (D.17) in Da Costa et al. ‘Active inference on discrete state-spaces: A synthesis’, Journal of Mathematical Psychology, 2020"""
    norm = 1. / A.sum(axis=0)
    avg = 1. / (A + MINVAL)
    wA = norm - avg
    return wA


def dirichlet_expected_value(dir_arr: ArrayLike, event_dim: int = 0) -> ArrayLike:
    """
    Returns the expected value of Dirichlet parameters over a set of
    Categorical distributions, whose event/output dimension is stored in the axis of each array given by event_dim (default is 0).

    Parameters
    ----------
    dir_arr: ArrayLike
        Dirichlet parameters.
    event_dim: int, default=0
        Event axis to normalize over.

    Returns
    -------
    ArrayLike
        Expected Categorical probabilities.
    """
    dir_arr = jnp.clip(dir_arr, min=MINVAL)
    expected_val = jnp.divide(dir_arr, dir_arr.sum(axis=event_dim, keepdims=True))
    return expected_val


# Infer states hybrid

# NOTE: In this script, factor_dot_flex and factor_dot are JIT-compiled:
#   @partial(jit, static_argnames=["dims", "keep_dims"])
# In the edge deployment workflow, these functions were *not* JIT-compiled at this stage.

def compute_log_likelihoods_padded(obs_padded: ArrayLike, A_padded: ArrayLike) -> ArrayLike:
    """Compute padded log-likelihoods.

    Parameters
    ----------
    obs_padded: ArrayLike
        Padded observations.
    A_padded: ArrayLike
        Padded likelihood tensor.

    Returns
    -------
    ArrayLike
        Log-stable likelihood over padded input.
    """
    return log_stable(
        (jnp.expand_dims(obs_padded, tuple(range(obs_padded.ndim, A_padded.ndim))) * A_padded).sum(axis=1)
    )

def deconstruct_lls(
    lls_padded: ArrayLike, A_shapes: Sequence[tuple[int, ...]]
) -> list[ArrayLike]:
    """Split padded likelihood tensor into modality-specific blocks.

    Parameters
    ----------
    lls_padded: ArrayLike
        Combined padded log-likelihood tensor.
    A_shapes: Sequence[tuple[int, ...]]
        Original unpadded shapes per modality.

    Returns
    -------
    list[ArrayLike]
        One tensor per modality.
    """
    # Extract batch size from the first A_shape (all should have the same batch size)
    batch_size = lls_padded.shape[0] // len(A_shapes)
    lls = []
    for i, a_shape in enumerate(A_shapes):
        ll = lls_padded[i*batch_size : (i+1)*batch_size]
        idx = [
            slice(0, batch_size)
        ] + [
            slice(0, dim) for dim in a_shape[2:]
        ] + [
            0 for _ in range(ll.ndim - len(a_shape) + 1)
        ]
        lls.append(ll[tuple(idx)])
    return lls

# Infer states hybrid block

def compute_log_likelihoods_flat_block_diag_einsum(
    A_big: ArrayLike, obs_big: ArrayLike
) -> ArrayLike:
    """
    Compute flat log-likelihoods using block-diagonal einsum.

    Parameters
    ----------
    A_big: ArrayLike
        Block-diagonal likelihood matrix.
    obs_big: ArrayLike
        Block-diagonal observations.

    Returns
    -------
    ArrayLike
        Flat log-likelihoods.
    """
    # ll_flat = A_big @ obs_big
    ll_flat = jnp.einsum('brc,bc->br', A_big, obs_big)
    if isinstance(ll_flat, jsparse.BCOO):
        ll_flat = jsparse.todense(ll_flat)
    return log_stable(ll_flat)

def compute_log_likelihoods_flat_block_diag(A_big: ArrayLike, obs_big: ArrayLike) -> ArrayLike:
    """
    Compute flat log-likelihoods using block-diagonal multiplication.

    Parameters
    ----------
    A_big: ArrayLike
        Block-diagonal likelihood matrix.
    obs_big: ArrayLike
        Block-diagonal observations.

    Returns
    -------
    ArrayLike
        Flat log-likelihoods.
    """
    ll_flat = A_big @ obs_big
    if isinstance(ll_flat, jsparse.BCOO):
        ll_flat = jsparse.todense(ll_flat)
    return log_stable(ll_flat)

def deconstruct_log_likelihoods_block_diag(
    ll_flat: ArrayLike, state_shapes: Sequence[tuple[int, ...]], cuts: Sequence[int]
) -> list[ArrayLike]:
    """
    Split block-diagonal likelihood tensor into per-modality tensors.

    Parameters
    ----------
    ll_flat: ArrayLike
        Flat log-likelihood tensor.
    state_shapes: Sequence[tuple[int, ...]]
        Unwrapped state shapes per modality.
    cuts: Sequence[int]
        Boundary indices for each modality block.

    Returns
    -------
    list[ArrayLike]
        Reshaped per-modality likelihood tensors.
    """
    pieces = jnp.split(ll_flat, cuts, axis=1)
    out = [p.reshape(p.shape[0], *shape) for p, shape in zip(pieces, state_shapes)]
    return out

def compute_log_likelihoods_block_diag(
    A_big: ArrayLike,
    obs_big: ArrayLike,
    state_shapes: Sequence[tuple[int, ...]],
    cuts: Sequence[int],
    use_einsum: bool = False,
) -> list[ArrayLike]:
    """
    Compute log-likelihoods using a block-diagonal approach.

    Parameters
    ----------
    A_big: ArrayLike
        Block-diagonal likelihood matrix.
    obs_big: ArrayLike
        Concatenated block observations.
    state_shapes: Sequence[tuple[int, ...]]
        Shape of each modality block.
    cuts: Sequence[int]
        Cumulative cut indices.
    use_einsum: bool, default=False
        Use explicit einsum path if `True`.

    Returns
    -------
    list[ArrayLike]
        Per-modality log-likelihood tensors.
    """
    # obs_big = concatenate_observations_block_diag(obs_list)
    if use_einsum:
        ll_flat = compute_log_likelihoods_flat_block_diag_einsum(A_big, obs_big)
    else:
        ll_flat = vmap(compute_log_likelihoods_flat_block_diag)(A_big, obs_big)
    out = deconstruct_log_likelihoods_block_diag(ll_flat, state_shapes, cuts)
    return out


# Infer states end2end padded

def log_stable_sparse(x: ArrayLike) -> ArrayLike:
    """Compute numerically stable log for sparse or dense input.

    Parameters
    ----------
    x: ArrayLike
        Input tensor.

    Returns
    -------
    ArrayLike
        Elementwise log with sparse support preserved.
    """
    if isinstance(x, jsparse.BCOO):
        x = x.sum_duplicates(nse=x.nse)
        return jsparse.BCOO((jnp.log(jnp.clip(x.data, min=MINVAL)), x.indices), shape=x.shape)
    return jnp.log(jnp.clip(x, min=MINVAL))
    
def compute_log_likelihood_per_modality_end2end2_padded(
    obs_padded: ArrayLike, A_padded: ArrayLike, sparsity: str
) -> ArrayLike:
    """Compute padded end-to-end per-modality likelihood.

    Parameters
    ----------
    obs_padded: ArrayLike
        Padded observations.
    A_padded: ArrayLike
        Padded likelihood tensors.
    sparsity: str
        If `"ll_only"` return only dense log-likelihoods, else sparse variant.

    Returns
    -------
    ArrayLike
        Log-likelihood tensor.
    """
    likelihood = (
        jnp.expand_dims(obs_padded, tuple(range(obs_padded.ndim, A_padded.ndim))) * A_padded
    ).sum(axis=2)
    return log_stable(likelihood) if sparsity == 'll_only' else log_stable_sparse(likelihood)

if __name__ == "__main__":
    obs = [0, 1, 2]
    obs_vec = [nn.one_hot(o, 3) for o in obs]
    A = [jnp.ones((3, 2)) / 3] * 3
    res = jit(compute_log_likelihood)(obs_vec, A)

    print(res)
