import jax.numpy as jnp

from functools import partial
from typing import Optional, Tuple, Sequence
from jax import tree_util, nn, jit, vmap, lax
from jax.scipy.special import xlogy, digamma
from opt_einsum import contract
from multimethod import multimethod
from jaxtyping import ArrayLike
from jax.experimental import sparse
from jax.experimental.sparse._base import JAXSparse
import jax.experimental.sparse as jsparse

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


def compute_accuracy(qs: list[ArrayLike], obs: list[ArrayLike], A: list[ArrayLike]) -> ArrayLike:
    """Compute the accuracy portion of variational free energy.

    Parameters
    ----------
    qs: list[ArrayLike]
        Marginal state beliefs.
    obs: list[ArrayLike]
        Observations for each modality.
    A: list[ArrayLike]
        Likelihood tensors for each modality.

    Returns
    -------
    ArrayLike
        Expected log-likelihood term.
    """

    log_likelihood = compute_log_likelihood(obs, A)

    x = qs[0]
    for q in qs[1:]:
        x = jnp.expand_dims(x, -1) * q

    joint = log_likelihood * x
    return joint.sum()


def compute_free_energy(
    qs: list[ArrayLike], prior: list[ArrayLike], obs: list[ArrayLike], A: list[ArrayLike]
) -> ArrayLike:
    """
    Calculate variational free energy by breaking its computation down into three steps:
    1. computation of the negative entropy of the posterior -H[Q(s)]
    2. computation of the cross entropy of the posterior with the prior H_{Q(s)}[P(s)]
    3. computation of the accuracy E_{Q(s)}[lnP(o|s)]

    Then add them all together -- except subtract the accuracy

    Parameters
    ----------
    qs: list[ArrayLike]
        Marginal state beliefs per factor.
    prior: list[ArrayLike]
        Prior distributions per factor.
    obs: list[ArrayLike]
        Observations for each modality.
    A: list[ArrayLike]
        Likelihood tensors per modality.

    Returns
    -------
    ArrayLike
        Variational free energy value.
    """

    vfe = 0.0  # initialize variational free energy
    for q, p in zip(qs, prior):
        negH_qs = - stable_entropy(q)
        xH_qp = stable_cross_entropy(q, p)
        vfe += (negH_qs + xH_qp)
    
    vfe -= compute_accuracy(qs, obs, A)

    return vfe


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
