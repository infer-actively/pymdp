import jax.numpy as jnp

from functools import partial
from typing import Optional, Tuple, List
from jax import tree_util, nn, jit, vmap, lax
from jax.scipy.special import xlogy, digamma
from opt_einsum import contract
from multimethod import multimethod
from jaxtyping import ArrayLike
from jax.experimental import sparse
from jax.experimental.sparse._base import JAXSparse
import jax.experimental.sparse as jsparse

MINVAL = jnp.finfo(float).eps

def stable_xlogx(x):
    return xlogy(x, jnp.clip(x, MINVAL))

def stable_entropy(x):
    return - stable_xlogx(x).sum()

def stable_cross_entropy(x, y):
    return - xlogy(x, y).sum()

def log_stable(x):
    if isinstance(x, jsparse.BCOO):
        x = jsparse.todense(x)
    return jnp.log(jnp.clip(x, min=MINVAL))

@multimethod
@partial(jit, static_argnames=["keep_dims"])
def factor_dot(M: ArrayLike, xs: list[ArrayLike], keep_dims: Optional[tuple[int]] = None):
    """Dot product of a multidimensional array with `x`.
    Parameters
    ----------
    - `qs` [list of 1D numpy.ndarray] - list of jnp.ndarrays

    Returns
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """
    d = len(keep_dims) if keep_dims is not None else 0
    assert M.ndim == len(xs) + d
    keep_dims = () if keep_dims is None else keep_dims
    dims = tuple((i,) for i in range(M.ndim) if i not in keep_dims)
    return factor_dot_flex(M, xs, dims, keep_dims=keep_dims)


@multimethod
def factor_dot(M: JAXSparse, xs: List[ArrayLike], keep_dims: Optional[Tuple[int]] = None):
    d = len(keep_dims) if keep_dims is not None else 0
    assert M.ndim == len(xs) + d
    keep_dims = () if keep_dims is None else keep_dims
    dims = tuple((i,) for i in range(M.ndim) if i not in keep_dims)
    return spm_dot_sparse(M, xs, dims, keep_dims=keep_dims)


def spm_dot_sparse(
    X: JAXSparse, x: List[ArrayLike], dims: Optional[List[Tuple[int]]], keep_dims: Optional[List[Tuple[int]]]
):
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
def factor_dot_flex(M, xs, dims: List[Tuple[int]], keep_dims: Optional[Tuple[int]] = None):
    """Dot product of a multidimensional array with `x`.

    Parameters
    ----------
    - `M` [numpy.ndarray] - tensor
    - 'xs' [list of numpyr.ndarray] - list of tensors
    - 'dims' [list of tuples] - list of dimensions of xs tensors in tensor M
    - 'keep_dims' [tuple] - tuple of integers denoting dimesions to keep
    Returns
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """
    all_dims = tuple(range(M.ndim))
    matrix = [[xs[f], dims[f]] for f in range(len(xs))]
    args = [M, all_dims]
    for row in matrix:
        args.extend(row)

    args += [keep_dims]
    return contract(*args, backend="jax")


def get_likelihood_single_modality(o_m, A_m, distr_obs=True):
    """Return observation likelihood for a single observation modality m"""
    if distr_obs:
        expanded_obs = jnp.expand_dims(o_m, tuple(range(1, A_m.ndim)))
        likelihood = (expanded_obs * A_m).sum(axis=0)
    else:
        likelihood = A_m[o_m]

    return likelihood

def compute_log_likelihood_single_modality(o_m, A_m, distr_obs=True):
    """Compute observation log-likelihood for a single modality"""
    return log_stable(get_likelihood_single_modality(o_m, A_m, distr_obs=distr_obs))


def compute_log_likelihood(obs, A, distr_obs=True):
    """Compute likelihood over hidden states across observations from different modalities"""
    result = tree_util.tree_map(lambda o, a: compute_log_likelihood_single_modality(o, a, distr_obs=distr_obs), obs, A)
    ll = jnp.sum(jnp.stack(result), 0)

    return ll


def compute_log_likelihood_per_modality(obs, A, distr_obs=True):
    """Compute likelihood over hidden states across observations from different modalities, and return them per modality"""
    ll_all = tree_util.tree_map(lambda o, a: compute_log_likelihood_single_modality(o, a, distr_obs=distr_obs), obs, A)

    return ll_all


def compute_accuracy(qs, obs, A):
    """Compute the accuracy portion of the variational free energy (expected log likelihood under the variational posterior)"""

    log_likelihood = compute_log_likelihood(obs, A)

    x = qs[0]
    for q in qs[1:]:
        x = jnp.expand_dims(x, -1) * q

    joint = log_likelihood * x
    return joint.sum()


def compute_free_energy(qs, prior, obs, A):
    """
    Calculate variational free energy by breaking its computation down into three steps:
    1. computation of the negative entropy of the posterior -H[Q(s)]
    2. computation of the cross entropy of the posterior with the prior H_{Q(s)}[P(s)]
    3. computation of the accuracy E_{Q(s)}[lnP(o|s)]

    Then add them all together -- except subtract the accuracy
    """

    vfe = 0.0  # initialize variational free energy
    for q, p in zip(qs, prior):
        negH_qs = - stable_entropy(q)
        xH_qp = stable_cross_entropy(q, p)
        vfe += (negH_qs + xH_qp)
    
    vfe -= compute_accuracy(qs, obs, A)

    return vfe


def multidimensional_outer(arrs):
    """Compute the outer product of a list of arrays by iteratively expanding the first array and multiplying it with the next array"""

    x = arrs[0]
    for q in arrs[1:]:
        x = jnp.expand_dims(x, -1) * q

    return x

def _exact_wnorm(A):
    """
    Implements (-1) * eq. (D.15) in Da Costa et al. ‘Active inference on discrete state-spaces: A synthesis’, Journal of Mathematical Psychology, 2020.

    Note: Like the legacy SPM implementation this function clips A for numerical stability. However note that if some values of Aare set to zero e.g. by Bayesian model reduction, these are non-zeroed in this calculation, and thus contribute a large amount to the information gain unless these are zeroed when multiplying by beliefs about states and expected observations. In principle, this should be the case.
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

def spm_wnorm(A, exact_param_info_gain=True):
    """
    Returns the weight matrix used in PyMDP's parameter information-gain term.

    Historically this was the heuristic ``1/Σα − 1/α``. If exact_param_info_gain is set to *True* we instead return the exact value of
    the weight matrix used in the info gain computation defined in _exact_wnorm 
    while keeping the original function signature so that the rest of the codebase remains unchanged.
    """

    if exact_param_info_gain:
        return _exact_wnorm(A)

    """spm legacy heuristic for computing information-gain over parameters:
    Implements (-2) * second line of eq. (D.17) in Da Costa et al. ‘Active inference on discrete state-spaces: A synthesis’, Journal of Mathematical Psychology, 2020"""
    norm = 1. / A.sum(axis=0)
    avg = 1. / (A + MINVAL)
    wA = norm - avg
    return wA


def dirichlet_expected_value(dir_arr, event_dim=0):
    """
    Returns the expected value of Dirichlet parameters over a set of
    Categorical distributions, whose event/output dimension is stored in the axis of each array given by event_dim (default is 0).
    """
    dir_arr = jnp.clip(dir_arr, min=MINVAL)
    expected_val = jnp.divide(dir_arr, dir_arr.sum(axis=event_dim, keepdims=True))
    return expected_val


# Infer states hybrid

# NOTE: In this script, factor_dot_flex and factor_dot are JIT-compiled:
#   @partial(jit, static_argnames=["dims", "keep_dims"])
# In the edge deployment workflow, these functions were *not* JIT-compiled at this stage.

def compute_log_likelihoods_padded(obs_padded, A_padded):
    return log_stable(
        (jnp.expand_dims(obs_padded, tuple(range(obs_padded.ndim, A_padded.ndim))) * A_padded).sum(axis=1)
    )

def deconstruct_lls(lls_padded, A_shapes):
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

def compute_log_likelihoods_flat_block_diag_einsum(A_big: jnp.ndarray, obs_big: jnp.ndarray):
    """Step 1: Compute flat log-likelihoods using block diagonal matrix multiplication."""
    # ll_flat = A_big @ obs_big
    ll_flat = jnp.einsum('brc,bc->br', A_big, obs_big)
    if isinstance(ll_flat, jsparse.BCOO):
        ll_flat = jsparse.todense(ll_flat)
    return log_stable(ll_flat)

def compute_log_likelihoods_flat_block_diag(A_big: jnp.ndarray, obs_big: jnp.ndarray):
    """Step 1: Compute flat log-likelihoods using block diagonal matrix multiplication."""
    ll_flat = A_big @ obs_big
    if isinstance(ll_flat, jsparse.BCOO):
        ll_flat = jsparse.todense(ll_flat)
    return log_stable(ll_flat)

def deconstruct_log_likelihoods_block_diag(ll_flat: jnp.ndarray, state_shapes, cuts):
    """Step 2: Deconstruct flat log-likelihoods back to original shapes."""
    pieces = jnp.split(ll_flat, cuts, axis=1)
    out = [p.reshape(p.shape[0], *shape) for p, shape in zip(pieces, state_shapes)]
    return out

def compute_log_likelihoods_block_diag(A_big, obs_big, state_shapes, cuts, use_einsum=False):
    """Compute log‑likelihoods using block diagonal approach (atomic version)."""
    # obs_big = concatenate_observations_block_diag(obs_list)
    if use_einsum:
        ll_flat = compute_log_likelihoods_flat_block_diag_einsum(A_big, obs_big)
    else:
        ll_flat = vmap(compute_log_likelihoods_flat_block_diag)(A_big, obs_big)
    out = deconstruct_log_likelihoods_block_diag(ll_flat, state_shapes, cuts)
    return out


# Infer states end2end padded

def log_stable_sparse(x):
    if isinstance(x, jsparse.BCOO):
        x = x.sum_duplicates(nse=x.nse)
        return jsparse.BCOO((jnp.log(jnp.clip(x.data, min=MINVAL)), x.indices), shape=x.shape)
    return jnp.log(jnp.clip(x, min=MINVAL))
    
def compute_log_likelihood_per_modality_end2end2_padded(obs_padded, A_padded, sparsity):
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
