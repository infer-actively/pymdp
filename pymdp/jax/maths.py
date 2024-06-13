import jax.numpy as jnp

from functools import partial
from typing import Optional, Tuple, List
from jax import tree_util, nn, jit
from opt_einsum import contract
from multimethod import multimethod
from jaxtyping import Array
from jax.experimental import sparse
from jax.experimental.sparse._base import JAXSparse

MINVAL = jnp.finfo(float).eps


def log_stable(x):
    return jnp.log(jnp.clip(x, min=MINVAL))


@multimethod
@partial(jit, static_argnames=["keep_dims"])
def factor_dot(M: Array, xs: List[Array], keep_dims: Optional[Tuple[int]] = None):
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
def factor_dot(M: JAXSparse, xs: List[Array], keep_dims: Optional[Tuple[int]] = None):
    d = len(keep_dims) if keep_dims is not None else 0
    assert M.ndim == len(xs) + d
    keep_dims = () if keep_dims is None else keep_dims
    dims = tuple((i,) for i in range(M.ndim) if i not in keep_dims)
    return spm_dot_sparse(M, xs, dims, keep_dims=keep_dims)


def spm_dot_sparse(
    X: JAXSparse, x: List[Array], dims: Optional[List[Tuple[int]]], keep_dims: Optional[List[Tuple[int]]]
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


def compute_log_likelihood_single_modality(o_m, A_m, distr_obs=True):
    """Compute observation likelihood for a single modality (observation and likelihood)"""
    if distr_obs:
        expanded_obs = jnp.expand_dims(o_m, tuple(range(1, A_m.ndim)))
        likelihood = (expanded_obs * A_m).sum(axis=0)
    else:
        likelihood = A_m[o_m]

    return log_stable(likelihood)


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

    ll = compute_log_likelihood(obs, A)

    x = qs[0]
    for q in qs[1:]:
        x = jnp.expand_dims(x, -1) * q

    joint = ll * x
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
        negH_qs = q.dot(log_stable(q))
        xH_qp = -q.dot(log_stable(p))
        vfe += negH_qs + xH_qp

    vfe -= compute_accuracy(qs, obs, A)

    return vfe


def multidimensional_outer(arrs):
    """Compute the outer product of a list of arrays by iteratively expanding the first array and multiplying it with the next array"""

    x = arrs[0]
    for q in arrs[1:]:
        x = jnp.expand_dims(x, -1) * q

    return x


def spm_wnorm(A):
    """
    Returns Expectation of logarithm of Dirichlet parameters over a set of
    Categorical distributions, stored in the columns of A.
    """
    A = jnp.clip(A, min=MINVAL)
    norm = 1.0 / A.sum(axis=0)
    avg = 1.0 / A
    wA = norm - avg
    return wA


def dirichlet_expected_value(dir_arr):
    """
    Returns Expectation of Dirichlet parameters over a set of
    Categorical distributions, stored in the columns of A.
    """
    expected_val = jnp.divide(dir_arr, jnp.clip(dir_arr.sum(axis=0, keepdims=True), min=MINVAL))
    return expected_val


if __name__ == "__main__":
    obs = [0, 1, 2]
    obs_vec = [nn.one_hot(o, 3) for o in obs]
    A = [jnp.ones((3, 2)) / 3] * 3
    res = jit(compute_log_likelihood)(obs_vec, A)

    print(res)
