#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

""" Functions
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import itertools
import numpy as np
from scipy import special
from inferactively.core import utils

def spm_dot(X, x, dims_to_omit=None, obs_mode=False):
    """ Dot product of a multidimensional array with `x`
        The dimensions in `dims_to_omit` will not be summed across during the dot product

    @TODO: we should look for an alternative to obs_mode
    
    Parameters
    ----------
    - `x` [1D numpy.ndarray] - either vector or array of arrays
        The alternative array to perform the dot product with
    - `dims_to_omit` [list :: int] (optional)
        Which dimensions to omit
    
    Returns 
    -------
    - `Y` [1D numpy.ndarray] - the result of the dot product
    """

    # construct dims to perform dot product on
    if utils.is_arr_of_arr(x):
        dims = (np.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    else:
        if obs_mode is True:
            """
            @NOTE Case when you're getting the likelihood of an observation under the generative model.
                  Equivalent to something like self.values[np.where(x),:]
                  when `x` is a discrete 'one-hot' observation vector
            """
            dims = np.array([0], dtype=int)
        else:
            """
            @NOTE Case when `x` leading dimension matches the lagging dimension of `values`
                  E.g. a more 'classical' dot product of a likelihood with hidden states
            """
            dims = np.array([1], dtype=int)

        x = utils.to_arr_of_arr(x)

    # delete ignored dims
    if dims_to_omit is not None:
        if not isinstance(dims_to_omit, list):
            raise ValueError("`dims_to_omit` must be a `list` of `int`")
        dims = np.delete(dims, dims_to_omit)
        if len(x) == 1:
            x = np.empty([0], dtype=object)
        else:
            x = np.delete(x, dims_to_omit)

    # compute dot product
    for d in range(len(x)):
        s = np.ones(np.ndim(X), dtype=int)
        s[dims[d]] = np.shape(x[d])[0]
        X = X * x[d].reshape(tuple(s))
        X = np.sum(X, axis=dims[d], keepdims=True)
    Y = np.squeeze(X)

    # check to see if `Y` is a scalar
    if np.prod(Y.shape) <= 1.0:
        Y = Y.item()
        Y = np.array([Y]).astype("float64")

    return Y


def spm_cross(X, x=None, *args):
    """ Multi-dimensional outer product
    
    Parameters
    ----------
    - `x` [np.ndarray] || [Categorical] (optional)
        The values to perfrom the outer-product with. If empty, then the
        outer-product is taken between X and itself. If x is not empty, then outer product is 
        taken between X and the various dimensions of x.
    - `args` [np.ndarray] || [Categorical] (optional)
        Remaining arrays to perform outer-product with. These extra arrays are recursively multiplied 
        with the 'initial' outer product (that between X and x).
    
    Returns
    -------
    - `y` [np.ndarray] || [Categorical]
          The result of the outer-product
    """

    if len(args) == 0 and x is None:
        if utils.is_arr_of_arr(X):
            Y = spm_cross(*list(X))
        elif np.issubdtype(X.dtype, np.number):
            Y = X
        return Y

    if utils.is_arr_of_arr(X):
        X = spm_cross(*list(X))

    if x is not None and utils.is_arr_of_arr(x):
        x = spm_cross(*list(x))

    reshape_dims = tuple(list(X.shape) + list(np.ones(x.ndim, dtype=int)))
    A = X.reshape(reshape_dims)

    reshape_dims = tuple(list(np.ones(X.ndim, dtype=int)) + list(x.shape))
    B = x.reshape(reshape_dims)
    Y = np.squeeze(A * B)

    for x in args:
        Y = spm_cross(Y, x)
    return Y


def spm_norm(A):
    """ 
    Returns normalization of Categorical distribution, 
    stored in the columns of A.
    """
    A = A + 1e-16
    normed_A = np.divide(A,A.sum(axis=0))
    return normed_A

def spm_wnorm(A):
    """ 
    Returns Expectation of logarithm of Dirichlet parameters over a set of Categorical distributions, 
    stored in the columns of A.
    """
    A = A + 1e-16
    norm = np.divide(1.0, np.sum(A, axis=0))
    avg = np.divide(1.0, A)
    wA = norm - avg
    return wA


def spm_betaln(z):
    """ Log of the multivariate beta function of a vector.

     @NOTE this function computes across columns if `z` is a matrix
    """
    return np.sum(special.gammaln(z), axis=0) - special.gammaln(np.sum(z, axis=0))


def calc_free_energy(qs, prior, n_factors, likelihood=None):
    """ Calculate variational free energy

    @TODO Primarily used in FPI algorithm, needs to be made general
    """
    free_energy = 0
    for factor in range(n_factors):
        negH_qs = qs[factor].dot(np.log(qs[factor][:, np.newaxis] + 1e-16)) #  (neg)entropy of posterior marginal H(q[f])
        xH_qp = -qs[factor].dot(prior[factor][:, np.newaxis])             #  cross entropy of posterior marginal with  prior marginal H(q[f],p[f])
        free_energy += negH_qs + xH_qp

    if likelihood is not None:
        accuracy = spm_dot(likelihood, qs)[0]
        free_energy -= accuracy
    return free_energy
    
def calc_free_energy_policy(A, B, obs_t, qs_policy, policy, curr_t, t_horizon, T, previous_actions=None):
    """ 
    Calculate variational free energy for a given policy
    """

    # extract dimensions of observation modalities and number of levels per modality. Do the same for hidden states
    if utils.is_arr_of_arr(obs_t[0]):
        n_observations = [ obs_t_i.shape[0] for obs_t_i in obs_t[0] ]
    else:
        n_observations = [obs_t[0].shape[0]]
    
    if utils.is_arr_of_arr(qs_t[0][0]):
        n_states = [ qs_t_i.shape[0] for qs_t_i in qs_t[0][0] ]
    else:
        n_states = [qs_t[0][0].shape[0]]
    
    n_modalities = len(n_observations)
    n_factors = len(n_states)

    # compute time-window, taking into account boundary conditions - this range goes from start of time-window (in absolute time) to current absolute time index
    obs_range = range(max(0,curr_t-t_horizon),curr_t)

    # likelihood of observations under configurations of hidden causes (over time)
    likelihood = np.empty(len(obs_range), dtype = object)
    for t in range(len(obs_range)):
        likelihood_t = np.ones(tuple(n_states))

        if n_modalities is 1:
            likelihood_t *= spm_dot(A, obs_t[obs_range[t]], obs_mode=True)
        else:
            for modality in range(n_modalities):
                likelihood_t *= spm_dot(A[modality], obs[obs_range[t]][modality], obs_mode=True)
        likelihood[t] = np.log(likelihood_t + 1e-16)
    
    if previous_actions is None:
        full_policy = policy
    else:
        full_policy = np.vstack( (previous_actions, policy))
    
    F_pol = 0.0
    for t in range(1, len(qs_policy)):

        lnBpast_tensor = np.empty(n_factors,dtype=object)
        for f in range(n_factors):
            lnBpast_tensor[f] = B[f][:,:,policy[t-1,f].dot(qs_policy[t-1][f])
        
        F_pol += calc_free_energy(qs_policy[t], lnBpast_tensor, n_factors, likelihood[t])

    return F_pol

def softmax(dist, return_numpy=True):
    """ Computes the softmax function on a set of values

    """
    if utils.is_distribution(dist):
        if dist.IS_AOA:
            output = []
            for i in range(len(dist.values)):
                output[i] = softmax(dist.values[i], return_numpy=True)
            output = utils.to_categorical(np.array(output))
        else:
            dist = np.copy(dist.values)
    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    if return_numpy:
        return output
    else:
        return utils.to_categorical(output)


def kl_divergence(q, p):
    """ Calculate KL divdivergence between two distributions

    @TODO: make this work for multi-dimensional arrays
    """
    q.remove_zeros()
    p.remove_zeros()
    q = np.copy(q.values)
    p = np.copy(p.values)
    kl = np.sum(q * np.log(q / p), axis=0)[0]
    return kl

def spm_MDP_G(A, x):
    """
    Calculates the Bayesian surprise in the same way as spm_MDP_G.m does in 
    the original matlab code.
    
    Parameters
    ----------
    A (numpy ndarray or array-object):
        array assigning likelihoods of observations/outcomes under the various hidden state configurations
    
    x (numpy ndarray or array-object):
        Categorical distribution presenting probabilities of hidden states (this can also be interpreted as the 
        predictive density over hidden states/causes if you're calculating the 
        expected Bayesian surprise)
        
    Returns
    -------
    G (float):
        the (expected or not) Bayesian surprise under the density specified by x --
        namely, this scores how much an expected observation would update beliefs about hidden states
        x, were it to be observed. 
    """
    if A.dtype == "object":
        Ng = len(A)
        AOA_flag = True
    else:
        Ng = 1
        AOA_flag = False

    # probability distribution over the hidden causes: i.e., Q(x)
    qx = spm_cross(x)
    G = 0
    qo = 0
    idx = np.array(np.where(qx > np.exp(-16))).T

    if AOA_flag:
        # accumulate expectation of entropy: i.e., E[lnP(o|x)]
        for i in idx:
            # probability over outcomes for this combination of causes
            po = np.ones(1)
            for g in range(Ng):
                index_vector = [slice(0, A[g].shape[0])] + list(i)
                po = spm_cross(po, A[g][tuple(index_vector)])

            po = po.ravel()
            qo += qx[tuple(i)] * po
            G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))
    else:
        for i in idx:
            po = np.ones(1)
            index_vector = [slice(0, A.shape[0])] + list(i)
            po = spm_cross(po, A[tuple(index_vector)])
            po = po.ravel()
            qo += qx[tuple(i)] * po
            G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))

    # subtract negative entropy of expectations: i.e., E[lnQ(o)]
    G = G - qo.dot(np.log(qo + np.exp(-16)))

    return G
