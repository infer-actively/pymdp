#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-an-iterable

""" Functions

__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import numpy as np
from scipy import special
from pymdp import utils
from itertools import chain

EPS_VAL = 1e-16 # global constant for use in spm_log() function

def spm_dot(X, x, dims_to_omit=None):
    """ Dot product of a multidimensional array with `x`. The dimensions in `dims_to_omit` 
    will not be summed across during the dot product
    
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

    # Construct dims to perform dot product on
    if utils.is_arr_of_arr(x):
        # dims = list((np.arange(0, len(x)) + X.ndim - len(x)).astype(int))
        dims = list(range(X.ndim - len(x),len(x)+X.ndim - len(x)))
        # dims = list(range(X.ndim))
    else:
        dims = [1]
        x = utils.to_arr_of_arr(x)

    if dims_to_omit is not None:
        arg_list = [X, list(range(X.ndim))] + list(chain(*([x[xdim_i],[dims[xdim_i]]] for xdim_i in range(len(x)) if xdim_i not in dims_to_omit))) + [dims_to_omit]
    else:
        arg_list = [X, list(range(X.ndim))] + list(chain(*([x[xdim_i],[dims[xdim_i]]] for xdim_i in range(len(x))))) + [[0]]

    Y = np.einsum(*arg_list)

    # check to see if `Y` is a scalar
    if np.prod(Y.shape) <= 1.0:
        Y = Y.item()
        Y = np.array([Y]).astype("float64")

    return Y


def spm_dot_classic(X, x, dims_to_omit=None):
    """ Dot product of a multidimensional array with `x`. The dimensions in `dims_to_omit` 
    will not be summed across during the dot product
    
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

    # Construct dims to perform dot product on
    if utils.is_arr_of_arr(x):
        dims = (np.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    else:
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
        # X = np.sum(X, axis=dims[d], keepdims=True)

    Y = np.sum(X, axis=tuple(dims.astype(int))).squeeze()
    # Y = np.squeeze(X)

    # check to see if `Y` is a scalar
    if np.prod(Y.shape) <= 1.0:
        Y = Y.item()
        Y = np.array([Y]).astype("float64")

    return Y

def spm_dot_old(X, x, dims_to_omit=None, obs_mode=False):
    """ Dot product of a multidimensional array with `x`. The dimensions in `dims_to_omit` 
    will not be summed across during the dot product

    #TODO: we should look for an alternative to obs_mode
    
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

    # Construct dims to perform dot product on
    if utils.is_arr_of_arr(x):
        dims = (np.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    else:
        if obs_mode is True:
            """
            @NOTE Case when you're getting the likelihood of an observation under 
                  the generative model. Equivalent to something like self.values[np.where(x),:]
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
        # X = np.sum(X, axis=dims[d], keepdims=True)

    Y = np.sum(X, axis=tuple(dims.astype(int))).squeeze()
    # Y = np.squeeze(X)

    # check to see if `Y` is a scalar
    if np.prod(Y.shape) <= 1.0:
        Y = Y.item()
        Y = np.array([Y]).astype("float64")

    return Y


def spm_cross(x, y=None, *args):
    """ Multi-dimensional outer product
    
    Parameters
    ----------
    - `x` [np.ndarray] || [Categorical] (optional)
        The values to perfrom the outer-product with. If empty, then the outer-product 
        is taken between x and itself. If y is not empty, then outer product is taken 
        between x and the various dimensions of y.
    - `args` [np.ndarray] || [Categorical] (optional)
        Remaining arrays to perform outer-product with. These extra arrays are recursively 
        multiplied with the 'initial' outer product (that between X and x).
    
    Returns
    -------
    - `z` [np.ndarray] || [Categorical]
          The result of the outer-product
    """

    if len(args) == 0 and y is None:
        if utils.is_arr_of_arr(x):
            z = spm_cross(*list(x))
        elif np.issubdtype(x.dtype, np.number):
            z = x
        else:
            raise ValueError(f"Invalid input to spm_cross ({x})")
        return z

    if utils.is_arr_of_arr(x):
        x = spm_cross(*list(x))

    if y is not None and utils.is_arr_of_arr(y):
        y = spm_cross(*list(y))

    reshape_dims = tuple(list(x.shape) + list(np.ones(y.ndim, dtype=int)))
    A = x.reshape(reshape_dims)

    reshape_dims = tuple(list(np.ones(x.ndim, dtype=int)) + list(y.shape))
    B = y.reshape(reshape_dims)
    z = np.squeeze(A * B)

    for x in args:
        z = spm_cross(z, x)
    return z

def dot_likelihood(A,obs):

    s = np.ones(np.ndim(A), dtype = int)
    s[0] = obs.shape[0]
    X = A * obs.reshape(tuple(s))
    X = np.sum(X, axis=0, keepdims=True)
    LL = np.squeeze(X)

    # check to see if `LL` is a scalar
    if np.prod(LL.shape) <= 1.0:
        LL = LL.item()
        LL = np.array([LL]).astype("float64")

    return LL


def get_joint_likelihood(A, obs, num_states):
    # deal with single modality case
    if type(num_states) is int:
        num_states = [num_states]
    A = utils.to_arr_of_arr(A)
    obs = utils.to_arr_of_arr(obs)
    ll = np.ones(tuple(num_states))
    for modality in range(len(A)):
        ll = ll * dot_likelihood(A[modality], obs[modality])
    return ll


def get_joint_likelihood_seq(A, obs, num_states):
    ll_seq = utils.obj_array(len(obs))
    for t, obs_t in enumerate(obs):
        ll_seq[t] = get_joint_likelihood(A, obs_t, num_states)
    return ll_seq


def spm_norm(A):
    """ 
    Returns normalization of Categorical distribution, 
    stored in the columns of A.
    """
    A = A + EPS_VAL
    normed_A = np.divide(A, A.sum(axis=0))
    return normed_A

def spm_log_single(arr):
    """
    Adds small epsilon value to an array before natural logging it
    """
    return np.log(arr + EPS_VAL)

def spm_log_obj_array(obj_arr):
    """
    Applies `spm_log_single` to multiple elements of a numpy object array
    """

    obj_arr_logged = utils.obj_array(len(obj_arr))
    for idx, arr in enumerate(obj_arr):
        obj_arr_logged[idx] = spm_log_single(arr)

    return obj_arr_logged

def spm_wnorm(A):
    """ 
    Returns Expectation of logarithm of Dirichlet parameters over a set of 
    Categorical distributions, stored in the columns of A.
    """
    A = A + EPS_VAL
    norm = np.divide(1.0, np.sum(A, axis=0))
    avg = np.divide(1.0, A)
    wA = norm - avg
    return wA


def spm_betaln(z):
    """ Log of the multivariate beta function of a vector.
     @NOTE this function computes across columns if `z` is a matrix
    """
    return np.sum(special.gammaln(z), axis=0) - special.gammaln(np.sum(z, axis=0))


def softmax(dist):
    """ 
    Computes the softmax function on a set of values
    """

    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output

def calc_free_energy(qs, prior, n_factors, likelihood=None):
    """ Calculate variational free energy
    @TODO Primarily used in FPI algorithm, needs to be made general
    """
    free_energy = 0
    for factor in range(n_factors):
        # Neg-entropy of posterior marginal H(q[f])
        negH_qs = qs[factor].dot(np.log(qs[factor][:, np.newaxis] + 1e-16))
        # Cross entropy of posterior marginal with prior marginal H(q[f],p[f])
        xH_qp = -qs[factor].dot(prior[factor][:, np.newaxis])
        free_energy += negH_qs + xH_qp

    if likelihood is not None:
        accuracy = spm_dot(likelihood, qs)[0]
        free_energy -= accuracy
    return free_energy


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
        array assigning likelihoods of observations/outcomes under the various 
        hidden state configurations
    
    x (numpy ndarray or array-object):
        Categorical distribution presenting probabilities of hidden states 
        (this can also be interpreted as the predictive density over hidden 
        states/causes if you're calculating the expected Bayesian surprise)
        
    Returns
    -------
    G (float):
        the (expected or not) Bayesian surprise under the density specified by x --
        namely, this scores how much an expected observation would update beliefs 
        about hidden states x, were it to be observed. 
    """

    num_modalities = len(A)

    # Probability distribution over the hidden causes: i.e., Q(x)
    qx = spm_cross(x)
    G = 0
    qo = 0
    idx = np.array(np.where(qx > np.exp(-16))).T

    if utils.is_arr_of_arr(A):
        # Accumulate expectation of entropy: i.e., E_{Q(o, s)}[lnP(o|x)]
        for i in idx:
            # Probability over outcomes for this combination of causes
            po = np.ones(1)
            for modality_idx, A_m in enumerate(A):
                index_vector = [slice(0, A_m.shape[0])] + list(i)
                po = spm_cross(po, A_m[tuple(index_vector)])

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
   
    # Subtract negative entropy of expectations: i.e., E_{Q(o)}[lnQ(o)]
    G = G - qo.dot(spm_log_single(qo))  # type: ignore

    return G


"""
 def calc_free_energy_policy(A, B, obs_t, qs_policy, policy, curr_t, t_horizon, T, previous_actions=None):

     Calculate variational free energy for a specific policy.

     Parameters
     ----------
     - 'A' [numpy nd.array (matrix or tensor or array-of-arrays)]:
         Observation likelihood of the generative model, mapping from hidden states to observations.
         Used in inference to get the likelihood of an observation, under different hidden state configurations.
     - 'B' [numpy.ndarray (tensor or array-of-arrays)]:
         Transition likelihood of the generative model, mapping from hidden states at t to hidden states at t+1.
         Used in inference to get expected future (or past) hidden states, given past (or future) hidden 
         states (or expectations thereof).
     - 'obs_t' [list of length t_horizon of numpy 1D array or array of arrays (with 1D numpy array entries)]:
         Sequence of observations sampled from beginning of time horizon until the current timestep t. The 
         first observation (the start of the time horizon)
         is either the first timestep of the generative process or the first timestep of the policy horizon 
         (whichever is closer to 'curr_t' in time).
         The observations over time are stored as a list of numpy arrays, where in case of multi-modalities 
         each numpy array is an array-of-arrays, with
         one 1D numpy.ndarray for each modality. In the case of a single modality, each observation is a 
         single 1D numpy.ndarray.
     - 'qs_policy' [list of length T of numpy 1D arrays or array of arrays (with 1D numpy array entries):
         Marginal posterior beliefs over hidden states (single- or multi-factor) under a given policy.
     - 'policy' [2D np.ndarray]:
         Array of actions constituting a single policy. Policy is a shape (n_steps, n_control_factors) 
         numpy.ndarray, the values of which
         indicate actions along a given control factor (column index) at a given timestep (row index).
     - 'curr_t' [int]:
         Current timestep (relative to the 'absolute' time of the generative process).
     - 't_horizon'[int]:
         Temporal horizon of inference for states and policies.
     - 'T' [int]:
         Temporal horizon of the generative process (absolute time)
     -'previous_actions' [numpy.ndarray with shape (num_steps, n_control_factors) or None]:
         Array of previous actions, which can be used to constrain the 'past' messages in inference to only 
         consider states of affairs that were possible
         under actions that are known to have been taken. The first dimension of previous-arrays 
         (previous_actions.shape[0]) encodes how far back in time
         the agent is considering. The first timestep of this either corresponds to either the first 
         timestep of the generative process or the f
         first timestep of the policy horizon (whichever is sooner in time).  (optional)

     Returns
     ----------
     -'F_pol' [float]:
         Total variational free energy of the policy under consideration.

     # extract dimensions of observation modalities and number of levels per modality. 
     # Do the same for hidden states
     if utils.is_arr_of_arr(obs_t[0]):
         n_observations = [ obs_t_m.shape[0] for obs_t_m in obs_t[0] ]
     else:
         n_observations = [obs_t[0].shape[0]]

     if utils.is_arr_of_arr(qs_policy[0][0]):
         n_states = [ qs_t_f.shape[0] for qs_t_f in qs_policy[0][0] ]
     else:
         n_states = [qs_policy[0][0].shape[0]]

     n_modalities = len(n_observations)
     n_factors = len(n_states)

     # compute time-window, taking into account boundary conditions - this range goes from start of 
     # time-window (in absolute time) to current absolute time index
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
             lnBpast_tensor[f] = B[f][:,:,full_policy[t-1,f].dot(qs_policy[t-1][f])

         F_pol += calc_free_energy(qs_policy[t], lnBpast_tensor, n_factors, likelihood[t])

     return F_pol
"""
