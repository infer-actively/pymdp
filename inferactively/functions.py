#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Functions

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

import itertools
import numpy as np
from scipy import special
from inferactively.distributions import Categorical


def softmax(distrib, return_numpy = True):
    """ Computes the softmax function on a set of values
    """
    if isinstance(distrib, Categorical):
        if distrib.IS_AOA:
            output = Categorical(dims = [list(el.shape) for el in distrib])
            for i in range(len(distrib.values)):
                output[i] = softmax(distrib.values[i],return_numpy=True)
            return output
        else:
            distrib = np.copy(distrib.values)
    output = distrib - distrib.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output,axis=0)
    if return_numpy:
        return output
    else:
        return Categorical(values=output)


def generate_policies(n_actions, policy_len):
    """ Generate of possible combinations of N actions for policy length T

    Returns
    -------
    `policies` [list]
        A list of tuples, each specifying a list of actions [int]

    """

    x = [n_actions] * policy_len
    return list(itertools.product(*[list(range(i)) for i in x]))


def kl_divergence(q, p):
    """
    TODO: make this work for multi-dimensional arrays
    """
    if not isinstance(type(q), type(Categorical)) or not isinstance(
        type(p), type(Categorical)
    ):
        raise ValueError("`kl_divergence` function takes `Categorical` objects")
    q.remove_zeros()
    p.remove_zeros()
    q = np.copy(q.values)
    p = np.copy(p.values)
    kl = np.sum(q * np.log(q / p), axis=0)[0]
    return kl


def spm_dot(X, x, dims_to_omit=None):
    """ Dot product of a multidimensional array with `x`

    The dimensions in `dims_to_omit` will not be summed across during the dot product

    Parameters
    ----------
    `x` [1D numpy.ndarray] - either vector or array of arrays
        The alternative array to perform the dot product with
    `dims_to_omit` [list :: int] (optional)
        Which dimensions to omit
    """

    if x.dtype == object:
        dims = (np.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    else:
        if x.shape[0] != X.shape[1]:
            """
            Case when the first dimension of `x` is likely the same as the first dimension of `A`
            e.g. inverting the generative model using observations.
            Equivalent to something like self.values[np.where(x),:]
            when `x` is a discrete 'one-hot' observation vector
            """
            dims = np.array([0], dtype=int)
        else:
            """
            Case when `x` leading dimension matches the lagging dimension of `values`
            E.g. a more 'classical' dot product of a likelihood with hidden states
            """
            dims = np.array([1], dtype=int)
        x_new = np.empty(1, dtype=object)
        x_new[0] = x.squeeze()
        x = x_new

    if dims_to_omit is not None:
        if not isinstance(dims_to_omit, list):
            raise ValueError("dims_to_omit must be a `list`")
        dims = np.delete(dims, dims_to_omit)
        if len(x) == 1:
            x = np.empty([0], dtype=object)
        else:
            x = np.delete(x, dims_to_omit)

    Y = X
    for d in range(len(x)):
        s = np.ones(np.ndim(Y), dtype=int)
        s[dims[d]] = np.shape(x[d])[0]
        Y = Y * x[d].reshape(tuple(s))
        Y = np.sum(Y, axis=dims[d], keepdims=True)
    Y = np.squeeze(Y)

    # perform check to see if `y` is a number
    if np.prod(Y.shape) <= 1.0:
        Y = np.asscalar(Y)
        Y = np.array([Y]).astype("float64")

    return Y


def spm_cross(X, x=None, *args):
    """ Multi-dimensional outer product

    If no `x` argument is passed, the function returns the "auto-outer product" of self

    Otherwise, the function will recursively take the outer product of the initial entry
    of `x` with `self` until it has depleted the possible entries of `x` that it can outer-product

    Parameters
    ----------
    `x` [np.ndarray] || [Categorical] (optional)
        The values to perfrom the outer-product with
    `args` [np.ndarray] || [Categorical] (optional)
        Perform the outer product of the `args` with self
    
    Returns
    -------
    `y` [np.ndarray] || [Categorical]
        The result of the outer-product

    """

    if len(args) == 0 and x is None:
        if X.dtype == "object":
            Y = spm_cross(*list(X))

        elif np.issubdtype(X.dtype, np.number):
            Y = X

        return Y

    if X.dtype == "object":
        X = spm_cross(*list(X))

    if x is not None and x.dtype == "object":
        x = spm_cross(*list(x))

    reshape_dims = tuple(list(X.shape) + list(np.ones(x.ndim, dtype=int)))
    A = X.reshape(reshape_dims)

    reshape_dims = tuple(list(np.ones(X.ndim, dtype=int)) + list(x.shape))
    B = x.reshape(reshape_dims)

    Y = np.squeeze(A * B)

    for x in args:
        Y = spm_cross(Y, x)

    return Y

def spm_betaln(z):
    """
    Returns the log of the multivariate beta function of a vector.
    FORMAT y = spm_betaln(z)
     y = spm_betaln(z) computes the natural logarithm of the beta function
     for corresponding elements of the vector z. if concerned is a matrix,
     the logarithm is taken over the columns of the matrix z.
    """

    y     = np.sum(special.gammaln(z),axis=0) - special.gammaln(np.sum(z,axis=0))

    return y

def update_posterior(A, observation, prior, return_numpy = True, method = 'FPI', **kwargs):
    """ 
    Update marginal posterior qx using variational inference, with optional selection of a message-passing algorithm

    Parameters
    ----------
    'A' [numpy nd.array (matrix or tensor or array-of-arrays) or Categorical]:
        Observation likelihood of the generative model, mapping from hidden states to observations. 
        Used to invert generative model to obtain marginal likelihood over hidden states, given the observation
    'observation' [numpy 1D array, array of arrays (with 1D numpy array entries), int or tuple]:
        The observation (generated by the environment). If single modality, this can be a 1D array (one-hot vector representation) or an int (observation index)
        If multi-mopdality, this can be an aray of arrays (whose entries are 1D one-hot vectors) or a tuple (of observation indices)
    'prior' [numpy 1D array, array of arrays (with 1D numpy array entries), or Categorical]:
        Prior beliefs of the agent, to be integrated with the marginal likelihood to obtain posterior
    'return_numpy' [Boolean]:
        True/False flag to determine whether the posterior is returned as a numpy array or a Categorical
    'method' [str]:
        Algorithm used to perform the variational inference. 
        Options: 'FPI' - Fixed point iteration - http://www.cs.cmu.edu/~guestrin/Class/10708/recitations/r9/VI-view.pdf, slides 13- 18
                                               - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.137.221&rep=rep1&type=pdf, slides 24 - 38
                 'VMP  - Variational message passing
                 'MMP' - Marginal message passing
                 'BP'  - Belief propagation
                 'EP'  - Expectation propagation
                 'CV'  - CLuster variation method
    **kwargs: List of keyword/parameter arguments corresponding to parameter values for the respective variational inference algorithm

    Returns
    ----------
    'qx' [numpy 1D array, array of arrays (with 1D numpy array entries), or Categorical]:
        Marginal posterior beliefs over hidden states (single- or multi-factor) achieved via variational approximation.
    """

    if isinstance(A, Categorical):
        A = A.values

    if A.dtype == "object":
        Nf = A[0].ndim - 1
        Ns = list(A[0].shape[1:])
        Ng = len(A)
        No = []
        for g in range(Ng):
            No.append(A[g].shape[0])
    else:
        Nf = A.ndim - 1
        Ns = list(A.shape[1:])
        Ng = 1
        No = [A.shape[0]]
    
    if isinstance(observation, Categorical):
        observation = observation.values
        if Ng == 1:
            observation = observation.squeeze()
        else:
            for g in range(Ng):
                observation[g] = observation[g].squeeze()

    if isinstance(observation, int):
        observation = np.eye(No[0])[observation]
    
    if isinstance(observation, tuple):
        observation_AoA = np.empty(Ng, dtype = object)
        for g in range(Ng):
            observation_AoA[g] = np.eye(No[g])[observation[g]]
        
        observation = observation_AoA

    if isinstance(prior, Categorical):

        prior_new = np.empty(Nf, dtype = object)

        if prior.IS_AOA:
            for f in range(Nf):
                prior_new[f] = prior[f].values.squeeze()
        else:
            prior_new[0] = prior.values.squeeze()
        
        prior = prior_new
        
    elif prior.dtype != "object":

        prior_new = np.empty(Nf, dtype = object)
        prior_new[0] = prior
        prior = prior_new

    if method == 'FPI':
        qx = run_FPI(A, observation, prior, No, Ns, numIter=kwargs['numIter'], dF=kwargs['dF'], dF_tol=kwargs['dF_tol'])
    if method == 'VMP':
        qx = run_VMP(A, observation, prior, **kwargs)
    if method == 'MMP':
        qx = run_MMP(A, observation, prior, **kwargs)
    if method == 'BP':
        qx = run_MMP(A, observation, prior, **kwargs)
    if method == 'EP':
        qx = run_MMP(A, observation, prior, **kwargs)
    if method == 'CV':
        qx = run_MMP(A, observation, prior, **kwargs)

    if return_numpy:
        return qx
    else:
        return Categorical(values=qx)

def run_FPI(A, observation, prior, No, Ns, numIter=10, dF=1.0, dF_tol=0.001):
    """
    Update marginal posterior beliefs about hidden states
    using variational fixed point iteration (FPI)
    Parameters
    ----------
    'A' [numpy nd.array (matrix or tensor or array-of-arrays)]:
        Observation likelihood of the generative model, mapping from hidden states to observations. 
        Used to invert generative model to obtain marginal likelihood over hidden states, given the observation
    'observation' [numpy 1D array or array of arrays (with 1D numpy array entries)]:
        The observation (generated by the environment). If single modality, this can be a 1D array (one-hot vector representation).
        If multi-mopdality, this can be an aray of arrays (whose entries are 1D one-hot vectors).
    'prior' [numpy 1D array, array of arrays (with 1D numpy array entries)]:
        Prior beliefs of the agent, to be integrated with the marginal likelihood to obtain posterior
    'numIter' [int]:
        Number of variational fixed-point iterations to run.
    'dF' [float]:
        Starting free energy gradient (dF/dQx) before updating in the course of gradient descent.
    'dF_tol' [float]:
        Threshold value of the gradient of the variational free energy (dF/dQx), to be checked at each iteration. If 
        dF <= dF_tol, the iterations are halted pre-emptively and the final marginal posterior belief(s) is(are) returned

    Returns
    ----------
    'qx' [numpy 1D array or array of arrays (with 1D numpy array entries):
        Marginal posterior beliefs over hidden states (single- or multi-factor) achieved via variational fixed point iteration (mean-field)
    """

    # Code should be changed to this, once you've defined the gradient of the free energy:
    # dF = 1
    # while iterNum < numIter or dF > dF_tol:
    #       [DO ITERATIONS]
    # until then, use the following code:

    Ng = len(No)
    Nf = len(Ns)

    L = np.ones(tuple(Ns))

    # loop over observation modalities and use mean-field assumption to multiply 'induced posterior' onto
    # a single joint likelihood over hidden factors - of size Ns
    if Ng == 1:
        L *= spm_dot(A, observation)
    else:
        for g in range(Ng):
            L *= spm_dot( A[g], observation[g] )

    # initialize posterior to flat distribution
    qx = np.empty(Nf,dtype = object)
    for f in range(Nf):
        qx[f] = np.ones(Ns[f])/Ns[f]

    # in the trivial case of one hidden state factor, inference doesn't require FPI
    if Nf == 1:
        qL = spm_dot(L, qx, [0])
        qx[0] = softmax(np.log(qL + 1e-16) + np.log(prior[0] + 1e-16))
        return qx[0]

    else:
        iter_i = 0
        while iter_i < numIter:

            for f in range(Nf):

                # get the marginal for hidden state factor f by marginalizing out
                # other factors (summing them, weighted by their posterior expectation)
                qL = spm_dot(L, qx, [f]) 

                # this math is wrong, but anyway in theory we should add this in at
                # some point -- calculate the free energy and update the derivative
                # accordingly:
                # lnP = spm_dot(A_gm[g1,g2],O)
                # dF += np.sum(np.log(qL + 1e-16)-np.log(prior + 1e-16) + spm_dot(lnL, Qs, [f]))

                ####### QUESTION #######
                # Do we log-add the prior *within* each variational iteration? Or afterwards, when the iterations have ended?

                # 1. this is the version where we log-add the prior WITHIN each iteration, AND WITHIN each pass over hidden state factors
                # qx[f] = softmax(np.log(qL + 1e-16) + np.log(prior[f] + 1e-16)) 

                # 2. this is the version where don't add the prior (and it gets log-added at the end, outside both the factor and the iteration loop)
                qx[f] = softmax(np.log(qL + 1e-16)) 

            # 3. Log-add the prior WITHIN each iteration, but outside the 1st loop over factors
            # for f in range(Nf):
            #     qx[f] = softmax(np.log(qx[f] + 1e-16) + np.log(prior[f] + 1e-16))

            iter_i += 1

        for f in range(Nf):
            # (CONSISTENT WITH OPTION [2] ABOVE ^^) integrate the prior by log-adding and softmax-ing
            qx[f] = softmax(np.log(qx[f] + 1e-16) + np.log(prior[f] + 1e-16))

        return qx

def spm_MDP_G(A, x):
    """
    Calculates the Bayesian surprise in the same way as spm_MDP_G.m does in 
    the original matlab code.
    
    Arguments
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
    G  = G - qo.dot(np.log(qo + np.exp(-16)))

    return G
    



        