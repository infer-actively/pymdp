#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

""" Functions
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import itertools
import numpy as np
from scipy import special
from inferactively.distributions import Categorical


def softmax(distrib, return_numpy=True):
    """ Computes the softmax function on a set of values
    """
    if isinstance(distrib, Categorical):
        if distrib.IS_AOA:
            output = Categorical(dims=[list(el.shape) for el in distrib])
            for i in range(len(distrib.values)):
                output[i] = softmax(distrib.values[i], return_numpy=True)
            return output
        else:
            distrib = np.copy(distrib.values)
    output = distrib - distrib.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
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
        Y = Y.item()
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

    y = np.sum(special.gammaln(z), axis=0) - special.gammaln(np.sum(z, axis=0))

    return y


def update_posterior_states(A, observation, prior, return_numpy=True, method="FPI", **kwargs):
    """ 
    Update marginal posterior qx using variational inference, with optional selection of a message-passing algorithm
    Parameters
    ----------
    'A' [numpy nd.array (matrix or tensor or array-of-arrays) or Categorical]:
        Observation likelihood of the generative model, mapping from hidden states to observations. 
        Used to invert generative model to obtain marginal likelihood over hidden states, given the observation
    'observation' [numpy 1D array, array of arrays (with 1D numpy array entries), int or tuple]:
        The observation (generated by the environment). If single modality, this can be a 1D array 
        (one-hot vector representation) or an int (observation index)
        If multi-modality, this can be an array of arrays (whose entries are 1D one-hot vectors) or a tuple (of observation indices)
    'prior' [numpy 1D array, array of arrays (with 1D numpy array entries), or Categorical]:
        Prior beliefs of the agent, to be integrated with the marginal likelihood to obtain posterior
    'return_numpy' [Boolean]:
        True/False flag to determine whether the posterior is returned as a numpy array or a Categorical
    'method' [str]:
        Algorithm used to perform the variational inference. 
        Options: 'FPI' - Fixed point iteration 
                - http://www.cs.cmu.edu/~guestrin/Class/10708/recitations/r9/VI-view.pdf, slides 13- 18
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
        observation_AoA = np.empty(Ng, dtype=object)
        for g in range(Ng):
            observation_AoA[g] = np.eye(No[g])[observation[g]]

        observation = observation_AoA

    if isinstance(prior, Categorical):

        prior_new = np.empty(Nf, dtype=object)

        if prior.IS_AOA:
            for f in range(Nf):
                prior_new[f] = prior[f].values.squeeze()
        else:
            prior_new[0] = prior.values.squeeze()

        prior = prior_new

    elif prior.dtype != "object":

        prior_new = np.empty(Nf, dtype=object)
        prior_new[0] = prior
        prior = prior_new

    if method == "FPI":
        qx = run_FPI(A, observation, prior, No, Ns, **kwargs)
    if method == "VMP":
        raise NotImplementedError("VMP is not implemented")
    if method == "MMP":
        raise NotImplementedError("MMP is not implemented")
    if method == "BP":
        raise NotImplementedError("BP is not implemented")
    if method == "EP":
        raise NotImplementedError("EP is not implemented")
    if method == "CV":
        raise NotImplementedError("CV is not implemented")

    if return_numpy:
        return qx
    else:
        return Categorical(values=qx)


def run_FPI(A, observation, prior, No, Ns, num_iter=10, dF=1.0, dF_tol=0.001):
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
        If multi-modality, this can be an array of arrays (whose entries are 1D one-hot vectors).
    'prior' [numpy 1D array, array of arrays (with 1D numpy array entries)]:
        Prior beliefs of the agent, to be integrated with the marginal likelihood to obtain posterior
    'num_iter' [int]:
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
            L *= spm_dot(A[g], observation[g])

    # initialize posterior to flat distribution
    qx = np.empty(Nf, dtype=object)
    for f in range(Nf):
        qx[f] = np.ones(Ns[f]) / Ns[f]

    # in the trivial case of one hidden state factor, inference doesn't require FPI
    if Nf == 1:
        qL = spm_dot(L, qx, [0])
        qx[0] = softmax(np.log(qL + 1e-16) + np.log(prior[0] + 1e-16))
        return qx[0]

    else:
        iter_i = 0
        while iter_i < num_iter:

            for f in range(Nf):

                # get the marginal for hidden state factor f by marginalizing out
                # other factors (summing them, weighted by their posterior expectation)
                qL = spm_dot(L, qx, [f])

                # this math is wrong, but anyway in theory we should add this in at
                # some point -- calculate the free energy and update the derivative
                # accordingly:
                # lnP = spm_dot(A_gm[g1,g2],O)
                # dF += np.sum(np.log(qL + 1e-16)-np.log(prior + 1e-16) + spm_dot(lnL, Qs, [f]))

                # @TODO:
                # Do we log-add the prior *within* each variational iteration? Or afterwards, when the iterations have ended?
                
                # 1. This is the version where we log-add the prior WITHIN each iteration, AND WITHIN each pass over hidden state factors
                qx[f] = softmax(np.log(qL + 1e-16) + np.log(prior[f] + 1e-16))

                """
                2. This is the version where don't add the prior until the end
                (and it gets log-added at the end, outside both the factor and the iteration loop)
                qx[f] = softmax(np.log(qL + 1e-16))
                """
            """ 
            3. Log-add the prior WITHIN each iteration, but outside the 1st loop over factors
            > for f in range(Nf):
            >     qx[f] = softmax(np.log(qx[f] + 1e-16) + np.log(prior[f] + 1e-16))
            """

            iter_i += 1

        # for f in range(Nf):
        #     # CONSISTENT WITH OPTION [2] ABOVE: integrate the prior by log-adding and softmax-ing
        #     qx[f] = softmax(np.log(qx[f] + 1e-16) + np.log(prior[f] + 1e-16))

        return qx

def update_posterior_policies(Qs, A, pA, B, pB, C, possiblePolicies, gamma = 16.0, return_numpy=True):
    '''
    Parameters
    ----------
    Qs [1D numpy array, array-of-arrays, or Categorical (either single- or multi-factor)]:
        current marginal beliefs about hidden state factors
    A [numpy ndarray, array-of-arrays (in case of multiple modalities), or Categorical (both single and multi-modality)]:
        Observation likelihood model (beliefs about the likelihood mapping entertained by the agent)
    pA [numpy ndarray, array-of-arrays (in case of multiple modalities), or Dirichlet (both single and multi-modality)]:
        Prior dirichlet parameters for A
    B [numpy ndarray, array-of-arrays (in case of multiple hidden state factors), or Categorical (both single and multi-factor)]:
        Transition likelihood model (beliefs about the likelihood mapping entertained by the agent)
    pB [numpy ndarray, array-of-arrays (in case of multiple hidden state factors), or Dirichlet (both single and multi-factor)]:
        Prior dirichlet parameters for B
    C [numpy 1D-array, array-of-arrays (in case of multiple modalities), or Categorical (both single and multi-modality)]:
        Prior beliefs about outcomes (prior preferences)
    possiblePolicies [list of tuples]:
        a list of all the possible policies, each expressed as a tuple of indices, where a given index corresponds to an action on a particular hidden state factor
        e.g. possiblePolicies[1][2] yields the index of the action under Policy 1 that affects Hidden State Factor 2
    gamma [float]:
        precision over policies, used as the inverse temperature parameter of a softmax transformation of the expected free energies of each policy
    return_numpy [Boolean]:
        True/False flag to determine whether output of function is a numpy array or a Categorical
    
    Returns
    --------
    p_i [1D numpy array or Categorical]:
        posterior beliefs about policies, defined here as a softmax function of the expected free energies of policies
    EFE [1D numpy array or Categorical]:
        the expected free energies of policies
    '''

    if not isinstance(C,Categorical):
        C = Categorical(values = C)
    
    C = softmax(C.log())

    Np = len(possiblePolicies)

    EFE = Categorical(dims=Np)
    p_i = Categorical(dims=Np)

    for p_i, policy in enumerate(possiblePolicies):

        Qs_pi = get_expected_states(Qs, B, policy)

        Qo_pi = get_expected_obs(A, Qs_pi)

        utility = calculate_expected_utility(Qo_pi,C)
        EFE[p_i] += utility

        surprise_states = calculate_expected_surprise(A, Qs_pi)
        EFE[p_i] += surprise_states

        infogain_pA = calculate_infogain_pA(pA, Qo_pi, Qs_pi)
        EFE[p_i] += infogain_pA

        infogain_pB = calculate_infogain_pB(pA, Qs_pi, Qs, policy)
        EFE[p_i] += infogain_pB
    
    p_i = softmax(EFE * gamma)

    return p_i, EFE

def get_expected_states(Qs, B, policy, return_numpy = True):
    '''
    Given a posterior density Qs, a transition likelihood model B, and a policy, 
    get the state distribution expected under that policy's pursuit.

    Parameters
    ----------
    Qs [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Current posterior beliefs about hidden states
    B [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or Categorical (either single-factor of AoA)]:
        Transition likelihood mapping from states at t to states at t + 1, with different actions (per factor) stored along the lagging dimension
    policy [tuple of ints]:
        Tuple storing indices of actions along each hidden state factor. E.g. policy[1] gives the index of the action occurring on Hidden State Factor 1
    return_numpy [Boolean]:
        True/False flag to determine whether output of function is a numpy array or a Categorical
    Returns
    -------
    Qs_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Expected states under the given policy - also known as the 'posterior predictive density'
    '''

    if isinstance(B, Categorical):

        if B.IS_AOA:
            Qs_pi = Categorical( values = [B[f][:,:,a].dot(Qs[f], return_numpy=True) for f, a in enumerate(policy)])
        else:
            Qs_pi = B[:,:,policy[0]].dot(Qs)
    
    elif B.dtype == 'object':

        Nf = len(B)

        Qs_pi = np.empty(Nf, dtype = object)

        for f in range(Nf):
            Qs_pi[f] = spm_dot(B[f][:,:,policy[f]], Qs[f])
    
    else:

        Qs_pi = spm_dot(B[:,:,policy[0]], Qs)

        
            

    # if isinstance(B, Categorical):
    #     B = B.values

    # if B.dtype == "object":
    #     Ns = [B_i.shape[0] for B_i in B]
    #     Nf = len(Ns)
    # else:
    #     Ns = B.shape[0]
    #     Nf = 1

    # if isinstance(Qs, Categorical):

    #     Qs_new = np.empty(Nf, dtype=object)

    #     if Qs.IS_AOA:
    #         for f in range(Nf):
    #             Qs_new[f] = Qs[f].values.squeeze()
    #     else:
    #         Qs_new[0] = Qs.values.squeeze()

    #     Qs = Qs_new









    return

def get_expected_obs(A, Qs_pi):
    '''
    @TODO:
    '''
    return

def calculate_expected_utility(Qo_pi,C):
    '''
    @TODO:
    '''
    return

def calculate_expected_surprise(A, Qs_pi):
    '''
    @TODO:
    This function will spm_MDP_G within it to do the heavy lifting,
    but the wrapper is needed for arguments-handling etc. (e.g. in case that arguments are Categoricals, 
    need to pull out values/possible squeeze() them if they're column vectors)
    '''
    return

def calculate_infogain_pA(pA, Qo_pi, Qs_pi):
    '''
    '''
    return

def calculate_infogain_pB(pA, Qs_pi, Qs, policy):
    '''
    '''
    return

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

def cross_product_beta(dist_a, dist_b):
    """
    @TODO: needs to be replaced by spm_cross
    """
    if not isinstance(type(dist_a), type(Categorical)) or not isinstance(type(dist_b), type(Categorical)):
        raise ValueError(
            '[cross_product] function takes [Categorical] objects')
    values_a = np.copy(dist_a.values)
    values_b = np.copy(dist_b.values)
    a = np.reshape(values_a, (values_a.shape[0], values_a.shape[1], 1, 1))
    b = np.reshape(values_b, (1, 1, values_b.shape[0], values_b.shape[1]))
    values = np.squeeze(a * b)
    return values