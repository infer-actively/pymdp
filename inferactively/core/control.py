#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-an-iterable

""" Functions
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import itertools
import numpy as np
import torch
from scipy import special
from inferactively.distributions import Categorical, Dirichlet
from inferactively.core import softmax, spm_dot, spm_wnorm, spm_cross

def construct_policies(Ns, Nf, control_fac_idx, policy_len):
    """Generate list of possible combinations of Ns[f_i] actions for Nf hidden state factors,
    where Nu[i] gives the number of actions available along hidden state factor f_i. Assumes that for each controllable hidden
    state factor, the number of possible actions == Ns[f_i]
    Arguments:
    -------
    Ns: list of dimensionalities of hidden state factors
    Nf: number of hidden state factors total
    control_fac_idx: indices of the hidden state factors that are controllable (i.e. those whose Nu[i] > 1)
    policy_len: length of each policy
    Returns:
    -------
    Nu: list of dimensionalities of actions along each hidden state factor
    possible_policies: list of arrays, where each array within the list corresponds to a policy, and each row
                        within a given policy (array) corresponds to a list of actions for each the several hidden state factor
                        for a given timestep (policy_len x Nf)
    """

    Nu = []

    for f_i in range(Nf):
        if f_i in control_fac_idx:
            Nu.append(Ns[f_i])
        else:
            Nu.append(1)

    x = Nu * policy_len

    possible_policies = list(itertools.product(*[list(range(i)) for i in x]))

    if policy_len > 1:
        for pol_i in range(len(possible_policies)):
            possible_policies[pol_i] = np.array(possible_policies[pol_i]).reshape(policy_len, Nf)

    Nu = list(np.array(Nu).astype(int))
    return Nu, possible_policies


def update_posterior_policies(
    Qs, A, B, C, possible_policies, pA=None, pB=None, gamma=16.0, return_numpy=True
):
    """ Updates the posterior beliefs about policies using the expected free energy approach
     (where belief in a policy is proportional to the free energy expected under its pursuit)
    @TODO: Needs to be amended for use with multi-step policies (where possiblePolicies is a list of np.arrays, not a list of tuples)
    Parameters
    ----------
    Qs [1D numpy array, array-of-arrays, or Categorical (either single- or multi-factor)]:
        current marginal beliefs about hidden state factors
    A [numpy ndarray, array-of-arrays (in case of multiple modalities), or Categorical (both single and multi-modality)]:
        Observation likelihood model (beliefs about the likelihood mapping entertained by the agent)
    B [numpy ndarray, array-of-arrays (in case of multiple hidden state factors), or Categorical (both single and multi-factor)]:
        Transition likelihood model (beliefs about the likelihood mapping entertained by the agent)
    C [numpy 1D-array, array-of-arrays (in case of multiple modalities), or Categorical (both single and multi-modality)]:
        Prior beliefs about outcomes (prior preferences)
    possible_policies [list of tuples]:
        a list of all the possible policies, each expressed as a tuple of indices, where a given index corresponds to an action on a particular hidden state factor
        e.g. possiblePolicies[1][2] yields the index of the action under Policy 1 that affects Hidden State Factor 2
    pA [numpy ndarray, array-of-arrays (in case of multiple modalities), or Dirichlet (both single and multi-modality)]:
        Prior dirichlet parameters for A. Defaults to none, in which case info gain w.r.t. Dirichlet parameters over A is skipped.
    pB [numpy ndarray, array-of-arrays (in case of multiple hidden state factors), or Dirichlet (both single and multi-factor)]:
        Prior dirichlet parameters for B. Defaults to none, in which case info gain w.r.t. Dirichlet parameters over A is skipped.
    gamma [float, defaults to 16.0]:
        precision over policies, used as the inverse temperature parameter of a softmax transformation of the expected free energies of each policy
    return_numpy [Boolean]:
        True/False flag to determine whether output of function is a numpy array or a Categorical
    
    Returns
    --------
    p_i [1D numpy array or Categorical]:
        posterior beliefs about policies, defined here as a softmax function of the expected free energies of policies
    EFE [1D numpy array or Categorical]:
        the expected free energies of policies
    """

    Np = len(possible_policies)

    EFE = np.zeros(Np)
    p_i = np.zeros((Np, 1))

    for p_i, policy in enumerate(possible_policies):
        Qs_pi = get_expected_states(Qs, B, policy)
        Qo_pi = get_expected_obs(Qs_pi, A)

        utility = calculate_expected_utility(Qo_pi, C)
        EFE[p_i] += utility

        surprise_states = calculate_expected_surprise(A, Qs_pi)
        EFE[p_i] += surprise_states
        if pA is not None:
            infogain_pA = calculate_infogain_pA(pA, Qo_pi, Qs_pi)
            EFE[p_i] += infogain_pA
        if pB is not None:
            infogain_pB = calculate_infogain_pB(pB, Qs_pi, Qs, policy)
            EFE[p_i] += infogain_pB

    p_i = softmax(EFE * gamma)

    if return_numpy:
        p_i = p_i / p_i.sum(axis=0)
    else:
        p_i = Categorical(values=p_i)
        p_i.normalize()

    return p_i, EFE


def get_expected_states(Qs, B, policy, return_numpy=False):
    """
    Given a posterior density Qs, a transition likelihood model B, and a policy, 
    get the state distribution expected under that policy's pursuit.

    @TODO: Needs to be amended for use with multi-step policies (where possiblePolicies is a list of np.arrays, not a list of tuples)
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
    """

    if isinstance(B, Categorical):
        if B.IS_AOA:
            Qs_pi = Categorical(
                values=np.array(
                    [
                        B[f][:, :, a].dot(Qs[f], return_numpy=True)[:, np.newaxis]
                        for f, a in enumerate(policy)
                    ],
                    dtype=object,
                )
            )
        else:
            Qs_pi = B[:, :, policy[0]].dot(Qs)

        if return_numpy and Qs_pi.IS_AOA:
            Qs_pi_flattened = np.empty(len(Qs_pi.values), dtype=object)
            for f in range(len(Qs_pi.values)):
                Qs_pi_flattened[f] = Qs_pi[f].values.flatten()
            return Qs_pi_flattened
        elif return_numpy and not Qs_pi.IS_AOA:
            return Qs_pi.values.flatten()
        else:
            return Qs_pi

    elif B.dtype == "object":
        Nf = len(B)
        Qs_pi = np.empty(Nf, dtype=object)

        if isinstance(Qs, Categorical):
            Qs = Qs.values
            for f in range(Nf):
                Qs[f] = Qs[f].flatten()
        for f in range(Nf):
            Qs_pi[f] = spm_dot(B[f][:, :, policy[f]], Qs[f])

    else:
        if isinstance(Qs, Categorical):
            Qs = Qs.values.flatten()
        Qs_pi = spm_dot(B[:, :, policy[0]], Qs)

    if not return_numpy:
        Qs_pi = Categorical(values=Qs_pi)
        return Qs_pi
    else:
        return Qs_pi


def get_expected_obs(Qs_pi, A, return_numpy=False):
    """
    Given a posterior predictive density Qs_pi and an observation likelihood model A,
    get the expected observations given the predictive posterior.

    @TODO: Needs to be amended for use with multi-step policies (where possible_policies is a list of np.arrays, not a list of tuples)
    Parameters
    ----------
    Qs_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Posterior predictive density over hidden states
    A [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or Categorical (either single-factor of AoA)]:
        Observation likelihood mapping from hidden states to observations, with different modalities (if there are multiple) stored in different arrays
    return_numpy [Boolean]:
        True/False flag to determine whether output of function is a numpy array or a Categorical
    Returns
    -------
    Qo_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Expected observations under the given policy 
    """
    if isinstance(A, Categorical):

        if not return_numpy:
            Qo_pi = A.dot(Qs_pi)
            return Qo_pi
        else:
            Qo_pi = A.dot(Qs_pi, return_numpy=True)
            if Qo_pi.dtype == "object":
                Qo_pi_flattened = np.empty(len(Qo_pi), dtype=object)
                for g in range(len(Qo_pi)):
                    Qo_pi_flattened[g] = Qo_pi[g].flatten()
                return Qo_pi_flattened
            else:
                return Qo_pi.flatten()

    elif A.dtype == "object":
        Ng = len(A)
        Qo_pi = np.empty(Ng, dtype=object)
        if isinstance(Qs_pi, Categorical):
            Qs_pi = Qs_pi.values
            for f in range(len(Qs_pi)):
                Qs_pi[f] = Qs_pi[f].flatten()
        for g in range(Ng):
            Qo_pi[g] = spm_dot(A[g], Qs_pi)

    else:
        if isinstance(Qs_pi, Categorical):
            Qs_pi = Qs_pi.values
        Qo_pi = spm_dot(A, Qs_pi)

    if not return_numpy:
        Qo_pi = Categorical(values=Qo_pi)
        return Qo_pi
    else:
        return Qo_pi


def calculate_expected_utility(Qo_pi, C):
    """
    Given expected observations under a policy Qo_pi and a prior over observations C
    compute the expected utility of the policy.

    @TODO: Needs to be amended for use with multi-step policies (where possible_policies is a list of np.arrays, not a list of tuples)
    Parameters
    ----------
    Qo_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Posterior predictive density over outcomes
    C [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array):
        Prior beliefs over outcomes, expressed in terms of relative log probabilities
    Returns
    -------
    expected_util [scalar]:
        Utility (reward) expected under the policy in question
    """

    if isinstance(Qo_pi, Categorical):
        Qo_pi = Qo_pi.values

    if Qo_pi.dtype == "object":
        for g in range(len(Qo_pi)):
            Qo_pi[g] = Qo_pi[g].flatten()

    if C.dtype == "object":

        expected_util = 0

        Ng = len(C)
        for g in range(Ng):
            lnC = np.log(softmax(C[g][:, np.newaxis]) + 1e-16)
            expected_util += Qo_pi[g].flatten().dot(lnC)

    else:

        lnC = np.log(softmax(C[:, np.newaxis]) + 1e-16)
        expected_util = Qo_pi.flatten().dot(lnC)

    return expected_util


def calculate_expected_surprise(A, Qs_pi):
    """
    Given a likelihood mapping A and a posterior predictive density over states Qs_pi,
    compute the Bayesian surprise (about states) expected under that policy

    @TODO: Needs to be amended for use with multi-step policies (where possiblePolicies is a list of np.arrays, not a list of tuples)
    Parameters
    ----------
    A [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or Categorical (either single-factor of AoA)]:
        Observation likelihood mapping from hidden states to observations, with different modalities (if there are multiple) stored in different arrays
    Qs_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Posterior predictive density over hidden states
    Returns
    -------
    states_surprise [scalar]:
        Surprise (about states) expected under the policy in question
    """

    if isinstance(A, Categorical):
        A = A.values

    if isinstance(Qs_pi, Categorical):
        Qs_pi = Qs_pi.values

    if Qs_pi.dtype == "object":
        for f in range(len(Qs_pi)):
            Qs_pi[f] = Qs_pi[f].flatten()
    else:
        Qs_pi = Qs_pi.flatten()

    states_surprise = spm_MDP_G(A, Qs_pi)

    return states_surprise


def calculate_infogain_pA(pA, Qo_pi, Qs_pi):
    """
    Compute expected Dirichlet information gain about parameters pA under a policy
    Parameters
    ----------
    pA [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or Dirichlet (either single-factor of AoA)]:
        Prior dirichlet parameters parameterizing beliefs about the likelihood mapping from hidden states to observations, 
        with different modalities (if there are multiple) stored in different arrays.
    Qo_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Posterior predictive density over observations, given hidden states expected under a policy
    Qs_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Posterior predictive density over hidden states
    Returns
    -------
    infogain_pA [scalar]:
        Surprise (about dirichlet parameters) expected under the policy in question
    """

    if isinstance(pA, Dirichlet):
        if pA.IS_AOA:
            Ng = pA.shape[0]
        else:
            Ng = 1
        wA = pA.expectation_of_log(return_numpy=True)
        pA = pA.values
    elif pA.dtype == "object":
        Ng = len(pA)
        wA = np.empty(Ng, dtype=object)
        for g in range(Ng):
            wA[g] = spm_wnorm(pA[g])
    else:
        Ng = 1
        wA = spm_wnorm(pA)

    if isinstance(Qo_pi, Categorical):
        Qo_pi = Qo_pi.values

    if Qo_pi.dtype == "object":
        for g in range(len(Qo_pi)):
            Qo_pi[g] = Qo_pi[g].flatten()
    else:
        Qo_pi = Qo_pi.flatten()

    if isinstance(Qs_pi, Categorical):
        Qs_pi = Qs_pi.values

    if Qs_pi.dtype == "object":
        for f in range(len(Qs_pi)):
            Qs_pi[f] = Qs_pi[f].flatten()
    else:
        Qs_pi = Qs_pi.flatten()

    if Ng > 1:
        infogain_pA = 0
        for g in range(Ng):
            wA_g = wA[g] * (pA[g] > 0).astype("float")
            infogain_pA -= Qo_pi[g].dot(spm_dot(wA_g, Qs_pi)[:, np.newaxis])

    else:
        wA = wA * (pA > 0).astype("float")
        infogain_pA = -Qo_pi.dot(spm_dot(wA, Qs_pi)[:, np.newaxis])

    return infogain_pA


def calculate_infogain_pB(pB, Qs_next, Qs_previous, policy):
    """
    Compute expected Dirichlet information gain about parameters pB under a given policy
    Parameters
    ----------
    pB [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or Dirichlet (either single-factor of AoA)]:
        Prior dirichlet parameters parameterizing beliefs about the likelihood describing transitions bewteen hidden states,
        with different factors (if there are multiple) stored in different arrays.
    Qs_next [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Posterior predictive density over hidden states under some policy
    Qs_previous [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Posterior predictive density over hidden states (prior to observations)
    Returns
    -------
    infogain_pB [scalar]:
        Surprise (about dirichlet parameters) expected under the policy in question
    """
    if isinstance(pB, Dirichlet):
        if pB.IS_AOA:
            Nf = pB.shape[0]
        else:
            Nf = 1
        wB = pB.expectation_of_log(return_numpy=True)
        pB = pB.values
    elif pB.dtype == "object":
        Nf = len(pB)
        wB = np.empty(Nf, dtype=object)
        for f in range(Nf):
            wB[f] = spm_wnorm(pB[f])
    else:
        Nf = 1
        wB = spm_wnorm(pB)

    if isinstance(Qs_next, Categorical):
        Qs_next = Qs_next.values

    if Qs_next.dtype == "object":
        for f in range(Nf):
            Qs_next[f] = Qs_next[f].flatten()
    else:
        Qs_next = Qs_next.flatten()

    if isinstance(Qs_previous, Categorical):
        Qs_previous = Qs_previous.values

    if Qs_previous.dtype == "object":
        for f in range(Nf):
            Qs_previous[f] = Qs_previous[f].flatten()
    else:
        Qs_previous = Qs_previous.flatten()

    if Nf > 1:
        infogain_pB = 0
        for f_i, a_i in enumerate(policy):
            wB_action = wB[f_i][:, :, a_i] * (pB[f_i][:, :, a_i] > 0).astype("float")
            infogain_pB -= Qs_next[f_i].dot(wB_action.dot(Qs_previous[f_i]))

    else:

        a_i = policy[0]

        wB = wB[:, :, a_i] * (pB[:, :, a_i] > 0).astype("float")
        infogain_pB = -Qs_next.dot(wB.dot(Qs_previous))

    return infogain_pB


def sample_action(p_i, possible_policies, Nu, sampling_type="marginal_action"):
    """
    Samples action from posterior over policies, using one of two methods. 
    @TODO: Needs to be amended for use with multi-step policies (where possible_policies is a list of np.arrays (nStep x nFactor), not just a list of tuples as it is now)
    Parameters
    ----------
    p_i [1D numpy.ndarray or Categorical]:
        Variational posterior over policies.
    possible_policies [list of tuples]:
        List of tuples that indicate the possible policies under consideration. Each tuple stores the actions taken upon the separate hidden state factors. 
        Same length as p_i.
    Nu [list of integers]:
        List of the dimensionalities of the different (controllable)) hidden states
    sampling_type [string, 'marginal_action' or 'posterior_sample']:
        Indicates whether the sampled action for a given hidden state factor is given by the evidence for that action, marginalized across different policies ('marginal_action')
        or simply the action entailed by the policy sampled from the posterior. 
    Returns
    ----------
    selectedPolicy [tuple]:
        tuple containing the list of actions selected by the agent
    """

    numControls = len(Nu)

    if sampling_type == "marginal_action":

        if isinstance(p_i, Categorical):
            p_i = p_i.values.squeeze()

        action_marginals = np.empty(numControls, dtype=object)
        for nu_i in range(numControls):
            action_marginals[nu_i] = np.zeros(Nu[nu_i])

        # Weight each action according to the posterior probability it gets across policies
        for pol_i, policy in enumerate(possible_policies):
            for nu_i, a_i in enumerate(policy):
                action_marginals[nu_i][a_i] += p_i[pol_i]

        action_marginals = Categorical(values=action_marginals)
        action_marginals.normalize()
        selected_policy = action_marginals.sample()

    elif sampling_type == "posterior_sample":
        if isinstance(p_i, Categorical):
            policy_index = p_i.sample()
            selected_policy = possible_policies[policy_index]
        else:
            sample_onehot = np.random.multinomial(1, p_i.squeeze())
            policy_index = np.where(sample_onehot == 1)[0][0]
            selected_policy = possible_policies[policy_index]

    return selected_policy


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
