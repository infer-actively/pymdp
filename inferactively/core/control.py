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
from inferactively.core import softmax, spm_dot, spm_wnorm, spm_cross, spm_MDP_G, utils

def update_posterior_policies(
    qs,
    A,
    B,
    C,
    policies,
    use_utility=True,
    use_states_info_gain=True,
    use_param_info_gain=False,
    pA=None,
    pB=None,
    gamma=16.0,
    return_numpy=True,
):
    """ Updates the posterior beliefs about policies based on expected free energy prior

        @TODO: Needs to be amended for use with multi-step policies (where possible_policies is a list of np.arrays (n_step x n_factor), not just a list of tuples as it is now)

        Parameters
        ----------
        - `qs` [1D numpy array, array-of-arrays, or Categorical (either single- or multi-factor)]:
            Current marginal beliefs about hidden state factors
        - `A` [numpy ndarray, array-of-arrays (in case of multiple modalities), or Categorical (both single and multi-modality)]:
            Observation likelihood model (beliefs about the likelihood mapping entertained by the agent)
        - `B` [numpy ndarray, array-of-arrays (in case of multiple hidden state factors), or Categorical (both single and multi-factor)]:
            Transition likelihood model (beliefs about the likelihood mapping entertained by the agent)
        - `C` [numpy 1D-array, array-of-arrays (in case of multiple modalities), or Categorical (both single and multi-modality)]:
            Prior beliefs about outcomes (prior preferences)
        - `policies` [list of tuples]:
            A list of all the possible policies, each expressed as a tuple of indices, where a given index corresponds to an action on a particular hidden state factor
            e.g. policies[1][2] yields the index of the action under policy 1 that affects hidden state factor 2
        - `use_utility` [bool]:
            Whether to calculate utility term, i.e how much expected observation confer with prior expectations
        - `use_states_info_gain` [bool]:
            Whether to calculate state information gain
        - `use_param_info_gain` [bool]:
            Whether to calculate parameter information gain @NOTE requires pA or pB to be specified 
        - `pA` [numpy ndarray, array-of-arrays (in case of multiple modalities), or Dirichlet (both single and multi-modality)]:
            Prior dirichlet parameters for A. Defaults to none, in which case info gain w.r.t. Dirichlet parameters over A is skipped.
        - `pB` [numpy ndarray, array-of-arrays (in case of multiple hidden state factors), or Dirichlet (both single and multi-factor)]:
            Prior dirichlet parameters for B. Defaults to none, in which case info gain w.r.t. Dirichlet parameters over A is skipped.
        - `gamma` [float, defaults to 16.0]:
            Precision over policies, used as the inverse temperature parameter of a softmax transformation of the expected free energies of each policy
        - `return_numpy` [Boolean]:
            True/False flag to determine whether output of function is a numpy array or a Categorical
        
        Returns
        --------
        - `qp` [1D numpy array or Categorical]:
            Posterior beliefs about policies, defined here as a softmax function of the expected free energies of policies
        - `efe` - [1D numpy array or Categorical]:
            The expected free energies of policies

    """

    n_policies = len(policies)

    efe = np.zeros(n_policies)
    q_pi = np.zeros((n_policies, 1))

    for idx, policy in enumerate(policies):
        qs_pi = get_expected_states(qs, B, policy)
        qo_pi = get_expected_obs(qs_pi, A)

        if use_utility:
            efe[idx] += calc_expected_utility(qo_pi, C)

        if use_states_info_gain:
            efe[idx] += calc_states_info_gain(A, qs_pi)

        if use_param_info_gain:
            if pA is not None:
                efe[idx] += calc_pA_info_gain(pA, qo_pi, qs_pi)
            if pB is not None:
                efe[idx] += calc_pB_info_gain(pB, qs_pi, qs, policy)

    q_pi = softmax(efe * gamma)

    if return_numpy:
        q_pi = q_pi / q_pi.sum(axis=0)
    else:
        q_pi = utils.to_categorical(q_pi).normalize()

    return q_pi, efe


def get_expected_states(qs, B, policy, return_numpy=False):
    """
    Given a posterior density qs, a transition likelihood model B, and a policy, 
    get the state distribution expected under that policy's pursuit

    Parameters
    ----------
    - `qs` [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Current posterior beliefs about hidden states
    - `B` [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or Categorical (either single-factor of AoA)]:
        Transition likelihood mapping from states at t to states at t + 1, with different actions (per factor) stored along the lagging dimension
   - `policy` [np.arrays]:
        np.array of size (policy_len x n_factors) where each value corrresponds to a control state
    - `return_numpy` [Boolean]:
        True/False flag to determine whether output of function is a numpy array or a Categorical
    Returns
    -------
    - `qs_pi` [ list of np.arrays with len n_steps, where in case of multiple hidden state factors, each np.array in the list is a 1 x n_factors array-of-arrays, otherwise a list of 1D numpy arrays]:
        Expected states under the given policy - also known as the 'posterior predictive density'

    """

    n_steps = policy.shape[0]
    n_factors = policy.shape[1]

    qs = utils.to_numpy(qs, flatten=True)
    B = utils.to_numpy(B)

    if utils.is_arr_of_arr(B):

        # initialise beliefs over expected states
        qs_pi = []
        for t in range(n_steps):
            qs_pi_t = np.empty(n_factors,dtype=object)
            qs_pi.append(qs_pi_t)

        # initialise expected states after first action using current posterior (t = 0)
        for control_factor, control in enumerate(policy[0,:]):
            qs_pi[0][control_factor] = spm_dot(B[control_factor][:,:,control], qs[control_factor])
        
        # then loop over future timepoints
        if n_steps > 1:
            for t in range(1,n_steps):
                for control_factor, control in enumerate(policy[t,:]):
                    qs_pi[t][control_factor] = spm_dot(B[control_factor][:,:,control], qs_pi[t-1][control_factor])

    else:

        # initialise beliefs over expected states
        qs_pi = []

        # initialise expected states after first action using current posterior (t = 0)
        qs_pi.append(spm_dot(B[0][:,:,policy[0,0]], qs))
        
        # then loop over future timepoints
        if n_steps > 1:
            for t in range(1,n_steps):
                qs_pi.append(spm_dot(B[0][:,:,policy[t,0]], qs_pi[t-1]))
   
    if return_numpy:
        if len(qs_pi) == 1:
            return qs_pi[0]
        else:
            return qs_pi
    else:
        if len(qs_pi) == 1:
            return utils.to_categorical(qs_pi[0])
        else:
            for t in range(n_steps):
                qs_pi[t] = utils.to_categorical(qs_pi[t])
            return qs_pi

def get_expected_obs(Qs_pi, A, return_numpy=False):
    """
    Given a posterior predictive density Qs_pi and an observation likelihood model A,
    get the expected observations given the predictive posterior.

    @TODO: Needs to be amended for use with multi-step policies (where possible_policies is a list of np.arrays (nStep x nFactor), not just a list of tuples as it is now)
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


def calc_expected_utility(Qo_pi, C):
    """
    Given expected observations under a policy Qo_pi and a prior over observations C
    compute the expected utility of the policy.

    @TODO: Needs to be amended for use with multi-step policies (where possible_policies is a list of np.arrays (nStep x nFactor), not just a list of tuples as it is now)
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


def calc_states_info_gain(A, Qs_pi):
    """
    Given a likelihood mapping A and a posterior predictive density over states Qs_pi,
    compute the Bayesian surprise (about states) expected under that policy
    @TODO: Needs to be amended for use with multi-step policies (where possible_policies is a list of np.arrays (nStep x nFactor), not just a list of tuples as it is now)
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


def calc_pA_info_gain(pA, Qo_pi, Qs_pi):
    """
    Compute expected Dirichlet information gain about parameters pA under a policy
    Parameters
    @TODO: Needs to be amended for use with multi-step policies (where possible_policies is a list of np.arrays (nStep x nFactor), not just a list of tuples as it is now)
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


def calc_pB_info_gain(pB, Qs_next, Qs_previous, policy):
    """
    Compute expected Dirichlet information gain about parameters pB under a given policy
    Parameters
    @TODO: Needs to be amended for use with multi-step policies (where possible_policies is a list of np.arrays (nStep x nFactor), not just a list of tuples as it is now)
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


def construct_policies(n_states, n_control=None, policy_len=1, control_fac_idx=None):
    """Generate a set of policies

    Each policy is encoded as a numpy.ndarray of shape (n_steps, n_factors), where each value corresponds to
    the index of an action for a given time step and control factor. The variable `policies` that is returned
    is a list of each policy-specific numpy nd.array.

    @NOTE: If the list of control state dimensionalities (`n_control`) is not provided, 
    `n_control` defaults to being equal to n_states, except for the indices provided by control_fac_idx, where
    the value of n_control for the indicated factor is 1.

    Arguments:
    -------
    - `n_states`: list of dimensionalities of hidden state factors
    - `n_control`: list of dimensionalities of control state factors 
    - `policy_len`: temporal length ('horizon') of policies
    - `control_fac_idx`: list of indices of the hidden state factors that are controllable (i.e. those whose n_control[i] > 1)

    Returns:
    -------
    - `policies`: list of np.ndarrays, where each array within the list is a numpy.ndarray of shape (n_steps, n_factors).
                Each value in a policy array corresponds to the index of an action for a given timestep and control factor.
    - `n_control`: list of dimensionalities of actions along each hidden state factor (i.e. control state dimensionalities). 
                The dimensionality of control states whose index is not in control_fac_idx is set to 1.
                This is only returned when n_control is not provided as argument.
    """

    n_factors = len(n_states)
    
    if control_fac_idx is None:
        control_fac_idx = list(range(n_factors))

    return_n_control = False
    if n_control is None:

        return_n_control = True
        n_control = []
        for c_idx in range(n_factors):
            # see comment above
            if c_idx in control_fac_idx:
                n_control.append(n_states[c_idx])
            else:
                n_control.append(1)
        n_control = list(np.array(n_control).astype(int))

    x = n_control * policy_len
    policies = list(itertools.product(*[list(range(i)) for i in x]))

    if policy_len > 1:
        for pol_i in range(len(policies)):
            policies[pol_i] = np.array(policies[pol_i]).reshape(policy_len, n_factors)

    if return_n_control:
        return policies, n_control
    else:
        return policies


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
