#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-an-iterable

""" Functions
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import itertools
import numpy as np
from pymdp.maths import softmax, spm_dot, spm_wnorm, spm_MDP_G, spm_log_single, spm_log_obj_array
from pymdp import utils
import copy

def update_posterior_policies_mmp(
    qs_seq_pi,
    A,
    B,
    C,
    policies,
    use_utility=True,
    use_states_info_gain=True,
    use_param_info_gain=False,
    prior=None,
    pA=None,
    pB=None,
    F = None,
    E = None,
    gamma=16.0
):  
    """
    `qs_seq_pi` [numpy object array]:
                Posterior marginals beliefs over hidden states for each policy. 
                The data structure is nested as policies --> timesteps --> hidden state factors. 
                So qs_seq_pi[p_idx][t][f] is the belief about factor `f` at time `t`, under policy `p_idx`.
                @TODO: Clarify whether this can also be lists, or must it be object arrays?
    `A`: numpy object array that stores likelihood mappings for each modality.
    `B`: numpy object array that stores transition matrices (possibly action-conditioned) for each hidden state factor
    `policies`: numpy object array that stores each (potentially-multifactorial) policy in `policies[p_idx]`. Shape of `policies[p_idx]` is `(num_timesteps, num_factors)`
    `use_utility`: Boolean that determines whether expected utility should be incorporated into computation of EFE (default: `True`)
    `use_states_info_gain`: Boolean that determines whether state epistemic value (info gain about hidden states) should be incorporated into computation of EFE (default: `True`)
    `use_param_info_gain`: Boolean that determines whether parameter epistemic value (info gain about generative model parameters) should be incorporated into computation of EFE (default: `False`)
    `prior`: numpy object array that stores priors over hidden states - this matters when computing the first value of the parameter info gain for the Dirichlet parameters over B
    `pA`: numpy object array that stores Dirichlet priors over likelihood mappings (one per modality)
    `pB`: numpy object array that stores Dirichlet priors over transition mappings (one per hidden state factor)
    `F` : 1D numpy array that stores variational free energy of each policy 
    `E` : 1D numpy array that stores prior probability each policy (e.g. 'habits')
    `gamma`: Float that encodes the precision over policies
    """

    num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A, B)
    horizon = len(qs_seq_pi[0])
    num_policies = len(qs_seq_pi)

    qo_seq = utils.obj_array(horizon)
    for t in range(horizon):
        qo_seq[t] = utils.obj_array_zeros(num_obs)

    # initialise expected observations
    qo_seq_pi = utils.obj_array(num_policies)

    efe = np.zeros(num_policies)

    if F is None:
        F = np.zeros(num_policies)
    if E is None:
        E = np.zeros(num_policies)

    for p_idx, policy in enumerate(policies):

        qo_seq_pi[p_idx] = get_expected_obs(qs_seq_pi[p_idx], A)

        if use_utility:
            efe[p_idx] += calc_expected_utility(qo_seq_pi[p_idx], C)
        
        if use_states_info_gain:
            efe[p_idx] += calc_states_info_gain(A, qs_seq_pi[p_idx])
        
        if use_param_info_gain:
            if pA is not None:
                efe[p_idx] += calc_pA_info_gain(pA, qo_seq_pi[p_idx], qs_seq_pi[p_idx])
            if pB is not None:
                efe[p_idx] += calc_pB_info_gain(pB, qs_seq_pi[p_idx], prior, policy)

        
    q_pi = softmax(efe * gamma - F + E)
   
    return q_pi, efe


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
    gamma=16.0
):
    """ Updates the posterior beliefs about policies based on expected free energy prior

        Parameters
        ----------
        - `qs` [numpy object array]:
            Current marginal beliefs about (single or multiple) hidden state factors
        - `A` [numpy object array (both single and multi-modality)]:
            Observation likelihood model
        - `B` [numpy object array (both single and multi-factor)]:
                Transition likelihood model
        - `C` [numpy object array (both single and multi-modality)]:
            Prior beliefs about outcomes (prior preferences)
        - `policies` [list of tuples]:
            A list of all the possible policies, each expressed as a tuple of indices, where a given 
            index corresponds to an action on a particular hidden state factor e.g. policies[1][2] yields the 
            index of the action under policy 1 that affects hidden state factor 2
        - `use_utility` [bool]:
            Whether to calculate utility term, i.e how much expected observation confer with prior expectations
        - `use_states_info_gain` [bool]:
            Whether to calculate state information gain
        - `use_param_info_gain` [bool]:
            Whether to calculate parameter information gain @NOTE requires pA or pB to be specified 
        - `pA` [numpy ndarray, array-of-arrays (in case of multiple modalities), or Dirichlet 
                (both single and multi-modality)]:
            Prior dirichlet parameters for A. Defaults to none, in which case info gain w.r.t. Dirichlet 
            parameters over A is skipped.
        - `pB` [numpy ndarray, array-of-arrays (in case of multiple hidden state factors), or 
            Dirichlet (both single and multi-factor)]:
            Prior dirichlet parameters for B. Defaults to none, in which case info gain w.r.t. 
            Dirichlet parameters over A is skipped.
        - `gamma` [float, defaults to 16.0]:
            Precision over policies, used as the inverse temperature parameter of a softmax transformation 
            of the expected free energies of each policy
        Returns
        --------
        - `qp` [1D numpy array]:
            Posterior beliefs about policies, defined here as a softmax function of the 
            (gamma-weighted) expected free energies of policies
        - `efe` - [1D numpy array]:
            A vector containing the expected free energies of each policy

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

    return q_pi, efe

def get_expected_states(qs, B, policy):
    """
    Given a posterior density qs, a transition likelihood model B, and a policy, 
    get the state distribution expected under that policy's pursuit

    Parameters
    ----------
    - `qs` [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array)]:
        Current posterior beliefs about hidden states
    - `B` [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array)]:
        Transition likelihood mapping from states at t to states at t + 1, with different actions 
        (per factor) stored along the lagging dimension
   - `policy` [np.arrays]:
        np.array of size (policy_len x n_factors) where each value corrresponds to a control state
    Returns
    -------
    - `qs_pi` [ list of numpy object arrays where `len(qs_pi) == n_steps`]
        Expected states under the given policy - also referred to in the literature as the 'posterior predictive density'
    """
    n_steps = policy.shape[0]
    n_factors = policy.shape[1]

    # initialise posterior predictive density as a list of beliefs over time, including current posterior beliefs about hidden states as the first element
    qs_pi = [qs] + [utils.obj_array(n_factors) for t in range(n_steps)]
    
    # get expected states over time
    for t in range(n_steps):
        for control_factor, action in enumerate(policy[t,:]):
            qs_pi[t+1][control_factor] = B[control_factor][:,:,int(action)].dot(qs_pi[t][control_factor])

    return qs_pi[1:]
 

def get_expected_obs(qs_pi, A):
    """
    Given a posterior predictive density Qs_pi and an observation likelihood model A,
    get the expected observations given the predictive posterior.

    Parameters
    ----------
    qs_pi [numpy object array (where each entry is a numpy 1D array), or list of numpy object arrays]:
        Posterior predictive density over hidden states. If a list, each entry of the list is the 
        posterior predictive for a given timepoint of an expected trajectory
    A [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array)]:
        Observation likelihood mapping from hidden states to observations, with different modalities 
        (if there are multiple) stored in different arrays
    Returns
    -------
    qo_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or list]:
        Expected observations under the given policy. If a list, a list of the expected observations 
        over the time horizon of policy evaluation, where each entry is the expected observations at a given timestep. 
    """

    n_steps = len(qs_pi) # each element of the list is the PPD at a different timestep

    # initialise expected observations
    qo_pi = []

    for t in range(n_steps):
        qo_pi_t = utils.obj_array(len(A))
        qo_pi.append(qo_pi_t)

    # compute expected observations over time
    for t in range(n_steps):
        for modality, A_m in enumerate(A):
            qo_pi[t][modality] = spm_dot(A_m, qs_pi[t])

    return qo_pi

def calc_expected_utility(qo_pi, C):
    """
    Given expected observations under a policy Qo_pi and a prior over observations C
    compute the expected utility of the policy.

    Parameters
    ----------
    qo_pi [list of numpy object arrays (both single and multi-modality)]:
        Expected observations under the given policy (predictive posterior over outcomes), for each timestep of planning
        Each entry is the expected observations at a given timestep of the forward horizon. 
    C [numpy object array (both single and multi-modality)]:
        Prior beliefs over outcomes (e.g. preferences), encoded in terms of relative log probabilities. This is softmaxed to form
        a proper probability distribution before being used to compute the expected utility.
    Returns
    -------
    expected_util [scalar]:
        Utility (reward) expected under the policy in question
    """
    n_steps = len(qo_pi)
    
    # initialise expected utility
    expected_util = 0

    # loop over time points and modalities
    num_modalities = len(C)
    for t in range(n_steps):
        for modality in range(num_modalities):
            lnC = spm_log_single(softmax(C[modality][:, np.newaxis]))
            expected_util += qo_pi[t][modality].dot(lnC)

    return expected_util


def calc_states_info_gain(A, qs_pi):
    """
    Given a likelihood mapping A and a posterior predictive density over states Qs_pi,
    compute the Bayesian surprise (about states) expected under that policy
    Parameters
    ----------
    A [numpy object array (both single and multi-modality)]:
        Observation likelihood mapping from hidden states to observations, with 
        different modalities (if there are multiple) stored in different sub-arrays of the object array.
    qs_pi [list of [numpy object array (both single and multi-factor)]:
        Posterior predictive density over hidden states. Each entry of 
        the list is the posterior predictive density over hidden states for a given timepoint 
        of an expected trajectory.
    Returns
    -------
    states_surprise [scalar]:
        Bayesian surprise (about states) or salience expected under the policy in question
    """

    n_steps = len(qs_pi)

    states_surprise = 0
    for t in range(n_steps):
        states_surprise += spm_MDP_G(A, qs_pi[t])

    return states_surprise


def calc_pA_info_gain(pA, qo_pi, qs_pi):
    """
    Compute expected Dirichlet information gain about parameters pA under a policy
    Parameters
    ----------
    pA [numpy object array]:
        Prior dirichlet parameters parameterizing beliefs about the likelihood 
        mapping from hidden states to observations, with each modality-specific Dirichlet prior stored in different arrays.
    qo_pi [list of numpy object arrays]:
        Expected observations. Each element of the list is the posterior 
        predictive density over observations for a given timepoint of an expected trajectory
    qs_pi list of numpy object arrays]:
        Posterior predictive density over hidden states. Each element of the list 
        is the posterior predictive for a given timepoint of an expected trajectory
    Returns
    -------
    infogain_pA [scalar]:
        Surprise (about dirichlet parameters) expected under the policy in question
    """

    n_steps = len(qo_pi)
    
    num_modalities = len(pA)
    wA = utils.obj_array(num_modalities)
    for modality, pA_m in enumerate(pA):
        wA[modality] = spm_wnorm(pA[modality])

    pA_infogain = 0
    
    for modality in range(num_modalities):
        wA_modality = wA[modality] * (pA[modality] > 0).astype("float")
        for t in range(n_steps):
            pA_infogain -= qo_pi[t][modality].dot(spm_dot(wA_modality, qs_pi[t])[:, np.newaxis])

    return pA_infogain


def calc_pB_info_gain(pB, qs_pi, qs_prev, policy):
    """
    Compute expected Dirichlet information gain about parameters pB under a given policy
    Parameters
    ----------
    pB [numpy object array]:
        Prior dirichlet parameters parameterizing beliefs about the likelihood 
        describing transitions between hidden states, with each factor-specific Dirichlet prior stored in different arrays.
    qs_pi [list numpy object arrays]:
        Posterior predictive density over hidden states. Each element of the list 
        is the posterior predictive for a given timepoint of an expected trajectory.
    qs_prev [numpy object array]:
        Posterior over hidden states (before getting observations)
    policy [numpy 2D ndarray, of size n_steps x n_control_factors]:
        Policy to consider. Each row of the matrix encodes the action index 
        along a different control factor for a given timestep.  
    Returns
    -------
    infogain_pB [scalar]:
        Surprise (about dirichlet parameters) expected under the policy in question
    """

    n_steps = len(qs_pi)

    num_factors = len(pB)
    wB = utils.obj_array(num_factors)
    for factor, pB_f in enumerate(pB):
        wB[factor] = spm_wnorm(pB_f)

    pB_infogain = 0

    for t in range(n_steps):
        # the 'past posterior' used for the information gain about pB here is the posterior
        # over expected states at the timestep previous to the one under consideration
        # if we're on the first timestep, we just use the latest posterior in the
        # entire action-perception cycle as the previous posterior
        if t == 0:
            previous_qs = qs_prev
        # otherwise, we use the expected states for the timestep previous to the timestep under consideration
        else:
            previous_qs = qs_pi[t - 1]

        # get the list of action-indices for the current timestep
        policy_t = policy[t, :]
        for factor, a_i in enumerate(policy_t):
            wB_factor_t = wB[factor][:, :, int(a_i)] * (pB[factor][:, :, int(a_i)] > 0).astype("float")
            pB_infogain -= qs_pi[t][factor].dot(wB_factor_t.dot(previous_qs[factor]))

    return pB_infogain

def construct_policies(num_states, num_controls = None, policy_len=1, control_fac_idx=None):
    """Generate a set of policies

    Each policy is encoded as a numpy.ndarray of shape (n_steps, n_factors), where each 
    value corresponds to the index of an action for a given time step and control factor. The variable 
    `policies` that is returned is a list of each policy-specific numpy nd.array.
.
    Arguments:
    -------
    - `num_states`: list of dimensionalities of hidden state factors
    - `num_controls`: list of dimensionalities of control state factors. If `None`, then defaults to being the dimensionality of each hidden state factor that is controllable
    - `policy_len`: temporal length ('horizon') of policies
    - `control_fac_idx`: list of indices of the hidden state factors 
    that are controllable (i.e. those whose n_control[i] > 1)

    Returns:
    -------
    - `policies`: list of np.ndarrays, where each array within the list is a 
                    numpy.ndarray of shape (n_steps, n_factors).
                Each value in a policy array corresponds to the index of an action for 
                a given timestep and control factor.
    """

    num_factors = len(num_states)
    if control_fac_idx is None:
        if num_controls is not None:
            control_fac_idx = [f for f, n_c in enumerate(num_controls) if n_c > 1]
        else:
            control_fac_idx = list(range(num_factors))

    if num_controls is None:
        num_controls = [num_states[c_idx] if c_idx in control_fac_idx else 1 for c_idx in range(num_factors)]
        
    x = num_controls * policy_len
    policies = list(itertools.product(*[list(range(i)) for i in x]))
    for pol_i in range(len(policies)):
        policies[pol_i] = np.array(policies[pol_i]).reshape(policy_len, num_factors)

    return policies
    
def get_num_controls_from_policies(policies):
    """
    This calculates the list of dimensionalities of control factors
    from the policy array.
    @NOTE: 
    This assumes a policy space such that for each control factor, there is at least
    one policy that entails taking the action with the maximum index along that control factor.
    """

    return list(np.max(np.vstack(policies), axis = 0) + 1)
    

def sample_action(q_pi, policies, num_controls, action_selection="deterministic", alpha = 16.0):
    """
    Samples action from posterior over policies, using one of two methods. 
    Parameters
    ----------
    q_pi [1D numpy.ndarray]:
        Posterior beliefs about (possibly multi-step) policies.
    policies [list of numpy ndarrays]:
        List of arrays that indicate the policies under consideration. Each element 
        within the list is a matrix that stores the 
        the indices of the actions  upon the separate hidden state factors, at 
        each timestep (n_step x n_control_factor)
    num_controls [list of integers]:
        List of the dimensionalities of the different (controllable)) hidden state factors
    action_selection [string, `deterministic` or `stochastic`]:
        Indicates whether the sampled action for a given hidden state factor is given by 
        the evidence for that action, marginalized across different policies ('marginal_action')
        or simply the action entailed by a sample from the posterior over policies
    alpha [np.float64]:
        Action selection precision -- the inverse temperature of the softmax that is used to scale the 
        action marginals before sampling.
    Returns
    ----------
    selected_policy [1D numpy ndarray]:
        Numpy array containing the indices of the actions along each control factor
    """

    num_factors = len(num_controls)

    action_marginals = utils.obj_array_zeros(num_controls)
    
    # weight each action according to its integrated posterior probability over policies and timesteps
    for pol_idx, policy in enumerate(policies):
        for t in range(policy.shape[0]):
            for factor_i, action_i in enumerate(policy[t, :]):
                action_marginals[factor_i][action_i] += q_pi[pol_idx]
    
    selected_policy = np.zeros(num_factors)
    for factor_i in range(num_factors):

        # Either you do this:
        if action_selection == 'deterministic':
            selected_policy[factor_i] = np.argmax(action_marginals[factor_i])
        elif action_selection == 'stochastic':
            p_actions = softmax(action_marginals[factor_i] * alpha)
            selected_policy[factor_i] = utils.sample(p_actions)

    return selected_policy
