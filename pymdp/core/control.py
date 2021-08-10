#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-an-iterable

""" Functions
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import itertools
import numpy as np
from pymdp.distributions import Categorical, Dirichlet
from pymdp.core.maths import softmax, spm_dot, spm_wnorm, spm_MDP_G, spm_log_single, spm_log_obj_array
from pymdp.core import utils
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
    gamma=16.0,
    return_numpy=True,
):  
    """
    `qs_seq_pi`: numpy object array that stores posterior marginals beliefs over hidden states for each policy. 
                The structure is nested as policies --> timesteps --> hidden state factors. So qs_seq_pi[p_idx][t][f] is the belief about factor `f` at time `t`, under policy `p_idx`
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
    `return_numpy`: Boolean that determines whether output should be a numpy array or an instance of the Categorical class (default: `True`)
    """

    A = utils.to_numpy(A)
    B = utils.to_numpy(B)
    num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A, B)
    horizon = len(qs_seq_pi[0])
    num_policies = len(qs_seq_pi)

    # initialise`qo_seq` as object arrays to initially populate `qo_seq_pi`
    qo_seq = utils.obj_array(horizon)
    for t in range(horizon):
        qo_seq[t] = utils.obj_array_zeros(num_obs)

    # initialise expected observations
    qo_seq_pi = utils.obj_array(num_policies)
    for p_idx in range(num_policies):
        # qo_seq_pi[p_idx] = copy.deepcopy(obs_over_time)
        qo_seq_pi[p_idx] = qo_seq

    efe = np.zeros(num_policies)

    if F is None:
        F = np.zeros(num_policies)
    if E is None:
        E = np.zeros(num_policies)

    for p_idx, policy in enumerate(policies):

        qs_seq_pi_i = qs_seq_pi[p_idx]

        for t in range(horizon):

            qo_pi_t = get_expected_obs(qs_seq_pi_i[t], A)
            qo_seq_pi[p_idx][t] = qo_pi_t

            if use_utility:
                efe[p_idx] += calc_expected_utility(qo_seq_pi[p_idx][t], C)

            if use_states_info_gain:
                efe[p_idx] += calc_states_info_gain(A, qs_seq_pi_i[t])

            if use_param_info_gain:
                if pA is not None:
                    efe[p_idx] += calc_pA_info_gain(pA, qo_seq_pi[p_idx][t], qs_seq_pi_i[t])
                if pB is not None:
                    if t > 0:
                        efe[p_idx] += calc_pB_info_gain(pB, qs_seq_pi_i[t], qs_seq_pi_i[t-1], policy)
                    else:
                        if prior is not None:
                            efe[p_idx] += calc_pB_info_gain(pB, qs_seq_pi_i[t], prior, policy)


    q_pi = softmax(efe * gamma - F + E)
    if not return_numpy:
        q_pi = utils.to_categorical(q_pi)
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
            qs_pi[t+1][control_factor] = B[control_factor][:,:,action].dot(qs_pi[t][control_factor])

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

    # wrap the predictive posterior density into a list, if it's not already in that form
    if not isinstance(qs_pi, list):
        qs_pi = [qs_pi]
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
    pA [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or 
    Dirichlet (either single-factor of AoA)]:
        Prior dirichlet parameters parameterizing beliefs about the likelihood 
        mapping from hidden states to observations, 
        with different modalities (if there are multiple) stored in different arrays.
    qo_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array),
     Categorical (either single-factor or AoA), or list]:
        Expected observations. If a list, each entry of the list is the posterior 
        predictive for a given timepoint of an expected trajectory
    qs_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), 
    Categorical (either single-factor or AoA), or list]:
        Posterior predictive density over hidden states. If a list, each entry of 
        the list is the posterior predictive for a given timepoint of an expected trajectory
    Returns
    -------
    infogain_pA [scalar]:
        Surprise (about dirichlet parameters) expected under the policy in question
    """

    if isinstance(qo_pi, list):
        n_steps = len(qo_pi)
        for t in range(n_steps):
            qo_pi[t] = utils.to_numpy(qo_pi[t], flatten=True)
    else:
        n_steps = 1
        qo_pi = [utils.to_numpy(qo_pi, flatten=True)]

    if isinstance(qs_pi, list):
        for t in range(n_steps):
            qs_pi[t] = utils.to_numpy(qs_pi[t], flatten=True)
    else:
        n_steps = 1
        qs_pi = [utils.to_numpy(qs_pi, flatten=True)]

    if isinstance(pA, Dirichlet):
        if pA.IS_AOA:
            num_modalities = pA.n_arrays
        else:
            num_modalities = 1
        wA = pA.expectation_of_log()
    else:
        if utils.is_arr_of_arr(pA):
            num_modalities = len(pA)
            wA = np.empty(num_modalities, dtype=object)
            for modality in range(num_modalities):
                wA[modality] = spm_wnorm(pA[modality])
        else:
            num_modalities = 1
            wA = spm_wnorm(pA)

    pA = utils.to_numpy(pA)
    pA_infogain = 0
    if num_modalities == 1:
        wA = wA * (pA > 0).astype("float")
        for t in range(n_steps):
            pA_infogain = -qo_pi[t].dot(spm_dot(wA, qs_pi[t])[:, np.newaxis])
    else:
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
    pB [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), 
    or Dirichlet (either single-factor of AoA)]:
        Prior dirichlet parameters parameterizing beliefs about the likelihood 
        describing transitions bewteen hidden states,
        with different factors (if there are multiple) stored in different arrays.
    qs_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), 
    Categorical (either single-factor or AoA), or list]:
        Posterior predictive density over hidden states. If a list, each entry of 
        the list is the posterior predictive for a given timepoint of an expected trajectory
    qs_prev [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), 
    or Categorical (either single-factor or AoA)]:
        Posterior over hidden states (before getting observations)
    policy [numpy 2D ndarray, of size n_steps x n_control_factors]:
        Policy to consider. Each row of the matrix encodes the action index 
        along a different control factor for a given timestep.  
    Returns
    -------
    infogain_pB [scalar]:
        Surprise (about dirichlet parameters) expected under the policy in question
    """

    if isinstance(qs_pi, list):
        n_steps = len(qs_pi)
        for t in range(n_steps):
            qs_pi[t] = utils.to_numpy(qs_pi[t], flatten=True)
    else:
        n_steps = 1
        qs_pi = [utils.to_numpy(qs_pi, flatten=True)]

    if isinstance(qs_prev, Categorical):
        qs_prev = utils.to_numpy(qs_prev, flatten=True)

    if isinstance(pB, Dirichlet):
        if pB.IS_AOA:
            num_factors = pB.n_arrays
        else:
            num_factors = 1
        wB = pB.expectation_of_log()
    else:
        if utils.is_arr_of_arr(pB):
            num_factors = len(pB)
            wB = np.empty(num_factors, dtype=object)
            for factor in range(num_factors):
                wB[factor] = spm_wnorm(pB[factor])
        else:
            num_factors = 1
            wB = spm_wnorm(pB)

    pB = utils.to_numpy(pB)
    pB_infogain = 0
    if num_factors == 1:
        for t in range(n_steps):
            if t == 0:
                previous_qs = qs_prev
            else:
                previous_qs = qs_pi[t - 1]
            a_i = policy[t, 0]
            wB_t = wB[:, :, a_i] * (pB[:, :, a_i] > 0).astype("float")
            pB_infogain = -qs_pi[t].dot(wB_t.dot(qs_prev))
    else:

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
                wB_factor_t = wB[factor][:, :, a_i] * (pB[factor][:, :, a_i] > 0).astype("float")
                pB_infogain -= qs_pi[t][factor].dot(wB_factor_t.dot(previous_qs[factor]))

    return pB_infogain

n_states = [5]
control_fac_idx = [0]

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
    This assumes a fully enumerated policy space (there is a policy that explores the maximum control state level for each control factor
    """

    return list(np.max(np.vstack(policies), axis = 0) + 1)
    

def sample_action(q_pi, policies, n_control, sampling_type="marginal_action"):
    """
    Samples action from posterior over policies, using one of two methods. 
    Parameters
    ----------
    q_pi [1D numpy.ndarray or Categorical]:
        Posterior beliefs about (possibly multi-step) policies.
    policies [list of numpy ndarrays]:
        List of arrays that indicate the policies under consideration. Each element 
        within the list is a matrix that stores the 
        the indices of the actions  upon the separate hidden state factors, at 
        each timestep (n_step x n_control_factor)
    n_control [list of integers]:
        List of the dimensionalities of the different (controllable)) hidden state factors
    sampling_type [string, 'marginal_action' or 'posterior_sample']:
        Indicates whether the sampled action for a given hidden state factor is given by 
        the evidence for that action, marginalized across different policies ('marginal_action')
        or simply the action entailed by a sample from the posterior over policies
    Returns
    ----------
    selected_policy [1D numpy ndarray]:
        Numpy array containing the indices of the actions along each control factor
    """

    n_factors = len(n_control)

    if sampling_type == "marginal_action":

        if utils.is_distribution(q_pi):
            q_pi = utils.to_numpy(q_pi)

        action_marginals = np.empty(n_factors, dtype=object)
        for c_idx in range(n_factors):
            action_marginals[c_idx] = np.zeros(n_control[c_idx])

        # weight each action according to its integrated posterior probability over policies and timesteps
        for pol_idx, policy in enumerate(policies):
            for t in range(policy.shape[0]):
                for factor_i, action_i in enumerate(policy[t, :]):
                    action_marginals[factor_i][action_i] += q_pi[pol_idx]
        
        # print(action_marginals[0])
        selected_policy = np.zeros(n_factors)
        for factor_i in range(n_factors):
            selected_policy[factor_i] = np.argmax(action_marginals[factor_i])

        # action_marginals = Categorical(values=action_marginals)
        # action_marginals.normalize()
        # selected_policy = np.array(action_marginals.sample())

    elif sampling_type == "posterior_sample":
        if utils.is_distribution(q_pi):
            policy_index = q_pi.sample()
            selected_policy = policies[policy_index]
        else:
            q_pi = Categorical(values=q_pi)
            policy_index = q_pi.sample()
            selected_policy = policies[policy_index]
    else:
        raise ValueError(f"{sampling_type} not supported")

    return selected_policy
