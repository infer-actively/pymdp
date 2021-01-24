#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import sys
import unittest

import numpy as np

from pymdp.distributions import Categorical, Dirichlet  # nopep8
from pymdp.core import control


def construct_generic_A(num_obs, n_states):
    """
    Generates a random likelihood array
    """
    num_modalities = len(num_obs)
    if num_modalities == 1:
        A = np.random.random_sample(tuple(num_obs + n_states))
        A = np.divide(A, A.sum(axis=0))
    else:
        A = np.empty(num_modalities, dtype=object)
        for modality, no in enumerate(num_obs):
            tmp = np.random.random_sample(tuple([no] + n_states))
            tmp = np.divide(tmp, tmp.sum(axis=0))
            A[modality] = tmp
    return A


def construct_pA(num_obs, n_states, prior_scale=1.0):
    """
    Generates Dirichlet prior over a observation likelihood distribution 
    (initialized to all ones * prior_scale parameter)
    """
    num_modalities = len(num_obs)
    if num_modalities == 1:
        pA = prior_scale * np.ones((num_obs + n_states))
    else:
        pA = np.empty(num_modalities, dtype=object)
        for modality, no in enumerate(num_obs):
            pA[modality] = prior_scale * np.ones((no, *n_states))
    return pA


def construct_generic_B(n_states, n_control):
    """
    Generates a fully controllable transition likelihood array, where each 
    action (control state) corresponds to a move to the n-th state from any 
    other state, for each control factor
    """

    num_factors = len(n_states)

    if num_factors == 1:
        B = np.eye(n_states[0])[:, :, np.newaxis]
        B = np.tile(B, (1, 1, n_control[0]))
        B = B.transpose(1, 2, 0)
    else:
        B = np.empty(num_factors, dtype=object)
        for factor, nc in enumerate(n_control):
            tmp = np.eye(nc)[:, :, np.newaxis]
            tmp = np.tile(tmp, (1, 1, nc))
            B[factor] = tmp.transpose(1, 2, 0)
    return B


def construct_pB(n_states, n_control, prior_scale=1.0):
    """
    Generates Dirichlet prior over a transition likelihood distribution 
    (initialized to all ones * prior_scale parameter)
    """
    num_factors = len(n_states)
    if num_factors == 1:
        pB = prior_scale * np.ones((n_states[0], n_states[0]))[:, :, np.newaxis]
        pB = np.tile(pB, (1, 1, n_control[0]))
        pB = pB.transpose(1, 2, 0)
    else:
        pB = np.empty(num_factors, dtype=object)
        for factor, nc in enumerate(n_control):
            tmp = prior_scale * np.ones((nc, nc))[:, :, np.newaxis]
            tmp = np.tile(tmp, (1, 1, nc))
            pB[factor] = tmp.transpose(1, 2, 0)
    return pB


def construct_generic_C(num_obs):
    """
    Generates a random C matrix
    """
    num_modalities = len(num_obs)
    if num_modalities == 1:
        C = np.random.rand(num_obs[0])
        C = np.divide(C, C.sum(axis=0))
    else:
        C = np.empty(num_modalities, dtype=object)
        for modality, no in enumerate(num_obs):
            tmp = np.random.rand(no)
            tmp = np.divide(tmp, tmp.sum())
            C[modality] = tmp
    return C


def construct_init_qs(n_states):
    """
    Creates a random initial posterior
    """
    num_factors = len(n_states)
    if num_factors == 1:
        qs = np.random.rand(n_states[0])
        qs = qs / qs.sum()
    else:
        qs = np.empty(num_factors, dtype=object)
        for factor, ns in enumerate(n_states):
            tmp = np.random.rand(ns)
            qs[factor] = tmp / tmp.sum()

    return qs


class TestControl(unittest.TestCase):
    def test_onestep_single_factor_posterior_policies(self):
        """
        Test for computing posterior over policies (and associated expected free energies)
        in the case of a posterior over hidden states with a single hidden state factor. 
        This version tests using a policy horizon of 1 step ahead
        """

        n_states = [3]
        n_control = [3]

        qs = Categorical(values=construct_init_qs(n_states))
        B = Categorical(values=construct_generic_B(n_states, n_control))
        pB = Dirichlet(values=construct_pB(n_states, n_control))

        # Single timestep
        n_step = 1
        policies = control.construct_policies(n_states, n_control, policy_len=n_step)

        # Single observation modality
        num_obs = [4]

        A = Categorical(values=construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values=construct_pA(num_obs, n_states))
        C = Categorical(values=construct_generic_C(num_obs))

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            pA=pA,
            pB=pB,
            gamma=16.0,
            return_numpy=True,
        )

        self.assertEqual(len(q_pi), len(policies)) # type: ignore
        self.assertEqual(len(efe), len(policies))

        # Multiple observation modalities
        num_obs = [3, 2]

        A = Categorical(values=construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values=construct_pA(num_obs, n_states))
        C = Categorical(values=construct_generic_C(num_obs))

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            pA=pA,
            pB=pB,
            gamma=16.0,
            return_numpy=True,
        )

        self.assertEqual(len(q_pi), len(policies))  # type: ignore
        self.assertEqual(len(efe), len(policies))

    def test_multistep_single_factor_posterior_policies(self):
        """
        Test for computing posterior over policies (and associated expected free energies)
        in the case of a posterior over hidden states with a single hidden state factor. 
        This version tests using a policy horizon of 3 steps ahead
        """

        n_states = [3]
        n_control = [3]

        qs = Categorical(values=construct_init_qs(n_states))
        B = Categorical(values=construct_generic_B(n_states, n_control))
        pB = Dirichlet(values=construct_pB(n_states, n_control))

        # Multiple timestep
        n_step = 3
        policies = control.construct_policies(n_states, n_control, policy_len=n_step)

        # Single observation modality
        num_obs = [4]

        A = Categorical(values=construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values=construct_pA(num_obs, n_states))
        C = Categorical(values=construct_generic_C(num_obs))

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            pA=pA,
            pB=pB,
            gamma=16.0,
            return_numpy=True,
        )

        self.assertEqual(len(q_pi), len(policies)) # type: ignore
        self.assertEqual(len(efe), len(policies))

        # Multiple observation modalities
        num_obs = [3, 2]

        A = Categorical(values=construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values=construct_pA(num_obs, n_states))
        C = Categorical(values=construct_generic_C(num_obs))

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            pA=pA,
            pB=pB,
            gamma=16.0,
            return_numpy=True,
        )

        self.assertEqual(len(q_pi), len(policies)) # type: ignore
        self.assertEqual(len(efe), len(policies))

    def test_onestep_multi_factor_posterior_policies(self):
        """
        Test for computing posterior over policies (and associated expected free energies)
        in the case of a posterior over hidden states with multiple hidden state factors. 
        This version tests using a policy horizon of 1 step ahead
        """
        n_states = [3, 4]
        n_control = [3, 4]

        qs = Categorical(values=construct_init_qs(n_states))
        B = Categorical(values=construct_generic_B(n_states, n_control))
        pB = Dirichlet(values=construct_pB(n_states, n_control))

        # Single timestep
        n_step = 1
        policies = control.construct_policies(n_states, n_control, policy_len=n_step)

        # Single observation modality
        num_obs = [4]

        A = Categorical(values=construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values=construct_pA(num_obs, n_states))
        C = Categorical(values=construct_generic_C(num_obs))

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            pA=pA,
            pB=pB,
            gamma=16.0,
            return_numpy=True,
        )

        self.assertEqual(len(q_pi), len(policies)) # type: ignore
        self.assertEqual(len(efe), len(policies))

        # multiple observation modalities
        num_obs = [3, 2]

        A = Categorical(values=construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values=construct_pA(num_obs, n_states))
        C = Categorical(values=construct_generic_C(num_obs))

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            pA=pA,
            pB=pB,
            gamma=16.0,
            return_numpy=True,
        )

        self.assertEqual(len(q_pi), len(policies)) # type: ignore
        self.assertEqual(len(efe), len(policies))

    def test_multistep_multi_factor_posterior_policies(self):
        """
        Test for computing posterior over policies (and associated expected free energies)
        in the case of a posterior over hidden states with multiple hidden state factors. 
        This version tests using a policy horizon of 3 steps ahead
        """
        n_states = [3, 4]
        n_control = [3, 4]

        qs = Categorical(values=construct_init_qs(n_states))
        B = Categorical(values=construct_generic_B(n_states, n_control))
        pB = Dirichlet(values=construct_pB(n_states, n_control))

        # Single timestep
        n_step = 3
        policies = control.construct_policies(n_states, n_control, policy_len=n_step)

        # Single observation modality
        num_obs = [4]

        A = Categorical(values=construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values=construct_pA(num_obs, n_states))
        C = Categorical(values=construct_generic_C(num_obs))

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            pA=pA,
            pB=pB,
            gamma=16.0,
            return_numpy=True,
        )

        self.assertEqual(len(q_pi), len(policies)) # type: ignore
        self.assertEqual(len(efe), len(policies))

        # Multiple observation modalities
        num_obs = [3, 2]

        A = Categorical(values=construct_generic_A(num_obs, n_states))
        pA = Dirichlet(values=construct_pA(num_obs, n_states))
        C = Categorical(values=construct_generic_C(num_obs))

        q_pi, efe = control.update_posterior_policies(
            qs,
            A,
            B,
            C,
            policies,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            pA=pA,
            pB=pB,
            gamma=16.0,
            return_numpy=True,
        )

        self.assertEqual(len(q_pi), len(policies)) # type: ignore
        self.assertEqual(len(efe), len(policies))

    def test_construct_policies_single_factor(self):
        """
        Test policy constructor function for single factor control states
        """
        n_states = [3]
        n_control = [3]
        control_fac_idx = [0]

        # One step policies
        policy_len = 1

        policies = control.construct_policies(n_states, n_control, policy_len, control_fac_idx)
        self.assertEqual(len(policies), n_control[0])
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len) # type: ignore

        # multistep step policies
        policy_len = 3

        policies = control.construct_policies(n_states, n_control, policy_len, control_fac_idx)
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len) # type: ignore

        # Now leave out the optional arguments of `construct_policies` such as `n_control` 
        # and `control_fac_idx`
        n_states = [3]

        # one step policies
        policy_len = 1

        policies, n_control = control.construct_policies(n_states, None, policy_len, None)
        self.assertEqual(len(policies), n_control[0])
        self.assertEqual(n_states[0], n_control[0])
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len) # type: ignore

        # multistep step policies
        policy_len = 3

        policies, n_control = control.construct_policies(n_states, None, policy_len, None)
        self.assertEqual(n_states[0], n_control[0])
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len) # type: ignore

    def test_construct_policies_multifactor(self):
        """
        Test policy constructor function for multi factor control states
        """
        n_states = [3, 4]
        n_control = [3, 1]
        control_fac_idx = [0]

        # One step policies
        policy_len = 1

        policies = control.construct_policies(n_states, n_control, policy_len, control_fac_idx)
        self.assertEqual(len(policies), n_control[0])
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len) # type: ignore

        # Multistep step policies
        policy_len = 3

        policies = control.construct_policies(n_states, n_control, policy_len, control_fac_idx)
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len) # type: ignore

        # One step policies
        policy_len = 1

        policies, n_control = control.construct_policies(n_states, None, policy_len, control_fac_idx)
        self.assertEqual(len(policies), n_control[0])
        self.assertEqual(n_control[1], 1)
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len) # type: ignore

        # multistep step policies
        policy_len = 3

        policies, n_control = control.construct_policies(n_states, None, policy_len, control_fac_idx)
        self.assertEqual(n_states[0], n_control[0])
        self.assertEqual(n_control[1], 1)
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len) # type: ignore

        control_fac_idx = [1]
        # One step policies
        policy_len = 1

        policies, n_control = control.construct_policies(n_states, None, policy_len, control_fac_idx)
        self.assertEqual(len(policies), n_control[1])
        self.assertEqual(n_control[0], 1)
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len) # type: ignore

        # multistep step policies
        policy_len = 3

        policies, n_control = control.construct_policies(n_states, None, policy_len, control_fac_idx)
        self.assertEqual(n_control[0], 1)
        for policy in policies:
            self.assertEqual(policy.shape[0], policy_len) # type: ignore

    def test_expected_utility(self):
        """
        Test for the expected utility function, for a simple single factor generative model 
        where there are imbalances in the preferences for different outcomes. Test for both single
        timestep policy horizons and multiple timestep horizons
        """
        n_states = [2]
        n_control = [2]

        qs = Categorical(values=construct_init_qs(n_states))
        B = Categorical(values=construct_generic_B(n_states, n_control))

        # Single timestep
        n_step = 1
        policies = control.construct_policies(n_states, n_control, policy_len=n_step)

        # Single observation modality
        num_obs = [2]

        # Create noiseless identity A matrix
        A = Categorical(values=np.eye(num_obs[0]))

        # Create imbalance in preferences for observations
        C = Categorical(values=np.eye(num_obs[0])[1])
        utilities = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states(qs, B, policy)
            qo_pi = control.get_expected_obs(qs_pi, A)
            utilities[idx] += control.calc_expected_utility(qo_pi, C)

        self.assertGreater(utilities[1], utilities[0])

        n_states = [3]
        n_control = [3]
        qs = Categorical(values=construct_init_qs(n_states))
        B = Categorical(values=construct_generic_B(n_states, n_control))

        # 3-step policies 
        # One involves going to state 0 two times in a row, and then state 2 at the end
        # One involves going to state 1 three times in a row

        policies = [np.array([0, 0, 2]).reshape(-1, 1), np.array([1, 1, 1]).reshape(-1, 1)]

        # single observation modality
        num_obs = [3]

        # create noiseless identity A matrix
        A = Categorical(values=np.eye(num_obs[0]))

        # create imbalance in preferences for observations
        # this is designed to illustrate the time-integrated nature of the expected free energy
        #  -- even though the first observation (index 0) is the most preferred, the policy
        # that frequents this observation the most is actually not optimal, because that policy
        # ends up visiting a less preferred state at the end.
        C = Categorical(values=np.array([1.2, 1, 0.5]))

        utilities = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states(qs, B, policy)
            qo_pi = control.get_expected_obs(qs_pi, A)
            utilities[idx] += control.calc_expected_utility(qo_pi, C)
        self.assertGreater(utilities[1], utilities[0])

    def test_state_info_gain(self):
        """
        Test the states_info_gain function. Demonstrates working
        by manipulating uncertainty in the likelihood matrices (A or B)
        in a ways that alternatively change the resolvability of uncertainty
        (via an imprecise expected state and a precise mapping, or high ambiguity
        and imprecise mapping).
        """
        n_states = [2]
        n_control = [2]

        qs = Categorical(values=np.eye(n_states[0])[0])
        # add some uncertainty into the consequences of the second policy, which
        # leads to increased epistemic value of observations, in case of pursuing
        # that policy -- in the case of a precise observation likelihood model
        B_matrix = construct_generic_B(n_states, n_control)
        B_matrix[:, :, 1] = control.softmax(B_matrix[:, :, 1])
        B = Categorical(values=B_matrix)

        # single timestep
        n_step = 1
        policies = control.construct_policies(n_states, n_control, policy_len=n_step)

        # single observation modality
        num_obs = [2]

        # create noiseless identity A matrix
        A = Categorical(values=np.eye(num_obs[0]))

        state_info_gains = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states(qs, B, policy)
            state_info_gains[idx] += control.calc_states_info_gain(A, qs_pi)
        self.assertGreater(state_info_gains[1], state_info_gains[0])

        # we can 'undo' the epistemic bonus of the second policy by making the A matrix
        # totally ambiguous, thus observations cannot resolve uncertainty about hidden states
        # - in this case, uncertainty in the posterior beliefs doesn't matter
        A = Categorical(values=np.ones((num_obs[0], num_obs[0])))
        A.normalize()

        state_info_gains = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states(qs, B, policy)
            state_info_gains[idx] += control.calc_states_info_gain(A, qs_pi)
        self.assertEqual(state_info_gains[0], state_info_gains[1])

    def test_pA_info_gain(self):
        """
        Test the pA_info_gain function. Demonstrates operation
        by manipulating shape of the Dirichlet priors over likelihood parameters
        (pA), which affects information gain for different expected observations
        """
        n_states = [2]
        n_control = [2]

        qs = Categorical(values=np.eye(n_states[0])[0])

        B = Categorical(values=construct_generic_B(n_states, n_control))

        # single timestep
        n_step = 1
        policies = control.construct_policies(n_states, n_control, policy_len=n_step)

        # single observation modality
        num_obs = [2]

        # create noiseless identity A matrix
        A = Categorical(values=np.eye(num_obs[0]))

        # create prior over dirichlets such that there is a skew
        # in the parameters about the likelihood mapping from the
        # second hidden state (index 1) to observations, such that one
        # observation is considered to be more likely than the other conditioned on that state.
        # Therefore sampling that observation would afford high info gain
        # about parameters for that part of the likelhood distribution.

        pA_matrix = construct_pA(num_obs, n_states)
        pA_matrix[0, 1] = 2.0
        pA = Dirichlet(values=pA_matrix)

        pA_info_gains = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states(qs, B, policy)
            qo_pi = control.get_expected_obs(qs_pi, A)
            pA_info_gains[idx] += control.calc_pA_info_gain(pA, qo_pi, qs_pi)
        self.assertGreater(pA_info_gains[1], pA_info_gains[0])

    def test_pB_info_gain(self):
        """
        Test the pB_info_gain function. Demonstrates operation
        by manipulating shape of the Dirichlet priors over likelihood parameters
        (pB), which affects information gain for different states
        """
        n_states = [2]
        n_control = [2]
        qs = Categorical(values=np.eye(n_states[0])[0])
        B = Categorical(values=construct_generic_B(n_states, n_control))
        pB_matrix = construct_pB(n_states, n_control)

        # create prior over dirichlets such that there is a skew
        # in the parameters about the likelihood mapping from the
        # hidden states to hidden states under the second action,
        # such that hidden state 0 is considered to be more likely than the other,
        # given the action in question
        # Therefore taking that action would yield an expected state that afford
        # high information gain about that part of the likelihood distribution.
        #
        pB_matrix[0, :, 1] = 2.0
        pB = Dirichlet(values=pB_matrix)

        # single timestep
        n_step = 1
        policies = control.construct_policies(n_states, n_control, policy_len=n_step)

        pB_info_gains = np.zeros(len(policies))
        for idx, policy in enumerate(policies):
            qs_pi = control.get_expected_states(qs, B, policy)
            pB_info_gains[idx] += control.calc_pB_info_gain(pB, qs_pi, qs, policy)
        self.assertGreater(pB_info_gains[1], pB_info_gains[0])

# class TestControl_v2(unittest.TestCase):
#     def test_update_posterior_policies(self):
#         """
#         Test for (the new-and-improved version of) computing posterior over policies (and associated expected free energies)
#         in the case of a posterior over hidden states with a single hidden state factor. 
#         This version tests using a policy horizon of 1 step ahead
#         """

#         # need to initialize some generic inputs for the new update_psoterior_policies, e.g.
#         # variables like `qs_seq_pi_future`

#         # add in __option__ for F and E
#         q_pi, efe = update_posterior_policies_v2(
#             qs_seq_pi_future,
#             A,
#             B,
#             C,
#             policies,
#             use_utility=True,
#             use_states_info_gain=True,
#             use_param_info_gain=False,
#             prior = None,
#             pA=None,
#             pB=None,
#             F = None,
#             E = None,
#             gamma=16.0,
#             return_numpy=True,
#         )

#         """
#         @NOTE: not sure these length checks are really rigorous enough - maybe we should have some pre-computed Q(pi)'s or EFE vectors,
#         either computed here 'manually' (outside the function) or even more rigorously, used a fixed random generative model benchmark from MATLAB
#         """
#         # self.assertEqual(len(q_pi), len(policies)) # type: ignore
#         # self.assertEqual(len(efe), len(policies))

    


if __name__ == "__main__":
    unittest.main()
