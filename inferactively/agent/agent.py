#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

from inferactively.distributions import Categorical, Dirichlet
import inferactively.core as core
import numpy as np


class Agent(object):
    def __init__(
        self,
        A=None,
        B=None,
        C=None,
        D=None,
        n_states=None,
        n_observations=None,
        n_controls=None,
        policy_len=1,
        control_fac_idx=None,
        possible_policies=None,
        gamma=16.0,
        sampling_type="marginal_action",
        message_passing_algo="FPI",
        message_passing_params=None,
    ):

        if A is not None:
            if not isinstance(A, Categorical):
                self.A = Categorical(values=A)
            else:
                self.A = A

            if self.A.IS_AOA:
                self.n_modalities = self.A.shape[0]
                self.n_observations = [self.A[g].shape[0] for g in range(self.n_modalities)]
            else:
                self.n_modalities = 1
                self.n_observations = self.A.shape[0]
            construct_A_flag = False
        else:
            if n_observations is None:
                raise ValueError(
                    "Must provide either `A` or `n_observations` to `Agent` constructor"
                )
            self.n_observations = n_observations
            self.n_modalities = len(self.n_observations)
            construct_A_flag = True

        if B is not None:
            if not isinstance(B, Categorical):
                self.B = Categorical(values=B)
            else:
                self.B = B

            if self.B.IS_AOA:
                self.n_factors = self.B.shape[0]
                self.n_states = [self.B[f].shape[0] for f in range(self.n_factors)]
            else:
                self.n_factors = 1
                self.n_states = [self.B.shape[0]]
            construct_B_flag = False
        else:
            if n_states is None:
                raise ValueError("Must provide either `B` or `n_states` to `Agent` constructor")
            self.n_states = n_states
            self.n_factors = len(self.n_factors)
            construct_B_flag = True

        if control_fac_idx is None:
            self.control_fac_idx = list(range(self.n_factors))
        else:
            self.control_fac_idx = control_fac_idx

        self.policy_len = policy_len

        if n_controls is None:
            self.n_controls, _ = self._construct_n_controls()
        else:
            self.n_controls = n_controls

        if possible_policies is None:
            _, self.possible_policies = self._construct_n_controls()
        else:
            self.possible_policies = possible_policies

        if C is not None:
            if isinstance(C, Categorical):
                self.C = C
            else:
                self.C = Categorical(values=C)
        else:
            self.C = self._construct_C_prior()

        if D is not None:
            if isinstance(D, Categorical):
                self.D = D
            else:
                self.D = Categorical(values=D)
        else:
            self.D = self._construct_D_prior()

        if construct_A_flag:
            self.A = self._construct_A_distribution()
        if construct_B_flag:
            self.B = self._construct_B_distribution()

        self.message_passing_algo = message_passing_algo
        if message_passing_params is None:
            self.message_passing_params = self._get_default_params()
        else:
            self.message_passing_params = message_passing_params

        self.gamma = gamma
        self.sampling_type = sampling_type

        self.qx = self.D
        self.action = None

        self.pA = None
        self.pB = None

    def _construct_A_distribution(self):
        if self.n_modalities == 1:
            A = Categorical(values=np.random.rand(*(self.n_observations + self.n_states)))
        else:
            A = np.empty(self.n_modalities, dtype=object)
            for g, No in enumerate(self.n_observations):
                A[g] = np.random.rand(*([No] + self.n_states))

            A = Categorical(values=A)
        A.normalize()
        return A

    def _construct_B_distribution(self):
        if self.n_factors == 1:
            B = np.eye(*self.n_states)[:, :, np.newaxis]
            if 0 in self.control_fac_idx:
                B = np.tile(B, (1, 1, self.n_controls[0]))
                B = B.transpose(1, 2, 0)
        else:
            B = np.empty(self.n_factors, dtype=object)

            for f, Ns in enumerate(self.n_states):
                B_basic = np.eye(Ns)[:, :, np.newaxis]
                if f in self.control_fac_idx:
                    B[f] = np.tile(B_basic, (1, 1, self.n_controls[f]))
                    B[f] = B[f].transpose(1, 2, 0)
                else:
                    B[f] = B_basic

        B = Categorical(values=B)
        B.normalize()
        return B

    def _construct_C_prior(self):
        if self.n_modalities == 1:
            C = np.zeros(*self.n_observations)
        else:
            C = np.array([np.zeros(No) for No in self.n_observations])

        return C

    def _construct_D_prior(self):
        if self.n_factors == 1:
            D = Categorical(values=np.ones(*self.n_states))
        else:
            D = Categorical(values=np.array([np.ones(Ns) for Ns in self.n_states]))
        D.normalize()

        return D

    def _construct_n_controls(self):
        n_controls, possible_policies = core.construct_policies(
            self.n_states, self.n_factors, self.control_fac_idx, self.policy_len
        )

        return n_controls, possible_policies

    def reset(self, init_qx=None):
        if init_qx is None:
            self.qx = self._construct_D_prior()
        else:
            if isinstance(init_qx, Categorical):
                self.qx = init_qx
            else:
                self.qx = Categorical(values=init_qx)

        return self.qx

    def infer_states(self, observation):
        if not hasattr(self, "qx"):
            self.reset()

        if self.message_passing_algo is "FPI":
            if self.action is not None:
                empirical_prior = core.get_expected_states(self.qx, self.B.log(), self.action)
            else:
                empirical_prior = self.D.log()
        else:
            if self.action is not None:
                empirical_prior = core.get_expected_states(self.qx, self.B.log(), self.action)
            else:
                empirical_prior = self.D

        qx = core.update_posterior_states(
            self.A,
            observation,
            empirical_prior,
            return_numpy=False,
            method=self.message_passing_algo,
            **self.message_passing_params
        )
        self.qx = qx

        return qx

    def infer_policies(self):
        q_pi, efe = core.update_posterior_policies(
            self.qx,
            self.A,
            self.B,
            self.C,
            self.possible_policies,
            self.pA,
            self.pB,
            self.gamma,
            return_numpy=False,
        )
        self.efe = efe
        self.q_pi = q_pi
        return q_pi, efe

    def sample_action(self):
        action = core.sample_action(
            self.q_pi, self.possible_policies, self.n_controls, self.sampling_type
        )

        self.action = action
        return action

    def _get_default_params(self):

        method = self.message_passing_algo

        if method == "FPI":
            default_params = {"dF": 1.0, "dF_tol": 0.001}
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

        return default_params
