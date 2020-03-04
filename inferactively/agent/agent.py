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
        n_modalities=None,
        n_factors=None,
        n_controls=None,
        policy_len=1,
        control_fac_idx=None,
        possible_policies=None,
        gamma=16.0,
        sampling_type="marginal_action",
        message_passing_algo="FPI",
        message_passing_params=None,
    ):

        self.policy_len = policy_len
        self.control_fac_idx = control_fac_idx
        self.possible_policies = possible_policies

        self.n_states = n_states
        self.n_factors = n_factors
        self.n_observations = n_observations
        self.n_modalities = n_modalities
        self.n_controls = n_controls

        if self.n_modalities is not None and self.n_observations is not None:
            if self.n_modalities != len(self.n_observations):
                raise ValueError("n_modalities must equal len(n_observations)")
        if self.n_modalities is None and self.n_observations is not None:
            self.n_modalities = len(self.n_observations)

        if self.n_factors is not None and self.n_states is not None:
            if self.n_factors != len(self.n_states):
                raise ValueError("n_factors must equal len(n_states)")
        if self.n_factors is None and self.n_states is not None:
            self.n_factors = len(self.n_states)

        if self.control_fac_idx is None and self.n_factors is not None:
            self.control_fac_idx = list(range(self.n_factors))
        if max(self.control_fac_idx) > (self.n_factors - 1):
            raise ValueError(
                "Indices of controllable factors greater than number of hidden state factors"
            )

        if self.n_controls is None:
            self.n_controls, _ = self._construct_n_controls()
        if len(self.n_controls) != self.n_factors:
            raise ValueError("n_controls must equal n_factors")

        if self.possible_policies is None:
            _, self.possible_policies = self._construct_n_controls()

        self.A = A
        if self.A is not None:
            if self.A is not None and self.A is not isinstance(self.A, Categorical):
                self.A = Categorical(values=self.A)
            if self.A.IS_AOA:
                self.n_modalities = self.A.shape[0]
                self.n_observations = [self.A[g].shape[0] for g in range(self.n_modalities)]
            else:
                self.n_modalities = 1
                self.n_observations = self.A.shape[0]
        else:
            self.A = self._construct_A_likelihood()

        self.B = B
        if self.B is not None:
            if self.B is not None and self.B is not isinstance(self.B, Categorical):
                self.B = Categorical(values=self.B)
            if self.B.IS_AOA:
                self.n_factors = self.B.shape[0]
                self.n_states = [self.B[f].shape[0] for f in range(self.n_factors)]
            else:
                self.n_factors = 1
                self.n_states = [self.B.shape[0]]

            if self.control_fac_idx is None:
                self.control_fac_idx = list(range(self.n_factors))
            self.n_controls, self.possible_policies = self._construct_n_controls()
        else:
            self.B = self._construct_B_likelihood()
            

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

    def _construct_n_controls(self):

        n_controls, possible_policies = core.construct_policies(
            self.n_states, self.n_factors, self.control_fac_idx, self.policy_len
        )

        return n_controls, possible_policies

    def _construct_A_likelihood(self):

        if self.n_modalities == 1:
            A = Categorical(values=np.random.rand(*(self.n_observations + self.n_states)))
        else:
            A = np.empty(self.n_modalities, dtype=object)
            for g, No in enumerate(self.n_observations):
                A[g] = np.random.rand(*([No] + self.n_states))

            A = Categorical(values=A)
        A.normalize()
        return A

    def _construct_B_likelihood(self):

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
                empirical_prior = core.get_expected_states(
                    self.qx, self.B.log(), self.action
                )
            else:
                empirical_prior = self.D.log()
        else:
            if self.action is not None:
                empirical_prior = core.get_expected_states(
                    self.qx, self.B.log(), self.action
                )
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
