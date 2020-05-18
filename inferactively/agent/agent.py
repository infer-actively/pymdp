#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

import numpy as np
from inferactively.distributions import Categorical, Dirichlet
from inferactively.core import inference, control

class Agent(object):
    """ 
    Agent class 
    """

    def __init__(
        self,
        A=None,
        pA=None,
        B=None,
        pB=None,
        C=None,
        D=None,
        n_states=None,
        n_observations=None,
        n_controls=None,
        policy_len=1,
        control_fac_idx=None,
        policies=None,
        gamma=16.0,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=False,
        action_sampling="marginal_action",
        inference_algo="FPI",
        inference_params=None,
    ):

        # Constant parameters
        self.policy_len = policy_len
        self.gamma = gamma
        self.action_sampling = action_sampling


        """ Initialise observation model (A matrices) """
        if A is not None:
            # Create `Categorical`
            if not isinstance(A, Categorical):
                self.A = Categorical(values=A)
            else:
                self.A = A

            # Determine number of modalities and observations
            if self.A.IS_AOA:
                self.n_modalities = self.A.shape[0]
                self.n_observations = [self.A[modality].shape[0] for modality in range(self.n_modalities)]
            else:
                self.n_modalities = 1
                self.n_observations = [self.A.shape[0]]
            construct_A_flag = False
        else:
            
            # If A is none, we randomly initialise the matrix. This requires some information
            if n_observations is None:
                raise ValueError(
                    "Must provide either `A` or `n_observations` to `Agent` constructor"
                )
            self.n_observations = n_observations
            self.n_modalities = len(self.n_observations)
            construct_A_flag = True
        
        """ Initialise prior Dirichlet parameters on observation model (pA matrices) """
        if pA is not None:
            if not isinstance(pA, Dirichlet):
                self.pA = Dirichlet(values = pA)
            else:
                self.pA = pA
        else:
            self.pA = None

        """ Initialise transition model (B matrices) """
        if B is not None:
            if not isinstance(B, Categorical):
                self.B = Categorical(values=B)
            else:
                self.B = B

            # Same logic as before, but here we need number of factors and states per factor
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

        """ Initialise prior Dirichlet parameters on transition model (pB matrices) """
        if pB is not None:
            if not isinstance(pB, Dirichlet):
                self.pB = Dirichlet(values = pA)
            else:
                self.pB = pB
        else:
            self.pB = None

        # Users have the option to make only certain factors controllable.
        # default behaviour is to make all hidden state factors controllable
        # (i.e. self.n_states == self.n_controls)
        if control_fac_idx is None:
            self.control_fac_idx = list(range(self.n_factors))
        else:
            self.control_fac_idx = control_fac_idx

        # The user can specify the number of control states
        # However, given the controllable factors, this can be inferred 
        if n_controls is None:
            self.n_controls, _ = self._construct_n_controls()
        else:
            self.n_controls = n_controls

        # Again, the use can specify a set of possible policies, or
        # all possible combinations of actions and timesteps will be considered
        if policies is None:
            _, self.policies = self._construct_n_controls()
        else:
            self.policies = policies

        # Construct prior preferences (uniform if not specified)
        if C is not None:
            if isinstance(C, Categorical):
                self.C = C
            else:
                self.C = Categorical(values=C)
        else:
            self.C = self._construct_C_prior()
        
        # Construct initial beliefs (uniform if not specified)
        if D is not None:
            if isinstance(D, Categorical):
                self.D = D
            else:
                self.D = Categorical(values=D)
        else:
            self.D = self._construct_D_prior()

        # Build model
        if construct_A_flag:
            self.A = self._construct_A_distribution()
        if construct_B_flag:
            self.B = self._construct_B_distribution()

        if inference_algo is None:
            self.inference_algo = "FPI"
            self.inference_params = self._get_default_params()
        else:
            self.inference_algo = inference_algo
            self.inference_params = self._get_default_params()

        self.qs = self.D
        self.action = None

    def _construct_A_distribution(self):
        if self.n_modalities == 1:
            A = Categorical(values=np.random.rand(*(self.n_observations[0] + self.n_states)))
        else:
            A = np.empty(self.n_modalities, dtype=object)
            for modality, no in enumerate(self.n_observations):
                A[modality] = np.random.rand(*([no] + self.n_states))
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

            for factor, ns in enumerate(self.n_states):
                B_basic = np.eye(ns)[:, :, np.newaxis]
                if factor in self.control_fac_idx:
                    B[factor] = np.tile(B_basic, (1, 1, self.n_controls[factor]))
                    B[factor] = B[factor].transpose(1, 2, 0)
                else:
                    B[factor] = B_basic

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
        n_controls, policies = control.construct_policies(
            self.n_states, self.n_controls, self.policy_len, self.control_fac_idx)

        return n_controls, policies

    def reset(self, init_qs=None):
        if init_qs is None:
            self.qs = self._construct_D_prior()
        else:
            if isinstance(init_qs, Categorical):
                self.qs = init_qs
            else:
                self.qs = Categorical(values=init_qs)

        return self.qs

    def infer_states(self, observation):
        if not hasattr(self, "qs"):
            self.reset()

        if self.inference_algo is "FPI":
            if self.action is not None:
                empirical_prior = control.get_expected_states(self.qs, self.B.log(), self.action.reshape(1,-1))
            else:
                empirical_prior = self.D.log()
        else:
            if self.action is not None:
                empirical_prior = control.get_expected_states(self.qs, self.B, self.action.reshape(1,-1))
            else:
                empirical_prior = self.D

        qs = inference.update_posterior_states(
            self.A,
            observation,
            empirical_prior,
            return_numpy=False,
            method=self.inference_algo,
            **self.inference_params
        )
        self.qs = qs

        return qs

    def infer_policies(self):

        q_pi, efe = control.update_posterior_policies(
            self.qs,
            self.A,
            self.B,
            self.C,
            self.policies,
            self.use_utility,
            self.use_states_info_gain,
            self.use_param_info_gain,
            self.pA,
            self.pB,
            self.gamma,
            return_numpy=False,
        )
        self.q_pi = q_pi
        self.efe = efe
        return q_pi, efe

    def sample_action(self):
        action = control.sample_action(
            self.q_pi, self.policies, self.n_controls, self.action_sampling
        )

        self.action = action
        return action

    def _get_default_params(self):

        method = self.inference_algo

        if method == "FPI":
            default_params = {"num_iter": 10, "dF": 1.0, "dF_tol": 0.001}
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
