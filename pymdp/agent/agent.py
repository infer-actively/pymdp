#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

import numpy as np
from pymdp.distributions import Categorical, Dirichlet
from pymdp.core import inference, control, learning
from pymdp.core import utils
import copy

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
        inference_horizon=1,
        control_fac_idx=None,
        policies=None,
        gamma=16.0,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=False,
        action_sampling="marginal_action",
        inference_algo="VANILLA",
        inference_params=None,
        modalities_to_learn="all",
        lr_pA=1.0,
        factors_to_learn="all",
        lr_pB=1.0,
        use_BMA = False,
        policy_sep_prior = False
    ):

        ### Constant parameters ###

        # policy parameters
        self.policy_len = policy_len
        self.gamma = gamma
        self.action_sampling = action_sampling
        self.use_utility = use_utility
        self.use_states_info_gain = use_states_info_gain
        self.use_param_info_gain = use_param_info_gain

        # learning parameters
        self.modalities_to_learn = modalities_to_learn
        self.lr_pA = lr_pA
        self.factors_to_learn = factors_to_learn
        self.lr_pB = lr_pB

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
                self.n_observations = [
                    self.A[modality].shape[0] for modality in range(self.n_modalities)
                ]
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
                self.pA = Dirichlet(values=pA)
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
            self.n_factors = len(self.n_factors) #type: ignore
            construct_B_flag = True

        """ Initialise prior Dirichlet parameters on transition model (pB matrices) """
        if pB is not None:
            if not isinstance(pB, Dirichlet):
                self.pB = Dirichlet(values=pA)
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
            _, self.n_controls = self._construct_n_controls()
        else:
            self.n_controls = n_controls

        # Again, the use can specify a set of possible policies, or
        # all possible combinations of actions and timesteps will be considered
        if policies is None:
            self.policies, _ = self._construct_n_controls()
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
            self.inference_algo = "VANILLA"
            self.inference_params = self._get_default_params()
            if inference_horizon > 1:
                print("WARNING: if `inference_algo` is VANILLA, then inference_horizon must be 1\n. \
                    Setting inference_horizon to default value of 1...\n")
            else:
                self.inference_horizon = 1
        else:
            self.inference_algo = inference_algo
            self.inference_params = self._get_default_params()
            self.inference_horizon = inference_horizon
        
        self.edge_handling_params = {}
        self.edge_handling_params['use_BMA'] = use_BMA
        self.edge_handling_params['policy_sep_prior'] = policy_sep_prior

        self.reset()
        
        self.action = None
        self.prev_actions = None

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
        policies, n_controls = control.construct_policies(
            self.n_states, None, self.policy_len, self.control_fac_idx
        )

        return policies, n_controls

    def reset(self, init_qs=None):

        self.curr_timestep = 1

        if init_qs is None:
            if self.inference_horizon == 1:
                self.qs = self._construct_D_prior()
            else: # in the case you're doing MMP (i.e. you have an inference_horizon > 1), we have to account for policy- and timestep-conditioned posterior beliefs
                self.qs = utils.obj_array(len(self.policies))
                for p_i, _ in enumerate(self.policies):
                    self.qs[p_i] = utils.obj_array(inference_horizon + self.policy_len + 1) # + 1 to include belief about current timestep
                    self.qs[p_i][0] = copy.deepcopy(self.D) # initialize the very first belief of the inference_len as the prior over initial hidden states
                
                first_belief = utils.obj_array(len(self.policies))
                for p_i, _ in enumerate(self.policies):
                    first_belief[p_i] = copy.deepcopy(self.D) 
                
                if self.edge_handling_params['policy_sep_prior']:
                    self.set_latest_beliefs(last_belief = first_belief)
                else:
                    self.set_latest_beliefs(last_belief = self.D)
            
        else:
            if isinstance(init_qs, Categorical):
                self.qs = init_qs
            else:
                self.qs = Categorical(values=init_qs)

        return self.qs

    def step_time(self):

        self.curr_timestep += 1

        if self.inference_algo == "MMP":
            if (self.curr_timestep - self.inference_horizon) > 0:
                self.set_latest_beliefs()
        
        return self.curr_timestep
    
    def set_latest_beliefs(self,last_belief=None):
        """
        This method sets the 'last' belief before the inference horizon. In the case that the inference
        horizon reaches back to the first timestep of the simulation, then the `latest_belief` is
        identical to the first belief / the prior (`self.D`). 
        """

        if last_belief is None:
            last_belief = utils.obj_array(len(self.policies))
            for p_i, _ in enumerate(self.policies):
                last_belief[p_i] = copy.deepcopy(self.qs[p_i][0])

        if self.edge_handling_params['use_BMA']:
            self.latest_belief = inference.average_states_over_policies(last_belief, self.q_pi) # average the earliest marginals together using posterior over policies (`self.q_pi`)
        else:
            self.latest_belief = last_belief

        return self.latest_belief

    def infer_states(self, observation):
        observation = tuple(observation)

        if not hasattr(self, "qs"):
            self.reset()

        if self.inference_algo is "VANILLA":
            if self.action is not None:
                empirical_prior = control.get_expected_states(
                    self.qs, self.B.log(), self.action.reshape(1, -1) #type: ignore
                )
            else:
                empirical_prior = self.D.log()
            qs = inference.update_posterior_states(
            self.A,
            observation,
            empirical_prior,
            return_numpy=False,
            method=self.inference_algo,
            **self.inference_params
            )
        elif self.inference_algo is "MMP":
            
            qs, F = inference.update_posterior_states_v2(
                self.A, 
                self.B, 
                prev_obs, 
                self.policies, 
                self.prev_actions, 
                prior = self.latest_belief, 
                policy_sep_prior = self.edge_handling_params['policy_sep_prior'],
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

        self.step_time()

        return action

    def update_A(self, obs):

        pA_updated = learning.update_likelihood_dirichlet(
            self.pA, self.A, obs, self.qs, self.lr_pA, self.modalities_to_learn, return_numpy=False
        )

        self.pA = pA_updated
        self.A = pA_updated.mean() 

        return pA_updated

    def update_B(self, qs_prev):

        pB_updated = learning.update_transition_dirichlet(
            self.pB,
            self.B,
            self.action,
            self.qs,
            qs_prev,
            self.lr_pB,
            self.factors_to_learn,
            return_numpy=False,
        )

        self.pB = pB_updated
        self.A = pB_updated.mean()

        return pB_updated

    def _get_default_params(self):
        method = self.inference_algo
        default_params = None
        if method == "VANILLA":
            default_params = {"num_iter": 10, "dF": 1.0, "dF_tol": 0.001}
        elif method == "MMP":
            default_params = {"num_iter": 10, "grad_descent": False, "tau": 0.25, "save_vfe_seq": False}
        elif method == "VMP":
            raise NotImplementedError("VMP is not implemented")
        elif method == "BP":
            raise NotImplementedError("BP is not implemented")
        elif method == "EP":
            raise NotImplementedError("EP is not implemented")
        elif method == "CV":
            raise NotImplementedError("CV is not implemented")

        return default_params
