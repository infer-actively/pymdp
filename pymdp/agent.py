#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class

__author__: Conor Heins, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

import warnings
import numpy as np
from pymdp import inference, control, learning
from pymdp import utils, maths
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
        num_states=None,
        num_obs=None,
        num_controls=None,
        policy_len=1,
        inference_horizon=1,
        control_fac_idx=None,
        policies=None,
        gamma=16.0,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=False,
        action_selection="deterministic",
        inference_algo="VANILLA",
        inference_params=None,
        modalities_to_learn="all",
        lr_pA=1.0,
        factors_to_learn="all",
        lr_pB=1.0,
        use_BMA = True,
        policy_sep_prior = False
    ):

        ### Constant parameters ###

        # policy parameters
        self.policy_len = policy_len
        self.gamma = gamma
        self.action_selection = action_selection
        self.use_utility = use_utility
        self.use_states_info_gain = use_states_info_gain
        self.use_param_info_gain = use_param_info_gain

        # learning parameters
        self.modalities_to_learn = modalities_to_learn
        self.lr_pA = lr_pA
        self.factors_to_learn = factors_to_learn
        self.lr_pB = lr_pB

        """ Initialise observation model (A matrices) """
        if not isinstance(A, np.ndarray):
            raise TypeError(
                'A matrix must be a numpy array'
            )
        self.A = A

        A = utils.to_arr_of_arr(A)

        # Determine number of modalities and their dimensionaliti
        self.num_obs = [A[m].shape[0] for m in range(len(A))]
        self.num_modalities = len(self.num_obs)

        """ Assigning prior parameters on observation model (pA matrices) """
        self.pA = pA

        """ Initialise transition model (B matrices) """
        if not isinstance(B, np.ndarray):
            raise TypeError(
                'B matrix must be a numpy array'
            )
        self.B = B

        B = utils.to_arr_of_arr(B)

        # Determine number of hidden state factors and their dimensionalities
        self.num_states = [B[f].shape[0] for f in range(len(B))]
        self.num_factors = len(self.num_states)

        """ Assigning prior parameters on transition model (pB matrices) """
        self.pB = pB

        # Users have the option to make only certain factors controllable.
        # default behaviour is to make all hidden state factors controllable
        # (i.e. self.num_states == self.num_controls)
        if control_fac_idx == None:
            self.control_fac_idx = [f for f in range(self.num_factors) if B[f].shape[2] > 1]
        else:
            self.control_fac_idx = control_fac_idx

            for factor_idx in control_fac_idx:
                assert B[factor_idx].shape[2] > 1, "B matrix dimensions are not consistent with control factors"

        # Again, the use can specify a set of possible policies, or
        # all possible combinations of actions and timesteps will be considered
        if policies == None:
            policies = self._construct_policies()
        self.policies = policies

        # The user can specify the number of control states
        # However, given the controllable factors, this can be inferred
        if num_controls == None:
            num_controls = self._construct_num_controls()
        self.num_controls = num_controls

        assert all([len(self.num_controls) == policy.shape[1] for policy in self.policies]), "Number of control states is not consistent with policy dimensionalities"
        
        all_policies = np.vstack(self.policies)

        assert all([n_c == max_action for (n_c, max_action) in zip(self.num_controls, list(np.max(all_policies, axis =0)+1))]), "Maximum number of actions is not consistent with `num_controls`"

        # Construct prior preferences (uniform if not specified)
        """ Initialise transition model (B matrices) """
        if C is not None:
            if not isinstance(C, np.ndarray):
                raise TypeError(
                    'C vector must be a numpy array'
                )
            self.C = utils.to_arr_of_arr(C)
        else:
            self.C = self._construct_C_prior()

        # Construct initial beliefs (uniform if not specified)
    
        if D is not None:
            if not isinstance(D, np.ndarray):
                raise TypeError(
                    'D vector must be a numpy array'
                )
            self.D = utils.to_arr_of_arr(D)
        else:
            self.D = self._construct_D_prior()

        self.edge_handling_params = {}
        self.edge_handling_params['use_BMA'] = use_BMA # creates a 'D-like' moving prior
        self.edge_handling_params['policy_sep_prior'] = policy_sep_prior # carries forward last timesteps posterior, in a policy-conditioned way

        # use_BMA and policy_sep_prior can both be False, but both cannot be simultaneously be True. If one of them is True, the other must be False
        if policy_sep_prior:
            if use_BMA:
                warnings.warn(
                    "Inconsistent choice of `policy_sep_prior` and `use_BMA`.\
                    You have set `policy_sep_prior` to True, so we are setting `use_BMA` to False"
                )
                self.edge_handling_params['use_BMA'] = False
        
        if inference_algo == None:
            self.inference_algo = "VANILLA"
            self.inference_params = self._get_default_params()
            if inference_horizon > 1:
                warnings.warn(
                    "If `inference_algo` is VANILLA, then inference_horizon must be 1\n. \
                    Setting inference_horizon to default value of 1...\n"
                    )
                self.inference_horizon = 1
            else:
                self.inference_horizon = 1
        else:
            self.inference_algo = inference_algo
            self.inference_params = self._get_default_params()
            self.inference_horizon = inference_horizon
        
        self.prev_obs = []
        self.reset()
        
        self.action = None
        self.prev_actions = None

    def _construct_C_prior(self):
        
        C = utils.obj_array_zeros(self.num_obs)

        return C

    def _construct_D_prior(self):

        D = utils.obj_array_uniform(self.num_states)

        return D

    def _construct_policies(self):
        policies =  control.construct_policies(
            self.num_states, None, self.policy_len, self.control_fac_idx
        )

        return policies

    def _construct_num_controls(self):
        num_controls = control.get_num_controls_from_policies(
            self.policies
        )
        
        return num_controls

    def reset(self, init_qs=None):

        self.curr_timestep = 0

        if init_qs is None:
            if self.inference_algo == 'VANILLA':
                self.qs = utils.obj_array_uniform(self.num_states)
            else: # in the case you're doing MMP (i.e. you have an inference_horizon > 1), we have to account for policy- and timestep-conditioned posterior beliefs
                self.qs = utils.obj_array(len(self.policies))
                for p_i, _ in enumerate(self.policies):
                    self.qs[p_i] = utils.obj_array(self.inference_horizon + self.policy_len + 1) # + 1 to include belief about current timestep
                    self.qs[p_i][0] = utils.obj_array_uniform(self.num_states)
                
                first_belief = utils.obj_array(len(self.policies))
                for p_i, _ in enumerate(self.policies):
                    first_belief[p_i] = copy.deepcopy(self.D) 
                
                if self.edge_handling_params['policy_sep_prior']:
                    self.set_latest_beliefs(last_belief = first_belief)
                else:
                    self.set_latest_beliefs(last_belief = self.D)
            
        else:
            self.qs = init_qs

        return self.qs

    def step_time(self):

        if self.prev_actions is None:
            self.prev_actions = [self.action]
        else:
            self.prev_actions.append(self.action)

        self.curr_timestep += 1

        if self.inference_algo == "MMP" and (self.curr_timestep - self.inference_horizon) >= 0:
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

        if self.edge_handling_params['use_BMA'] and (self.curr_timestep - self.inference_horizon >= 0):
            self.latest_belief = inference.average_states_over_policies(last_belief, self.q_pi) # average the earliest marginals together using posterior over policies (`self.q_pi`)
        else:
            self.latest_belief = last_belief

        return self.latest_belief
    
    def get_future_qs(self):
        """
        This method only gets the last `policy_len` timesteps of each policy-conditioned belief
        over hidden states. This is a step of pre-processing that needs to be done before computing
        the expected free energy of policies. We do this to avoid computing the expected free energy of 
        policies using ('post-dictive') beliefs about hidden states in the past
        """
        
        future_qs_seq = utils.obj_array(len(self.qs))
        for p_idx in range(len(self.qs)):
            future_qs_seq[p_idx] = self.qs[p_idx][-(self.policy_len+1):] # this grabs only the last `policy_len`+1 beliefs about hidden states, under each policy

        return future_qs_seq


    def infer_states(self, observation):
        '''
        Docstring @ TODO 
        Update variational posterior over hidden states, i.e. Q(\tilde{s})
        Parameters
        ----------
        `self` [type]:
            -Description
        `observation` [list or tuple of ints]:
            The observation (generated by the environment). Each observation[m] stores the index of the discrete
            observation for that modality.
        Returns:
        ---------
        `qs` [numpy object array]:
            - posterior beliefs over hidden states. 
        '''

        observation = tuple(observation) 

        if not hasattr(self, "qs"):
            self.reset()

        if self.inference_algo == "VANILLA":
            if self.action is not None:
                empirical_prior = control.get_expected_states(
                    self.qs, self.B, self.action.reshape(1, -1) #type: ignore
                )[0]
            else:
                empirical_prior = self.D
            qs = inference.update_posterior_states(
            self.A,
            observation,
            empirical_prior,
            **self.inference_params
            )
        elif self.inference_algo == "MMP":

            self.prev_obs.append(observation)
            if len(self.prev_obs) > self.inference_horizon:
                latest_obs = self.prev_obs[-self.inference_horizon:]
                latest_actions = self.prev_actions[-(self.inference_horizon-1):]
            else:
                latest_obs = self.prev_obs
                latest_actions = self.prev_actions

            qs, F = inference.update_posterior_states_v2(
                self.A,
                self.B,
                latest_obs,
                self.policies, 
                latest_actions, 
                prior = self.latest_belief, 
                policy_sep_prior = self.edge_handling_params['policy_sep_prior'],
                **self.inference_params
            )

            self.F = F # variational free energy of each policy  
    
        self.qs = qs

        return qs

    def infer_states_test(self, observation):
        observation = tuple(observation)

        if not hasattr(self, "qs"):
            self.reset()

        if self.inference_algo == "VANILLA":
            if self.action is not None:
                empirical_prior = control.get_expected_states(
                    self.qs, self.B, self.action.reshape(1, -1) #type: ignore
                )
            else:
                empirical_prior = self.D
            qs = inference.update_posterior_states(
            self.A,
            observation,
            empirical_prior,
            **self.inference_params
            )
        elif self.inference_algo == "MMP":

            self.prev_obs.append(observation)
            if len(self.prev_obs) > self.inference_horizon:
                latest_obs = self.prev_obs[-self.inference_horizon:]
                latest_actions = self.prev_actions[-(self.inference_horizon-1):]
            else:
                latest_obs = self.prev_obs
                latest_actions = self.prev_actions

            qs, F, xn, vn = inference.update_posterior_states_v2_test(
                self.A,
                self.B, 
                latest_obs,
                self.policies, 
                latest_actions, 
                prior = self.latest_belief, 
                policy_sep_prior = self.edge_handling_params['policy_sep_prior'],
                **self.inference_params
            )

            self.F = F # variational free energy of each policy  
    
        self.qs = qs

        return qs, xn, vn

    def infer_policies(self):

        if self.inference_algo == "VANILLA":
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
                self.gamma
            )
        elif self.inference_algo == "MMP":

            future_qs_seq = self.get_future_qs()

            q_pi, efe = control.update_posterior_policies_mmp(
                future_qs_seq,
                self.A,
                self.B,
                self.C,
                self.policies,
                self.use_utility,
                self.use_states_info_gain,
                self.use_param_info_gain,
                self.latest_belief,
                self.pA,
                self.pB,
                self.F,
                E = None,
                gamma = self.gamma
            )

        self.q_pi = q_pi
        self.efe = efe
        return q_pi, efe

    def sample_action(self):
        action = control.sample_action(
            self.q_pi, self.policies, self.num_controls, self.action_selection
        )

        self.action = action

        self.step_time()

        return action

    def update_A(self, obs):

        pA_updated = learning.update_likelihood_dirichlet(
            self.pA, 
            self.A, 
            obs, 
            self.qs, 
            self.lr_pA, 
            self.modalities_to_learn
        )

        self.pA = pA_updated
        self.A = utils.norm_dist_obj_arr(self.pA) 

        return pA_updated

    def update_B(self, qs_prev):

        pB_updated = learning.update_transition_dirichlet(
            self.pB,
            self.B,
            self.action,
            self.qs,
            qs_prev,
            self.lr_pB,
            self.factors_to_learn
        )

        self.pB = pB_updated
        self.B = utils.norm_dist_obj_arr(self.pB) 

        return pB_updated

    def _get_default_params(self):
        method = self.inference_algo
        default_params = None
        if method == "VANILLA":
            default_params = {"num_iter": 10, "dF": 1.0, "dF_tol": 0.001}
        elif method == "MMP":
            default_params = {"num_iter": 10, "grad_descent": True, "tau": 0.25}
        elif method == "VMP":
            raise NotImplementedError("VMP is not implemented")
        elif method == "BP":
            raise NotImplementedError("BP is not implemented")
        elif method == "EP":
            raise NotImplementedError("EP is not implemented")
        elif method == "CV":
            raise NotImplementedError("CV is not implemented")

        return default_params

    
    
