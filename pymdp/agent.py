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
        A,
        B,
        C=None,
        D=None,
        E = None,
        pA=None,
        pB = None,
        pD = None,
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
        lr_pD=1.0,
        use_BMA = True,
        policy_sep_prior = False,
        save_belief_hist = False
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
        self.lr_pD = lr_pD

        """ Initialise observation model (A matrices) """
        if not isinstance(A, np.ndarray):
            raise TypeError(
                'A matrix must be a numpy array'
            )

        self.A = utils.to_arr_of_arr(A)

        assert utils.is_normalized(self.A), "A matrix is not normalized (i.e. A.sum(axis = 0) must all equal 1.0"

        """ Determine number of observation modalities and their respective dimensions """
        self.num_obs = [self.A[m].shape[0] for m in range(len(self.A))]
        self.num_modalities = len(self.num_obs)

        """ Assigning prior parameters on observation model (pA matrices) """
        self.pA = pA

        """ Initialise transition model (B matrices) """
        if not isinstance(B, np.ndarray):
            raise TypeError(
                'B matrix must be a numpy array'
            )

        self.B = utils.to_arr_of_arr(B)

        assert utils.is_normalized(self.B), "A matrix is not normalized (i.e. A.sum(axis = 0) must all equal 1.0"

        # Determine number of hidden state factors and their dimensionalities
        self.num_states = [self.B[f].shape[0] for f in range(len(self.B))]
        self.num_factors = len(self.num_states)

        """ Assigning prior parameters on transition model (pB matrices) """
        self.pB = pB

        # If no `num_controls` are given, then this is inferred from the shapes of the input B matrices
        if num_controls == None:
            self.num_controls = [self.B[f].shape[2] for f in range(self.num_factors)]

        # Users have the option to make only certain factors controllable.
        # default behaviour is to make all hidden state factors controllable
        # (i.e. self.num_states == self.num_controls)
        if control_fac_idx == None:
            self.control_fac_idx = [f for f in range(self.num_factors) if self.num_controls[f] > 1]
        else:

            assert max(control_fac_idx) <= (self.num_factors - 1), "Check control_fac_idx - must be consistent with `num_states` and `num_factors`..."
            self.control_fac_idx = control_fac_idx

            for factor_idx in self.control_fac_idx:
                assert self.num_controls[factor_idx] > 1, "Control factor (and B matrix) dimensions are not consistent with user-given control_fac_idx"

        # Again, the use can specify a set of possible policies, or
        # all possible combinations of actions and timesteps will be considered
        if policies == None:
            policies = self._construct_policies()
        self.policies = policies

        assert all([len(self.num_controls) == policy.shape[1] for policy in self.policies]), "Number of control states is not consistent with policy dimensionalities"
        
        all_policies = np.vstack(self.policies)

        assert all([n_c == max_action for (n_c, max_action) in zip(self.num_controls, list(np.max(all_policies, axis =0)+1))]), "Maximum number of actions is not consistent with `num_controls`"

        """ Construct prior preferences (uniform if not specified) """

        if C is not None:
            if not isinstance(C, np.ndarray):
                raise TypeError(
                    'C vector must be a numpy array'
                )
            self.C = utils.to_arr_of_arr(C)

            assert len(self.C) == self.num_modalities, f"Check C vector: number of sub-arrays must be equal to number of observation modalities: {self.num_modalities}"

            for modality, c_m in enumerate(self.C):
                assert c_m.shape[0] == self.num_obs[modality], f"Check C vector: number of rows of C vector for modality {modality} should be equal to {self.num_obs[modality]}"
        else:
            self.C = self._construct_C_prior()

        """ Construct prior over hidden states (uniform if not specified) """
    
        if D is not None:
            if not isinstance(D, np.ndarray):
                raise TypeError(
                    'D vector must be a numpy array'
                )
            self.D = utils.to_arr_of_arr(D)

            assert len(self.D) == self.num_factors, f"Check D vector: number of sub-arrays must be equal to number of hidden state factors: {self.num_factors}"

            for f, d_f in enumerate(self.D):
                assert d_f.shape[0] == self.num_states[f], f"Check D vector: number of entries of D vector for factor {f} should be equal to {self.num_states[f]}"
        else:
            if pD is not None:
                self.D = utils.norm_dist_obj_arr(pD)
            else:
                self.D = self._construct_D_prior()

        assert utils.is_normalized(self.D), "A matrix is not normalized (i.e. A.sum(axis = 0) must all equal 1.0"

        """ Assigning prior parameters on initial hidden states (pD vectors) """
        self.pD = pD

        """ Construct prior over policies (uniform if not specified) """

        if E is not None:
            if not isinstance(E, np.ndarray):
                raise TypeError(
                    'E vector must be a numpy array'
                )
            self.E = E

            assert len(self.E) == len(self.policies), f"Check E vector: length of E must be equal to number of policies: {len(self.policies)}"

        else:
            self.E = self._construct_E_prior()
        
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

        if save_belief_hist:
            self.qs_hist = []
            self.q_pi_hist = []
        
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
            self.num_states, self.num_controls, self.policy_len, self.control_fac_idx
        )

        return policies

    def _construct_num_controls(self):
        num_controls = control.get_num_controls_from_policies(
            self.policies
        )
        
        return num_controls
    
    def _construct_E_prior(self):
        E = np.ones(len(self.policies)) / len(self.policies)
        return E

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

        begin_horizon_step = self.curr_timestep - self.inference_horizon
        if self.edge_handling_params['use_BMA'] and (begin_horizon_step >= 0):
            if hasattr(self, "q_pi_hist"):
                self.latest_belief = inference.average_states_over_policies(last_belief, self.q_pi_hist[begin_horizon_step]) # average the earliest marginals together using contemporaneous posterior over policies (`self.q_pi_hist[0]`)
            else:
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

        if hasattr(self, "qs_hist"):
            self.qs_hist.append(qs)
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

        if hasattr(self, "qs_hist"):
            self.qs_hist.append(qs)

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
                E = self.E,
                gamma = self.gamma
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
                F = self.F,
                E = self.E,
                gamma = self.gamma
            )

        if hasattr(self, "q_pi_hist"):
            self.q_pi_hist.append(q_pi)
            if len(self.q_pi_hist) > self.inference_horizon:
                self.q_pi_hist = self.q_pi_hist[-(self.inference_horizon-1):]

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
        """
        Update posterior beliefs about Dirichlet parameters that parameterise the observation likelihood 
        """

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
        """
        Update posterior beliefs about Dirichlet parameters that parameterise the transition likelihood 
        """

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
    
    def update_D(self, qs_t0 = None):
        """
        Update posterior beliefs about Dirichlet parameters that parameterise the prior over initial hidden states
        """
        
        if self.inference_algo == "VANILLA":
            
            if qs_t0 is None:
                
                try:
                    qs_t0 = self.qs_hist[0]
                except ValueError:
                    print("qs_t0 must either be passed as argument to `update_D` or `save_belief_hist` must be set to True!")             

        elif self.inference_algo == "MMP":
            
            if self.edge_handling_params['use_BMA']:
                qs_t0 = self.latest_belief
            elif self.edge_handling_params['policy_sep_prior']:
              
                qs_pi_t0 = self.latest_belief

                # get beliefs about policies at the time at the beginning of the inference horizon
                if hasattr(self, "q_pi_hist"):
                    begin_horizon_step = max(0, self.curr_timestep - self.inference_horizon)
                    q_pi_t0 = np.copy(self.q_pi_hist[begin_horizon_step])
                else:
                    q_pi_t0 = np.copy(self.q_pi)
            
                qs_t0 = inference.average_states_over_policies(qs_pi_t0,q_pi_t0) # beliefs about hidden states at the first timestep of the inference horizon
        
        pD_updated = learning.update_state_prior_dirichlet(self.pD, qs_t0, self.lr_pD, factors = self.factors_to_learn)
        
        self.pD = pD_updated
        self.D = utils.norm_dist_obj_arr(self.pD)

        return pD_updated

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

    
    
