#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class iplementation in Jax

__author__: Conor Heins, Dimitrije Markovic, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

import jax.numpy as jnp
from jax import nn
from . import inference, control, learning, utils, maths

class Agent(object):
    """ 
    The Agent class, the highest-level API that wraps together processes for action, perception, and learning under active inference.

    The basic usage is as follows:

    >>> my_agent = Agent(A = A, B = C, <more_params>)
    >>> observation = env.step(initial_action)
    >>> qs = my_agent.infer_states(observation)
    >>> q_pi, G = my_agent.infer_policies()
    >>> next_action = my_agent.sample_action()
    >>> next_observation = env.step(next_action)

    This represents one timestep of an active inference process. Wrapping this step in a loop with an ``Env()`` class that returns
    observations and takes actions as inputs, would entail a dynamic agent-environment interaction.
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

        self.A = A

        # self.A = pytree.map(utils.normalized, A)

        """ Determine number of observation modalities and their respective dimensions """
        self.num_obs = [self.A[m].shape[0] for m in range(len(self.A))]
        self.num_modalities = len(self.num_obs)

        """ Assigning prior parameters on observation model (pA matrices) """
        self.pA = pA

        # self.B = map( utils.normalized, B)
        self.B = B

        # Determine number of hidden state factors and their dimensionalities
        self.num_states = [self.B[f].shape[0] for f in range(len(self.B))]
        self.num_factors = len(self.num_states)

        """ Assigning prior parameters on transition model (pB matrices) """
        self.pB = pB

        # If no `num_controls` are given, then this is inferred from the shapes of the input B matrices
        self.num_controls = [self.B[f].shape[2] for f in range(self.num_factors)]

        # Users have the option to make only certain factors controllable.
        # default behaviour is to make all hidden state factors controllable
        # (i.e. self.num_states == self.num_controls)
        self.control_fac_idx = control_fac_idx
        self.policies = policies

        self.C = C

        """ Construct prior over hidden states (uniform if not specified) """
        self.D = D
        self.empirical_prior = D    

        """ Assigning prior parameters on initial hidden states (pD vectors) """
        self.pD = pD

        """ Construct prior over policies (uniform if not specified) """

        self.E = E
        
        self.prev_obs = []
        self.reset()
        
        self.action = None
        self.prev_actions = None

    def reset(self, init_qs=None):
        """
        Resets the posterior beliefs about hidden states of the agent to a uniform distribution, and resets time to first timestep of the simulation's temporal horizon.
        Returns the posterior beliefs about hidden states.

        Returns
        ---------
        qs: ``numpy.ndarray`` of dtype object
           Initialized posterior over hidden states. Depending on the inference algorithm chosen and other parameters (such as the parameters stored within ``edge_handling_paramss),
           the resulting ``qs`` variable will have additional sub-structure to reflect whether beliefs are additionally conditioned on timepoint and policy.
            For example, in case the ``self.inference_algo == 'MMP' `, the indexing structure of ``qs`` is policy->timepoint-->factor, so that 
            ``qs[p_idx][t_idx][f_idx]`` refers to beliefs about marginal factor ``f_idx`` expected under policy ``p_idx`` 
            at timepoint ``t_idx``. In this case, the returned ``qs`` will only have entries filled out for the first timestep, i.e. for ``q[p_idx][0]``, for all 
            policy-indices ``p_idx``. Subsequent entries ``q[:][1, 2, ...]`` will be initialized to empty ``numpy.ndarray`` objects.
        """

        self.curr_timestep = 0

        self.qs = utils.list_array_uniform(self.num_states)

        return self.qs

    def step_time(self):
        """
        Advances time by one step. This involves updating the ``self.prev_actions``, and in the case of a moving
        inference horizon, this also shifts the history of post-dictive beliefs forward in time (using ``self.set_latest_beliefs()``),
        so that the penultimate belief before the beginning of the horizon is correctly indexed.

        Returns
        ---------
        curr_timestep: ``int``
            The index in absolute simulation time of the current timestep.
        """

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
        Both sets and returns the penultimate belief before the first timestep of the backwards inference horizon. 
        In the case that the inference horizon includes the first timestep of the simulation, then the ``latest_belief`` is
        simply the first belief of the whole simulation, or the prior (``self.D``). The particular structure of the ``latest_belief``
        depends on the value of ``self.edge_handling_params['use_BMA']``.

        Returns
        ---------
        latest_belief: ``numpy.ndarray`` of dtype object
            Penultimate posterior beliefs over hidden states at the timestep just before the first timestep of the inference horizon. 
            Depending on the value of ``self.edge_handling_params['use_BMA']``, the shape of this output array will differ.
            If ``self.edge_handling_params['use_BMA'] == True``, then ``latest_belief`` will be a Bayesian model average 
            of beliefs about hidden states, where the average is taken with respect to posterior beliefs about policies.
            Otherwise, `latest_belief`` will be the full, policy-conditioned belief about hidden states, and will have indexing structure
            policies->factors, such that ``latest_belief[p_idx][f_idx]`` refers to the penultimate belief about marginal factor ``f_idx``
            under policy ``p_idx``.
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
        Returns the last ``self.policy_len`` timesteps of each policy-conditioned belief
        over hidden states. This is a step of pre-processing that needs to be done before computing
        the expected free energy of policies. We do this to avoid computing the expected free energy of 
        policies using beliefs about hidden states in the past (so-called "post-dictive" beliefs).

        Returns
        ---------
        future_qs_seq: ``numpy.ndarray`` of dtype object
            Posterior beliefs over hidden states under a policy, in the future. This is a nested ``numpy.ndarray`` object array, with one
            sub-array ``future_qs_seq[p_idx]`` for each policy. The indexing structure is policy->timepoint-->factor, so that 
            ``future_qs_seq[p_idx][t_idx][f_idx]`` refers to beliefs about marginal factor ``f_idx`` expected under policy ``p_idx`` 
            at future timepoint ``t_idx``, relative to the current timestep.
        """
        
        future_qs_seq = utils.obj_array(len(self.qs))
        for p_idx in range(len(self.qs)):
            future_qs_seq[p_idx] = self.qs[p_idx][-(self.policy_len+1):] # this grabs only the last `policy_len`+1 beliefs about hidden states, under each policy

        return future_qs_seq


    def infer_states(self, observations):
        """
        Update approximate posterior over hidden states by solving variational inference problem, given an observation.

        Parameters
        ----------
        observation: ``list`` or ``tuple`` of ints
            The observation input. Each entry ``observation[m]`` stores the index of the discrete
            observation for modality ``m``.

        Returns
        ---------
        qs: ``numpy.ndarray`` of dtype object
            Posterior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``qs`` variable will have additional sub-structure to reflect whether
            beliefs are additionally conditioned on timepoint and policy.
            For example, in case the ``self.inference_algo == 'MMP' `` indexing structure is policy->timepoint-->factor, so that 
            ``qs[p_idx][t_idx][f_idx]`` refers to beliefs about marginal factor ``f_idx`` expected under policy ``p_idx`` 
            at timepoint ``t_idx``.
        """

        o_vec = [nn.one_hot(o, self.A[i].shape[0]) for i, o in enumerate(observations)]
        qs = inference.update_posterior_states(
            self.A,
            o_vec,
            prior=self.empirical_prior
        )

        self.qs = qs

        return qs

    def update_empirical_prior(self, action):
        # update self.empirical_prior
        self.empirical_prior = control.compute_expected_state(
                self.qs, self.B, action
            )

    def infer_policies(self):
        """
        Perform policy inference by optimizing a posterior (categorical) distribution over policies.
        This distribution is computed as the softmax of ``G * gamma + lnE`` where ``G`` is the negative expected
        free energy of policies, ``gamma`` is a policy precision and ``lnE`` is the (log) prior probability of policies.
        This function returns the posterior over policies as well as the negative expected free energy of each policy.

        Returns
        ----------
        q_pi: 1D ``numpy.ndarray``
            Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
        G: 1D ``numpy.ndarray``
            Negative expected free energies of each policy, i.e. a vector containing one negative expected free energy per policy.
        """

        q_pi, G = control.update_posterior_policies(
            self.policies,
            self.qs,
            self.A,
            self.B,
            self.C,
            gamma = self.gamma
        )

        self.q_pi = q_pi
        self.G = G
        return q_pi, G

    def sample_action(self):
        """
        Sample or select a discrete action from the posterior over control states.
        This function both sets or cachés the action as an internal variable with the agent and returns it.
        This function also updates time variable (and thus manages consequences of updating the moving reference frame of beliefs)
        using ``self.step_time()``.
        
        Returns
        ----------
        action: 1D ``numpy.ndarray``
            Vector containing the indices of the actions for each control factor
        """

        action = control.sample_action(
            self.q_pi, self.policies, self.num_controls, self.action_selection
        )

        self.action = action

        self.step_time()

        return action

    def update_A(self, obs):
        """
        Update approximate posterior beliefs about Dirichlet parameters that parameterise the observation likelihood or ``A`` array.

        Parameters
        ----------
        observation: ``list`` or ``tuple`` of ints
            The observation input. Each entry ``observation[m]`` stores the index of the discrete
            observation for modality ``m``.

        Returns
        -----------
        qA: ``numpy.ndarray`` of dtype object
            Posterior Dirichlet parameters over observation model (same shape as ``A``), after having updated it with observations.
        """

        qA = learning.update_obs_likelihood_dirichlet(
            self.pA, 
            self.A, 
            obs, 
            self.qs, 
            self.lr_pA, 
            self.modalities_to_learn
        )

        self.pA = qA # set new prior to posterior
        self.A = utils.norm_dist_obj_arr(qA) # take expected value of posterior Dirichlet parameters to calculate posterior over A array

        return qA

    def update_B(self, qs_prev):
        """
        Update posterior beliefs about Dirichlet parameters that parameterise the transition likelihood 
        
        Parameters
        -----------
        qs_prev: 1D ``numpy.ndarray`` or ``numpy.ndarray`` of dtype object
            Marginal posterior beliefs over hidden states at previous timepoint.
    
        Returns
        -----------
        qB: ``numpy.ndarray`` of dtype object
            Posterior Dirichlet parameters over transition model (same shape as ``B``), after having updated it with state beliefs and actions.
        """

        pB_updated = learning.update_state_likelihood_dirichlet(
            self.pB,
            self.B,
            self.action,
            self.qs,
            qs_prev,
            self.lr_pB,
            self.factors_to_learn
        )

        self.pB = qB # set new prior to posterior
        self.B = utils.norm_dist_obj_arr(qB)  # take expected value of posterior Dirichlet parameters to calculate posterior over B array

        return qB
    
    def update_D(self, qs_t0 = None):
        """
        Update Dirichlet parameters of the initial hidden state distribution 
        (prior beliefs about hidden states at the beginning of the inference window).

        Parameters
        -----------
        qs_t0: 1D ``numpy.ndarray``, ``numpy.ndarray`` of dtype object, or ``None``
            Marginal posterior beliefs over hidden states at current timepoint. If ``None``, the 
            value of ``qs_t0`` is set to ``self.qs_hist[0]`` (i.e. the initial hidden state beliefs at the first timepoint).
            If ``self.inference_algo == "MMP"``, then ``qs_t0`` is set to be the Bayesian model average of beliefs about hidden states
            at the first timestep of the backwards inference horizon, where the average is taken with respect to posterior beliefs about policies.
      
        Returns
        -----------
        qD: ``numpy.ndarray`` of dtype object
            Posterior Dirichlet parameters over initial hidden state prior (same shape as ``qs_t0``), after having updated it with state beliefs.
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
        
        qD = learning.update_state_prior_dirichlet(self.pD, qs_t0, self.lr_pD, factors = self.factors_to_learn)
        
        self.pD = qD # set new prior to posterior
        self.D = utils.norm_dist_obj_arr(qD) # take expected value of posterior Dirichlet parameters to calculate posterior over D array

        return qD

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

    
    
