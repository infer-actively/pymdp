#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class implementation in Jax

__author__: Conor Heins, Dimitrije Markovic, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn, vmap
from . import inference, control, learning, utils, maths
from equinox import Module, static_field

from typing import Any, List, AnyStr, Optional

class Agent(Module):
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

    A: List
    B: List
    C: List 
    D: List
    E: jnp.ndarray
    # empirical_prior: List
    gamma: jnp.ndarray
    qs: Optional[List]
    q_pi: Optional[List]

    pA: List
    pB: List
    
    # static parameters not leaves of the PyTree
    A_dependencies: Optional[List] = static_field()
    num_iter: int = static_field()
    num_obs: List = static_field()
    num_modalities: int = static_field()
    num_states: List = static_field()
    num_factors: int = static_field()
    num_controls: List = static_field()
    inference_algo: AnyStr = static_field()
    control_fac_idx: Any = static_field()
    policy_len: int = static_field()
    policies: Any = static_field()
    use_utility: bool = static_field()
    use_states_info_gain: bool = static_field()
    use_param_info_gain: bool = static_field()
    action_selection: AnyStr = static_field()

    def __init__(
        self,
        A,
        B,
        C,
        D,
        E,
        pA,
        pB,
        A_dependencies=None,
        qs=None,
        q_pi=None,
        policy_len=1,
        control_fac_idx=None,
        policies=None,
        gamma=16.0,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=False,
        action_selection="deterministic",
        inference_algo="fpi",
        num_iter=16,
    ):
        ### PyTree leaves

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        # self.empirical_prior = D
        self.E = E
        self.pA = pA
        self.pB = pB
        self.qs = qs
        self.q_pi = q_pi

        if A_dependencies is not None:
            self.A_dependencies = A_dependencies
        else:
            num_factors = len(B)
            num_modalities = len(A)
            self.A_dependencies = [list(range(num_factors)) for _ in range(num_modalities)]


        batch_dim = (self.A[0].shape[0],)

        self.gamma = jnp.broadcast_to(gamma, batch_dim) 

        ### Static parameters ###

        self.num_iter = num_iter

        self.inference_algo = inference_algo

        # policy parameters
        self.policy_len = policy_len
        self.action_selection = action_selection
        self.use_utility = use_utility
        self.use_states_info_gain = use_states_info_gain
        self.use_param_info_gain = use_param_info_gain

        """ Determine number of observation modalities and their respective dimensions """
        self.num_obs = [self.A[m].shape[1] for m in range(len(self.A))]
        self.num_modalities = len(self.num_obs)

        # Determine number of hidden state factors and their dimensionalities
        self.num_states = [self.B[f].shape[1] for f in range(len(self.B))]
        self.num_factors = len(self.num_states)

        # If no `num_controls` are given, then this is inferred from the shapes of the input B matrices
        self.num_controls = [self.B[f].shape[-1] for f in range(self.num_factors)]

        # Users have the option to make only certain factors controllable.
        # default behaviour is to make all hidden state factors controllable
        # (i.e. self.num_states == self.num_controls)
        self.control_fac_idx = control_fac_idx
        if policies is not None:
            self.policies = policies
        else:
            self._construct_policies()

    def _construct_policies(self):
        
        self.policies =  control.construct_policies(
            self.num_states, self.num_controls, self.policy_len, self.control_fac_idx
        )

    @vmap
    def learning(self, *args, **kwargs):
        # do stuff
        # variables = ...
        # parameters = ...
        # varibles = {'A': jnp.ones(5)}

        # return Agent(variables, parameters)
        raise NotImplementedError
    
    @vmap
    def infer_states(self, observations, past_actions,  empirical_prior, qs_hist):
        """
        Update approximate posterior over hidden states by solving variational inference problem, given an observation.

        Parameters
        ----------
        observations: ``list`` or ``tuple`` of ints
            The observation input. Each entry ``observation[m]`` stores one-hot vectors representing the observations for modality ``m``.
        past_actions: ``list`` or ``tuple`` of ints
            The action input. Each entry ``past_actions[f]`` stores indices (or one-hots?) representing the actions for control factor ``f``.
        empirical_prior: ``list`` or ``tuple`` of ``jax.numpy.ndarray`` of dtype object
            Empirical prior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``empirical_prior`` variable may be a matrix (or list of matrices) 
            of additional dimensions to encode extra conditioning variables like timepoint and policy.
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
        output = inference.update_posterior_states(
            self.A,
            self.B,
            o_vec,
            past_actions,
            prior=empirical_prior,
            qs_hist=qs_hist,
            A_dependencies=self.A_dependencies,
            num_iter=self.num_iter,
            method=self.inference_algo
        )

        return output

    @vmap
    def update_empirical_prior(self, action, qs):
        # return empirical_prior, and the history of posterior beliefs (filtering distributions) held about hidden states at times 1, 2 ... t

        qs_last = jtu.tree_map( lambda x: x[-1], qs)
        pred = control.compute_expected_state(qs_last, self.B, action)
        
        return (pred, qs)

    @vmap
    def infer_policies(self, qs: List):
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

        latest_belief = jtu.tree_map(lambda x: x[-1], qs) # only get the posterior belief held at the current timepoint
        q_pi, G = control.update_posterior_policies(
            self.policies,
            latest_belief, 
            self.A,
            self.B,
            self.C,
            self.pA,
            self.pB,
            gamma=self.gamma,
            use_utility=self.use_utility,
            use_states_info_gain=self.use_states_info_gain,
            use_param_info_gain=self.use_param_info_gain
        )

        return q_pi, G
    
    @vmap
    def action_probabilities(self, q_pi: jnp.ndarray):
        """
        Compute probabilities of discrete actions from the posterior over policies.

        Parameters
        ----------
        q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
        
        Returns
        ----------
        action: 2D ``jax.numpy.ndarray``
            Vector containing probabilities of possible actions for different factors
        """

        marginals = control.get_marginals(q_pi, self.policies, self.num_controls)

        # make all arrays same length (add 0 probability)
        lengths = jtu.tree_map(lambda x: len(x), marginals)
        max_length = max(lengths)
        marginals = jtu.tree_map(lambda x: jnp.pad(x, (0, max_length - len(x))), marginals)

        return jnp.stack(marginals, -2)


    def sample_action(self, q_pi: jnp.ndarray):
        """
        Sample or select a discrete action from the posterior over control states.
        
        Returns
        ----------
        action: 1D ``jax.numpy.ndarray``
            Vector containing the indices of the actions for each control factor
        """

        sample_action = lambda x: control.sample_action(x, self.policies, self.num_controls, self.action_selection)

        action = vmap(sample_action)(q_pi)

        return action

    def _get_default_params(self):
        method = self.inference_algo
        default_params = None
        if method == "VANILLA":
            default_params = {"num_iter": 10, "dF": 1.0, "dF_tol": 0.001}
        elif method == "MMP":
            raise NotImplementedError("MMP is not implemented")
        elif method == "VMP":
            raise NotImplementedError("VMP is not implemented")
        elif method == "BP":
            raise NotImplementedError("BP is not implemented")
        elif method == "EP":
            raise NotImplementedError("EP is not implemented")
        elif method == "CV":
            raise NotImplementedError("CV is not implemented")

        return default_params