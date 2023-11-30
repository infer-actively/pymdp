#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class implementation in Jax

__author__: Conor Heins, Dimitrije Markovic, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn, vmap
from . import inference, control, learning, utils, maths
from equinox import Module, static_field, tree_at

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
    alpha: jnp.ndarray
    qs: Optional[List]
    q_pi: Optional[List]

    pA: List
    pB: List
    
    # static parameters not leaves of the PyTree
    A_dependencies: Optional[List] = static_field()
    B_dependencies: Optional[List] = static_field()
    num_iter: int = static_field()
    num_obs: List = static_field()
    num_modalities: int = static_field()
    num_states: List = static_field()
    num_factors: int = static_field()
    num_controls: List = static_field()
    control_fac_idx: Any = static_field()
    policy_len: int = static_field()
    policies: Any = static_field()
    use_utility: bool = static_field()
    use_states_info_gain: bool = static_field()
    use_param_info_gain: bool = static_field()
    action_selection: AnyStr = static_field() # determinstic or stochastic
    sampling_mode : AnyStr = static_field() # whether to sample from full posterior over policies ("full") or from marginal posterior over actions ("marginal")
    inference_algo: AnyStr = static_field() # fpi, vmp, mmp, ovf

    learn_A: bool = static_field()
    learn_B: bool = static_field()
    learn_C: bool = static_field()
    learn_D: bool = static_field()
    learn_E: bool = static_field()

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
        B_dependencies=None,
        qs=None,
        q_pi=None,
        policy_len=1,
        control_fac_idx=None,
        policies=None,
        gamma=16.0,
        alpha=16.0,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=False,
        action_selection="deterministic",
        sampling_mode="marginal",
        inference_algo="fpi",
        num_iter=16,
        learn_A=True,
        learn_B=True,
        learn_C=False,
        learn_D=True,
        learn_E=False
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

        element_size = lambda x: x.shape[1]
        self.num_factors = len(self.B)
        self.num_states = jtu.tree_map(element_size, self.B) 

        self.num_modalities = len(self.A)
        self.num_obs = jtu.tree_map(element_size, self.A)

        # Ensure consistency of A_dependencies with num_states and num_factors
        if A_dependencies is not None:
            self.A_dependencies = A_dependencies
        else:
            # assume full dependence of A matrices and state factors
            self.A_dependencies = [list(range(self.num_factors)) for _ in range(self.num_modalities)]
        
        for m in range(self.num_modalities):
            factor_dims = tuple([self.num_states[f] for f in self.A_dependencies[m]])
            assert self.A[m].shape[2:] == factor_dims, f"Please input an `A_dependencies` whose {m}-th indices correspond to the hidden state factors that line up with lagging dimensions of A[{m}]..." 
            if self.pA != None:
                assert self.pA[m].shape[2:] == factor_dims, f"Please input an `A_dependencies` whose {m}-th indices correspond to the hidden state factors that line up with lagging dimensions of pA[{m}]..." 
            assert max(self.A_dependencies[m]) <= (self.num_factors - 1), f"Check modality {m} of `A_dependencies` - must be consistent with `num_states` and `num_factors`..."
           
        # Ensure consistency of B_dependencies with num_states and num_factors
        if B_dependencies is not None:
            self.B_dependencies
        else:
            num_factors = len(B)
            self.B_dependencies = [[f] for f in range(self.num_factors)] # defaults to having all factors depend only on themselves

        for f in range(self.num_factors):
            factor_dims = tuple([self.num_states[f] for f in self.B_dependencies[f]])
            assert self.B[f].shape[2:-1] == factor_dims, f"Please input a `B_dependencies` whose {f}-th indices pick out the hidden state factors that line up with the all-but-final lagging dimensions of B[{f}]..." 
            if self.pB != None:
                assert self.pB[f].shape[2:-1] == factor_dims, f"Please input a `B_dependencies` whose {f}-th indices pick out the hidden state factors that line up with the all-but-final lagging dimensions of pB[{f}]..." 
            assert max(self.B_dependencies[f]) <= (self.num_factors - 1), f"Check factor {f} of `B_dependencies` - must be consistent with `num_states` and `num_factors`..."

        batch_dim = (self.A[0].shape[0],)

        self.gamma = jnp.broadcast_to(gamma, batch_dim) 
        self.alpha = jnp.broadcast_to(alpha, batch_dim) 

        ### Static parameters ###

        self.num_iter = num_iter

        self.inference_algo = inference_algo

        # policy parameters
        self.policy_len = policy_len
        self.action_selection = action_selection
        self.sampling_mode = sampling_mode
        self.use_utility = use_utility
        self.use_states_info_gain = use_states_info_gain
        self.use_param_info_gain = use_param_info_gain

        # learning parameters
        self.learn_A = learn_A
        self.learn_B = learn_B
        self.learn_C = learn_C
        self.learn_D = learn_D
        self.learn_E = learn_E

        """ Determine number of observation modalities and their respective dimensions """
        self.num_obs = [self.A[m].shape[1] for m in range(len(self.A))]
        self.num_modalities = len(self.num_obs)

        # If no `num_controls` are given, then this is inferred from the shapes of the input B matrices
        self.num_controls = [self.B[f].shape[-1] for f in range(self.num_factors)]

        # Users have the option to make only certain factors controllable.
        # default behaviour is to make all hidden state factors controllable
        # (i.e. self.num_states == self.num_controls)
        # Users have the option to make only certain factors controllable.
        # default behaviour is to make all hidden state factors controllable, i.e. `self.num_factors == len(self.num_controls)`
        if control_fac_idx == None:
            self.control_fac_idx = [f for f in range(self.num_factors) if self.num_controls[f] > 1]
        else:
            assert max(control_fac_idx) <= (self.num_factors - 1), "Check control_fac_idx - must be consistent with `num_states` and `num_factors`..."
            self.control_fac_idx = control_fac_idx

            for factor_idx in self.control_fac_idx:
                assert self.num_controls[factor_idx] > 1, "Control factor (and B matrix) dimensions are not consistent with user-given control_fac_idx"

        if policies is not None:
            self.policies = policies
        else:
            self._construct_policies()

    def _construct_policies(self):
        
        self.policies =  control.construct_policies(
            self.num_states, self.num_controls, self.policy_len, self.control_fac_idx
        )

    @vmap
    def learning(self, beliefs, outcomes, actions, **kwargs):

        if self.learn_A:
            o_vec_seq = jtu.tree_map(lambda o, dim: nn.one_hot(o, dim), outcomes, self.num_obs)
            qA = learning.update_obs_likelihood_dirichlet(self.pA, self.A, o_vec_seq, beliefs, self.A_dependencies, lr=1.)
            E_qA = jtu.tree_map(lambda x: maths.dirichlet_expected_value(x), qA)
        if self.learn_B:
            actions_seq = [actions[...,i] for i in range(actions.shape[-1])] # as many elements as there are control factors, where each element is a jnp.ndarray of shape (n_timesteps, )
            actions_onehot = jtu.tree_map(lambda a, dim: nn.one_hot(a, dim, axis=-1), actions_seq, self.num_controls)
            qB = learning.update_state_likelihood_dirichlet(self.pB, self.B, beliefs, actions_onehot, self.B_dependencies)
            E_qB = jtu.tree_map(lambda x: maths.dirichlet_expected_value(x), qB)
        # if self.learn_C:
        #     self.qC = learning.update_C(self.C, *args, **kwargs)
        #     self.C = jtu.tree_map(lambda x: maths.dirichlet_expected_value(x), self.qC)
        # if self.learn_D:
        #     self.qD = learning.update_D(self.D, *args, **kwargs)
        #     self.D = jtu.tree_map(lambda x: maths.dirichlet_expected_value(x), self.qD)
        # if self.learn_E:
        #     self.qE = learning.update_E(self.E, *args, **kwargs)
        #     self.E = maths.dirichlet_expected_value(self.qE)

        # do stuff
        # variables = ...
        # parameters = ...
        # varibles = {'A': jnp.ones(5)}

        agent = tree_at(lambda x: (x.A, x.pA, x.B, x.pB), self, (E_qA, qA, E_qB, qB))

        return agent
    
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

        o_vec = [nn.one_hot(o, self.num_obs[m]) for m, o in enumerate(observations)]
        output = inference.update_posterior_states(
            self.A,
            self.B,
            o_vec,
            past_actions,
            prior=empirical_prior,
            qs_hist=qs_hist,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            num_iter=self.num_iter,
            method=self.inference_algo
        )

        # if ovf_smooth:
        #     output = inference.smoothing(output)

        return output

    @vmap
    def update_empirical_prior(self, action, qs):
        # return empirical_prior, and the history of posterior beliefs (filtering distributions) held about hidden states at times 1, 2 ... t

        qs_last = jtu.tree_map( lambda x: x[-1], qs)
        pred = control.compute_expected_state(qs_last, self.B, action, B_dependencies=self.B_dependencies)
        
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
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
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

    @vmap
    def sample_action(self, q_pi: jnp.ndarray, rng_key=None):
        """
        Sample or select a discrete action from the posterior over control states.
        
        Returns
        ----------
        action: 1D ``jax.numpy.ndarray``
            Vector containing the indices of the actions for each control factor
        """

        if (rng_key is None) and (self.action_selection == "stochastic"):
            raise ValueError("Please provide a random number generator key to sample actions stochastically")

        if self.sampling_mode == "marginal":
            action = control.sample_action(q_pi, self.policies, self.num_controls, self.action_selection, self.alpha, rng_key=rng_key)
        elif self.sampling_mode == "full":
            action = control.sample_policy(q_pi, self.policies, self.num_controls, self.action_selection, self.alpha, rng_key=rng_key)

        return action

    def sample_action_old(self, q_pi: jnp.ndarray, rng_key=None):
        """
        Sample or select a discrete action from the posterior over control states.
        
        Returns
        ----------
        action: 1D ``jax.numpy.ndarray``
            Vector containing the indices of the actions for each control factor
        """ 

        if (rng_key is None) and (self.action_selection == "stochastic"):
            raise ValueError("Please provide a random number generator key to sample actions stochastically")

        if self.sampling_mode == "marginal":
            sample_action = lambda x, alpha: control.sample_action(x, self.policies, self.num_controls, self.action_selection, alpha, rng_key=rng_key)
            action = vmap(sample_action)(q_pi, self.alpha)
        elif self.sampling_mode == "full":
            sample_policy = lambda x, alpha: control.sample_policy(x, self.policies, self.num_controls, self.action_selection, alpha, rng_key=rng_key)
            action = vmap(sample_policy)(q_pi, self.alpha)

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