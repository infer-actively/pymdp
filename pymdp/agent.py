#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class implementation in Jax

__author__: Conor Heins, Dimitrije Markovic, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""
import jax
import math as pymath
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn, vmap, random
from pymdp import inference, control, learning, utils, maths
from pymdp.distribution import Distribution, get_dependencies
from equinox import Module, field, tree_at

from typing import List, Optional, Union
from jaxtyping import Array
from functools import partial

class Agent(Module):
    """
    The Agent class, the highest-level API that wraps together processes for action, perception, and learning under active inference.

    The basic usage is as follows:

    >>> my_agent = Agent(A = A, B = C, <more_params>)
    >>> observation = env.step(initial_action)
    >>> qs = my_agent.infer_states(observation)
    >>> q_pi, G = my_agent.infer_policies(qs)
    >>> next_action = my_agent.sample_action()
    >>> next_observation = env.step(next_action)

    This represents one timestep of an active inference process. Wrapping this step in a loop with an ``Env()`` class that returns
    observations and takes actions as inputs, would entail a dynamic agent-environment interaction.
    """

    A: List[Array]
    B: List[Array]
    C: List[Array]
    D: List[Array]
    E: Array
    pA: List[Array]
    pB: List[Array]
    gamma: Array
    alpha: Array

    # matrix of all possible policies (each row is a policy of shape (num_controls[0], num_controls[1], ..., num_controls[num_control_factors-1])
    policies: Array

    # threshold for inductive inference (the threshold for pruning transitions that are below a certain probability)
    inductive_threshold: Array
    # epsilon for inductive inference (trade-off/weight for how much inductive value contributes to EFE of policies)
    inductive_epsilon: Array
    # H vectors (one per hidden state factor) used for inductive inference -- these encode goal states or constraints
    H: List[Array]
    # I matrices (one per hidden state factor) used for inductive inference -- these encode the 'reachability' matrices of goal states encoded in `self.H`
    I: List[Array]
    # static parameters not leaves of the PyTree
    A_dependencies: Optional[List] = field(static=True)
    B_dependencies: Optional[List] = field(static=True)
    B_action_dependencies: Optional[List] = field(static=True)
    # mapping from multi action dependencies to flat action dependencies for each B
    action_maps: List[dict] = field(static=True)
    batch_size: int = field(static=True)
    num_iter: int = field(static=True)
    num_obs: List[int] = field(static=True)
    num_modalities: int = field(static=True)
    num_states: List[int] = field(static=True)
    num_factors: int = field(static=True)
    num_controls: List[int] = field(static=True)
    # Used to store original action dimensions in case there are multiple action dependencies per state
    num_controls_multi: List[int] = field(static=True)
    control_fac_idx: Optional[List[int]] = field(static=True)
    # depth of planning during roll-outs (i.e. number of timesteps to look ahead when computing expected free energy of policies)
    policy_len: int = field(static=True)
    # depth of inductive inference (i.e. number of future timesteps to use when computing inductive `I` matrix)
    inductive_depth: int = field(static=True)
    # flag for whether to use expected utility ("reward" or "preference satisfaction") when computing expected free energy
    use_utility: bool = field(static=True)
    # flag for whether to use state information gain ("salience") when computing expected free energy
    use_states_info_gain: bool = field(static=True)
    # flag for whether to use parameter information gain ("novelty") when computing expected free energy
    use_param_info_gain: bool = field(static=True)
    # flag for whether to use inductive inference ("intentional inference") when computing expected free energy
    use_inductive: bool = field(static=True)
    onehot_obs: bool = field(static=True)
    # determinstic or stochastic action selection
    action_selection: str = field(static=True)
    # whether to sample from full posterior over policies ("full") or from marginal posterior over actions ("marginal")
    sampling_mode: str = field(static=True)
    # fpi, vmp, mmp, ovf
    inference_algo: str = field(static=True)

    learn_A: bool = field(static=True)
    learn_B: bool = field(static=True)
    learn_C: bool = field(static=True)
    learn_D: bool = field(static=True)
    learn_E: bool = field(static=True)

    def __init__(
        self,
        A: Union[List[Array], List[Distribution]],
        B: Union[List[Array], List[Distribution]],
        C: Optional[List[Array]] = None,
        D: Optional[List[Array]] = None,
        E: Optional[Array] = None,
        pA=None,
        pB=None,
        H=None,
        I=None,
        A_dependencies=None,
        B_dependencies=None,
        B_action_dependencies=None,
        num_controls=None,
        control_fac_idx=None,
        policy_len=1,
        policies=None,
        gamma=1.0,
        alpha=1.0,
        inductive_depth=1,
        inductive_threshold=0.1,
        inductive_epsilon=1e-3,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=False,
        use_inductive=False,
        onehot_obs=False,
        action_selection="deterministic",
        sampling_mode="full",
        inference_algo="fpi",
        num_iter=16,
        apply_batch=True,
        learn_A=True,
        learn_B=True,
        learn_C=False,
        learn_D=True,
        learn_E=False,
    ):
        if B_action_dependencies is not None:
            assert num_controls is not None, "Please specify num_controls for complex action dependencies"

        # extract high level variables
        self.num_modalities = len(A)
        self.num_factors = len(B)
        self.num_controls = num_controls
        self.num_controls_multi = num_controls

        # extract dependencies for A and B matrices
        (
            self.A_dependencies,
            self.B_dependencies,
            self.B_action_dependencies,
        ) = self._construct_dependencies(A_dependencies, B_dependencies, B_action_dependencies, A, B)

        # extract A, B, C and D tensors from optional Distributions
        A = [jnp.array(a.data) if isinstance(a, Distribution) else a for a in A]
        B = [jnp.array(b.data) if isinstance(b, Distribution) else b for b in B]
        if C is not None:
            C = [jnp.array(c.data) if isinstance(c, Distribution) else c for c in C]
        if D is not None:
            D = [jnp.array(d.data) if isinstance(d, Distribution) else d for d in D]
        if E is not None:
            E = jnp.array(E.data) if isinstance(E, Distribution) else E
        if H is not None:
            H = [jnp.array(h.data) if isinstance(h, Distribution) else h for h in H]

        self.batch_size = A[0].shape[0] if not apply_batch else 1

        # flatten B action dims for multiple action dependencies
        self.action_maps = None
        self.num_controls_multi = num_controls
        if (
            B_action_dependencies is not None
        ):  # note, this only works when B_action_dependencies is not the trivial case of [[0], [1], ...., [num_factors-1]]
            policies_multi = control.construct_policies(
                self.num_controls_multi,
                self.num_controls_multi,
                policy_len,
                control_fac_idx,
            )
            B, pB, self.action_maps = self._flatten_B_action_dims(B, pB, self.B_action_dependencies)
            policies = self._construct_flattend_policies(policies_multi, self.action_maps)
            self.sampling_mode = "full"

        # extract shapes from A and B
        batch_dim_fn = lambda x: x.shape[0] if apply_batch else x.shape[1]
        self.num_states = jtu.tree_map(batch_dim_fn, B)
        self.num_obs = jtu.tree_map(batch_dim_fn, A)
        self.num_controls = [B[f].shape[-1] for f in range(self.num_factors)]

        # static parameters
        self.num_iter = num_iter
        self.inference_algo = inference_algo
        self.inductive_depth = inductive_depth

        # policy parameters
        self.policy_len = policy_len
        self.action_selection = action_selection
        self.sampling_mode = sampling_mode
        self.use_utility = use_utility
        self.use_states_info_gain = use_states_info_gain
        self.use_param_info_gain = use_param_info_gain
        self.use_inductive = use_inductive

        # learning parameters
        self.learn_A = learn_A
        self.learn_B = learn_B
        self.learn_C = learn_C
        self.learn_D = learn_D
        self.learn_E = learn_E

        # construct control factor indices
        if control_fac_idx == None:
            self.control_fac_idx = [f for f in range(self.num_factors) if self.num_controls[f] > 1]
        else:
            msg = "Check control_fac_idx - must be consistent with `num_states` and `num_factors`..."
            assert max(control_fac_idx) <= (self.num_factors - 1), msg
            self.control_fac_idx = control_fac_idx

        # construct policies
        if policies is None:
            self.policies = control.construct_policies(
                self.num_states,
                self.num_controls,
                self.policy_len,
                self.control_fac_idx,
            )
        else:
            self.policies = policies

        # setup pytree leaves A, B, C, D, E, pA, pB, H, I
        if apply_batch:
            A = jtu.tree_map(lambda x: jnp.broadcast_to(x, (self.batch_size,) + x.shape), A)
            B = jtu.tree_map(lambda x: jnp.broadcast_to(x, (self.batch_size,) + x.shape), B)

        if pA is not None and apply_batch:
            pA = jtu.tree_map(lambda x: jnp.broadcast_to(x, (self.batch_size,) + x.shape), pA)

        if pB is not None and apply_batch:
            pB = jtu.tree_map(lambda x: jnp.broadcast_to(x, (self.batch_size,) + x.shape), pB)

        if C is None:
            C = [jnp.ones((self.batch_size, self.num_obs[m])) / self.num_obs[m] for m in range(self.num_modalities)]
        elif apply_batch:
            C = jtu.tree_map(lambda x: jnp.broadcast_to(x, (self.batch_size,) + x.shape), C)

        if D is None:
            D = [jnp.ones((self.batch_size, self.num_states[f])) / self.num_states[f] for f in range(self.num_factors)]
        elif apply_batch:
            D = jtu.tree_map(lambda x: jnp.broadcast_to(x, (self.batch_size,) + x.shape), D)

        if E is None:
            E = jnp.ones((self.batch_size, len(self.policies))) / len(self.policies)
        elif apply_batch:
            E = jnp.broadcast_to(E, (self.batch_size,) + E.shape)

        if H is not None and apply_batch:
            H = jtu.tree_map(
                lambda x: jnp.broadcast_to(x, (self.batch_size,) + x.shape),
                H,
            )

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.H = H
        self.I = I
        self.pA = pA
        self.pB = pB

        self.gamma = jnp.broadcast_to(gamma, (self.batch_size,))
        self.alpha = jnp.broadcast_to(alpha, (self.batch_size,))

        self.inductive_threshold = jnp.broadcast_to(inductive_threshold, (self.batch_size,))
        self.inductive_epsilon = jnp.broadcast_to(inductive_epsilon, (self.batch_size,))

        if self.use_inductive and H is not None:
            I = vmap(
                partial(
                    control.generate_I_matrix,
                    depth=self.inductive_depth,
                )
            )(H, B, self.inductive_threshold)
        elif self.use_inductive and I is not None:
            I = I
        else:
            I = jtu.tree_map(lambda x: jnp.expand_dims(jnp.zeros_like(x), 1), D)

        self.onehot_obs = onehot_obs

        # validate model
        self._validate()

    @property
    def unique_multiactions(self):
        size = pymath.prod(self.num_controls)
        return jnp.unique(self.policies[:, 0], axis=0, size=size, fill_value=-1)

    def infer_parameters(self, beliefs_A, outcomes, actions, beliefs_B=None, lr_pA=1., lr_pB=1., **kwargs):
        agent = self
        beliefs_B = beliefs_A if beliefs_B is None else beliefs_B
        if self.inference_algo == 'ovf':
            smoothed_marginals_and_joints = vmap(inference.smoothing_ovf)(beliefs_A, self.B, actions)
            marginal_beliefs = smoothed_marginals_and_joints[0]
            joint_beliefs = smoothed_marginals_and_joints[1]
        else:
            marginal_beliefs = beliefs_A
            if self.learn_B:
                nf = len(beliefs_B)
                joint_fn = lambda f: [beliefs_B[f][:, 1:]] + [beliefs_B[f_idx][:, :-1] for f_idx in self.B_dependencies[f]]
                joint_beliefs = jtu.tree_map(joint_fn, list(range(nf)))

        if self.learn_A:
            update_A = partial(
                learning.update_obs_likelihood_dirichlet,
                A_dependencies=self.A_dependencies,
                num_obs=self.num_obs,
                onehot_obs=self.onehot_obs,
            )
            
            lr = jnp.broadcast_to(lr_pA, (self.batch_size,))
            qA, E_qA = vmap(update_A)(
                self.pA,
                self.A,
                outcomes,
                marginal_beliefs,
                lr=lr,
            )
            
            agent = tree_at(lambda x: (x.A, x.pA), agent, (E_qA, qA))
            
        if self.learn_B:
            assert beliefs_B[0].shape[1] == actions.shape[1] + 1
            update_B = partial(
                learning.update_state_transition_dirichlet,
                num_controls=self.num_controls
            )

            lr = jnp.broadcast_to(lr_pB, (self.batch_size,))
            qB, E_qB = vmap(update_B)(
                self.pB,
                joint_beliefs,
                actions,
                lr=lr
            )
            
            # if you have updated your beliefs about transitions, you need to re-compute the I matrix used for inductive inferenece
            if self.use_inductive and self.H is not None:
                I_updated = vmap(control.generate_I_matrix)(self.H, E_qB, self.inductive_threshold, self.inductive_depth)
                agent = tree_at(lambda x: (x.B, x.pB, x.I), agent, (E_qB, qB, I_updated))
            else:
                agent = tree_at(lambda x: (x.B, x.pB), agent, (E_qB, qB))

        return agent


    def infer_states(self, observations, empirical_prior, *, past_actions=None, qs_hist=None, mask=None):
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

        # TODO: infer this from shapes
        if not self.onehot_obs:
            o_vec = [nn.one_hot(o, self.num_obs[m]) for m, o in enumerate(observations)]
        else:
            o_vec = observations

        A = self.A
        if mask is not None:
            for i, m in enumerate(mask):
                o_vec[i] = m * o_vec[i] + (1 - m) * jnp.ones_like(o_vec[i]) / self.num_obs[i]
                A[i] = m * A[i] + (1 - m) * jnp.ones_like(A[i]) / self.num_obs[i]

        infer_states = partial(
            inference.update_posterior_states,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            num_iter=self.num_iter,
            method=self.inference_algo,
        )
        
        output = vmap(infer_states)(
            A,
            self.B,
            o_vec,
            past_actions,
            prior=empirical_prior,
            qs_hist=qs_hist
        )

        return output

    def update_empirical_prior(self, action, qs):
        # return empirical_prior, and the history of posterior beliefs (filtering distributions) held about hidden states at times 1, 2 ... t

        # this computation of the predictive prior is correct only for fully factorised Bs.
        if self.inference_algo in ['mmp', 'vmp']:
            # in the case of the 'mmp' or 'vmp' we have to use D as prior parameter for infer states
            pred = self.D
        else:
            qs_last = jtu.tree_map( lambda x: x[:, -1], qs)
            propagate_beliefs = partial(control.compute_expected_state, B_dependencies=self.B_dependencies)
            pred = vmap(propagate_beliefs)(qs_last, self.B, action)
        
        return (pred, qs)

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

        latest_belief = jtu.tree_map(lambda x: x[:, -1], qs) # only get the posterior belief held at the current timepoint
        infer_policies = partial(
            control.update_posterior_policies_inductive,
            self.policies,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            use_utility=self.use_utility,
            use_states_info_gain=self.use_states_info_gain,
            use_param_info_gain=self.use_param_info_gain,
            use_inductive=self.use_inductive
        )

        q_pi, G = vmap(infer_policies)(
            latest_belief, 
            self.A,
            self.B,
            self.C,
            self.E,
            self.pA,
            self.pB,
            I = self.I,
            gamma=self.gamma,
            inductive_epsilon=self.inductive_epsilon
        )

        return q_pi, G
    
    def multiaction_probabilities(self, q_pi: Array):
        """
        Compute probabilities of unique multi-actions from the posterior over policies.

        Parameters
        ----------
        q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.

        Returns
        ----------
        multi-action: 1D ``jax.numpy.ndarray``
            Vector containing probabilities of possible multi-actions for different factors
        """

        if self.sampling_mode == "marginal":
            get_marginals = partial(control.get_marginals, policies=self.policies, num_controls=self.num_controls)
            marginals = get_marginals(q_pi)
            outer = lambda a, b: jnp.outer(a, b).reshape(-1)
            marginals = jtu.tree_reduce(outer, marginals)

        elif self.sampling_mode == "full":
            locs = jnp.all(
                self.policies[:, 0] == jnp.expand_dims(self.unique_multiactions, -2),
                -1,
            )
            get_marginals = lambda x: jnp.where(locs, x, 0.).sum(-1)
            marginals = vmap(get_marginals)(q_pi)

        return marginals

    def sample_action(self, q_pi: Array, rng_key=None):
        """
        Sample or select a discrete action from the posterior over control states.
        
        Returns
        ----------
        action: 1D ``jax.numpy.ndarray``
            Vector containing the indices of the actions for each control factor
        action_probs: 2D ``jax.numpy.ndarray``
            Array of action probabilities
        """
        if (rng_key is None) and (self.action_selection == "stochastic"):
            raise ValueError("Please provide a random number generator key to sample actions stochastically")

        if self.sampling_mode == "marginal":
            sample_action = partial(control.sample_action, self.policies, self.num_controls, action_selection=self.action_selection)
            action = vmap(sample_action)(q_pi, alpha=self.alpha, rng_key=rng_key)
        elif self.sampling_mode == "full":
            sample_policy = partial(control.sample_policy, self.policies, action_selection=self.action_selection)
            action = vmap(sample_policy)(q_pi, alpha=self.alpha, rng_key=rng_key)

        return action
    
    def decode_multi_actions(self, action):
        """Decode flattened actions to multiple actions"""
        if self.action_maps is None:
            return action

        action_multi = jnp.zeros((self.batch_size, len(self.num_controls_multi))).astype(action.dtype)
        for f, action_map in enumerate(self.action_maps):
            if action_map["multi_dependency"] == []:
                continue

            action_multi_f = utils.index_to_combination(action[..., f], action_map["multi_dims"])
            action_multi = action_multi.at[..., action_map["multi_dependency"]].set(action_multi_f)
        return action_multi

    def encode_multi_actions(self, action_multi):
        """Encode multiple actions to flattened actions"""
        if self.action_maps is None:
            return action_multi

        action = jnp.zeros((self.batch_size, len(self.num_controls))).astype(action_multi.dtype)
        for f, action_map in enumerate(self.action_maps):
            if action_map["multi_dependency"] == []:
                action = action.at[..., f].set(jnp.zeros_like(action_multi[..., 0]))
                continue

            action_f = utils.get_combination_index(
                action_multi[..., action_map["multi_dependency"]],
                action_map["multi_dims"],
            )
            action = action.at[..., f].set(action_f)
        return action

    def _construct_dependencies(self, A_dependencies, B_dependencies, B_action_dependencies, A, B):
        if A_dependencies is not None:
            A_dependencies = A_dependencies
        elif isinstance(A[0], Distribution) and isinstance(B[0], Distribution):
            A_dependencies, _ = get_dependencies(A, B)
        else:
            A_dependencies = [list(range(self.num_factors)) for _ in range(self.num_modalities)]

        if B_dependencies is not None:
            B_dependencies = B_dependencies
        elif isinstance(A[0], Distribution) and isinstance(B[0], Distribution):
            _, B_dependencies = get_dependencies(A, B)
        else:
            B_dependencies = [[f] for f in range(self.num_factors)]

        """TODO: check B action shape"""
        if B_action_dependencies is not None:
            B_action_dependencies = B_action_dependencies
        else:
            B_action_dependencies = [[f] for f in range(self.num_factors)]
        return A_dependencies, B_dependencies, B_action_dependencies

    def _flatten_B_action_dims(self, B, pB, B_action_dependencies):
        assert hasattr(B[0], "shape"), "Elements of B must be tensors and have attribute shape"
        action_maps = []  # mapping from multi action dependencies to flat action dependencies for each B
        B_flat = []
        pB_flat = []
        for i, (B_f, action_dependency) in enumerate(zip(B, B_action_dependencies)):
            if action_dependency == []:
                B_flat.append(jnp.expand_dims(B_f, axis=-1))
                if pB is not None:
                    pB_flat.append(jnp.expand_dims(pB[i], axis=-1))
                action_maps.append(
                    {"multi_dependency": [], "multi_dims": [], "flat_dependency": [i], "flat_dims": [1]}
                )
                continue

            dims = [self.num_controls_multi[d] for d in action_dependency]
            target_shape = list(B_f.shape)[: -len(action_dependency)] + [pymath.prod(dims)]
            B_flat.append(B_f.reshape(target_shape))
            if pB is not None:
                pB_flat.append(pB[i].reshape(target_shape))
            action_maps.append(
                {
                    "multi_dependency": action_dependency,
                    "multi_dims": dims,
                    "flat_dependency": [i],
                    "flat_dims": [pymath.prod(dims)],
                }
            )
        if pB is None:
            pB_flat = None
        return B_flat, pB_flat, action_maps

    def _construct_flattend_policies(self, policies, action_maps):
        policies_flat = []
        for action_map in action_maps:
            if action_map["multi_dependency"] == []:
                policies_flat.append(jnp.zeros_like(policies[..., 0]))
                continue

            policies_flat.append(
                utils.get_combination_index(
                    policies[..., action_map["multi_dependency"]],
                    action_map["multi_dims"],
                )
            )
        policies_flat = jnp.stack(policies_flat, axis=-1)
        return policies_flat

    def _get_default_params(self):
        method = self.inference_algo
        default_params = None
        if method == "VANILLA":
            default_params = {"num_iter": 8, "dF": 1.0, "dF_tol": 0.001}
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

    def _validate(self):
        for m in range(self.num_modalities):
            factor_dims = tuple([self.num_states[f] for f in self.A_dependencies[m]])
            assert (
                self.A[m].shape[2:] == factor_dims
            ), f"Please input an `A_dependencies` whose {m}-th indices correspond to the hidden state factors that line up with lagging dimensions of A[{m}]..."
            if self.pA != None:
                assert (
                    self.pA[m].shape[2:] == factor_dims if self.pA[m] is not None else True,
                ), f"Please input an `A_dependencies` whose {m}-th indices correspond to the hidden state factors that line up with lagging dimensions of pA[{m}]..."
            assert max(self.A_dependencies[m]) <= (
                self.num_factors - 1
            ), f"Check modality {m} of `A_dependencies` - must be consistent with `num_states` and `num_factors`..."

        for f in range(self.num_factors):
            factor_dims = tuple([self.num_states[f] for f in self.B_dependencies[f]])
            assert (
                self.B[f].shape[2:-1] == factor_dims
            ), f"Please input a `B_dependencies` whose {f}-th indices pick out the hidden state factors that line up with the all-but-final lagging dimensions of B[{f}]..."
            if self.pB != None:
                assert (
                    self.pB[f].shape[2:-1] == factor_dims
                ), f"Please input a `B_dependencies` whose {f}-th indices pick out the hidden state factors that line up with the all-but-final lagging dimensions of pB[{f}]..."
            assert max(self.B_dependencies[f]) <= (
                self.num_factors - 1
            ), f"Check factor {f} of `B_dependencies` - must be consistent with `num_states` and `num_factors`..."

        for factor_idx in self.control_fac_idx:
            assert (
                self.num_controls[factor_idx] > 1
            ), "Control factor (and B matrix) dimensions are not consistent with user-given control_fac_idx"