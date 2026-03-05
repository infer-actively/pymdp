#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Agent API for Active Inference with the modern JAX backend."""
import math as pymath
import warnings
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn, vmap
from pymdp import inference, control, learning, utils
from pymdp.distribution import Distribution, get_dependencies
from equinox import Module, field, tree_at

from typing import Any, Callable, Optional, Sequence, Union
from jaxtyping import Array
from functools import partial
from jax import lax

class Agent(Module):
    """
    The Agent class, the highest-level API that wraps together processes for action, perception, and learning under active inference.

    Examples
    --------
    A single timestep of active inference:

        from jax import random as jr

        my_agent = Agent(A=A, B=B, C=C, <more_params>)
        observation = env.step(initial_action)
        qs = my_agent.infer_states(observation, empirical_prior=my_agent.D)
        q_pi, G = my_agent.infer_policies(qs)
        keys = jr.split(rng_key, my_agent.batch_size + 1)
        next_action = my_agent.sample_action(q_pi, rng_key=keys[1:])
        next_observation = env.step(next_action)

    This represents one timestep of an active inference process. Wrapping this step in a loop with an `Env()` class that returns
    observations and takes actions as inputs, would entail a dynamic agent-environment interaction.

    Observation Formats
    -------------------
    Observations can be provided in two formats:

    1. **Discrete observations** (default, categorical_obs=False):
       Each `observations[m]` is an integer observation index for modality `m`.
       These are converted to one-hot vectors internally.

    2. **Categorical observations** (categorical_obs=True):
       Each `observations[m]` is a probability vector over observations for
       modality `m`.

    Advanced preprocessing
    ----------------------
    You can override default preprocessing with `preprocess_fn` (set on the
    agent or per `infer_states` call). If provided, this function should
    return categorical observations and takes precedence over default
    discrete/categorical handling.
    """

    A: list[Array]
    B: list[Array]
    C: list[Array]
    D: list[Array]
    E: Array
    pA: list[Array]
    pB: list[Array]
    gamma: Array
    alpha: Array

    # matrix of all possible policies (each row is a policy of shape (num_controls[0], num_controls[1], ..., num_controls[num_control_factors-1])
    policies: control.Policies = field(static=True)

    # threshold for inductive inference (the threshold for pruning transitions that are below a certain probability)
    inductive_threshold: Array
    # epsilon for inductive inference (trade-off/weight for how much inductive value contributes to EFE of policies)
    inductive_epsilon: Array
    # H vectors (one per hidden state factor) used for inductive inference -- these encode goal states or constraints
    H: list[Array]
    # I matrices (one per hidden state factor) used for inductive inference -- these encode the 'reachability' matrices of goal states encoded in `self.H`
    I: list[Array]
    # static parameters not leaves of the PyTree
    A_dependencies: Optional[list[list[int]]] = field(static=True)
    B_dependencies: Optional[list[list[int]]] = field(static=True)
    B_action_dependencies: Optional[list[list[int]]] = field(static=True)
    # mapping from multi action dependencies to flat action dependencies for each B
    action_maps: list[dict] = field(static=True)
    batch_size: int = field(static=True)
    num_iter: int = field(static=True)
    num_obs: list[int] = field(static=True)
    num_modalities: int = field(static=True)
    num_states: list[int] = field(static=True)
    num_factors: int = field(static=True)
    num_controls: list[int] = field(static=True)
    # Used to store original action dimensions in case there are multiple action dependencies per state
    num_controls_multi: list[int] = field(static=True)
    control_fac_idx: Optional[list[int]] = field(static=True)
    # depth of planning during roll-outs (i.e. number of timesteps to look ahead when computing expected free energy of policies)
    policy_len: int = field(static=True)
    # number of past timesteps (including current) to use for sequence inference (mmp, vmp, exact)
    inference_horizon: Optional[int] = field(static=True)
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
    categorical_obs: bool = field(static=True)
    preprocess_fn: Optional[Callable] = field(static=True)
    # determinstic or stochastic action selection
    action_selection: str = field(static=True)
    # whether to sample from full posterior over policies ("full") or from marginal posterior over actions ("marginal")
    sampling_mode: str = field(static=True)
    # fpi, vmp, mmp, ovf, exact
    inference_algo: str = field(static=True)
    # whether to perform learning online or offline (options: "online", "offline")
    learning_mode: str = field(static=True, default="online")

    learn_A: bool = field(static=True)
    learn_B: bool = field(static=True)
    learn_C: bool = field(static=True)
    learn_D: bool = field(static=True)
    learn_E: bool = field(static=True)

    def __init__(
        self,
        A: Union[list[Array], list[Distribution]],
        B: Union[list[Array], list[Distribution]],
        C: Optional[list[Array]] = None,
        D: Optional[list[Array]] = None,
        E: Optional[Array] = None,
        pA: Optional[list[Array]] = None,
        pB: Optional[list[Array]] = None,
        H: Optional[list[Array]] = None,
        I: Optional[list[Array]] = None,
        A_dependencies: Optional[list[list[int]]] = None,
        B_dependencies: Optional[list[list[int]]] = None,
        B_action_dependencies: Optional[list[list[int]]] = None,
        num_controls: Optional[list[int]] = None,
        control_fac_idx: Optional[list[int]] = None,
        policy_len: int = 1,
        policies: Optional[Union[Array, control.Policies]] = None,
        gamma: float | Array = 1.0,
        alpha: float | Array = 1.0,
        inductive_depth: int = 1,
        inductive_threshold: float | Array = 0.1,
        inductive_epsilon: float | Array = 1e-3,
        use_utility: bool = True,
        use_states_info_gain: bool = True,
        use_param_info_gain: bool = False,
        use_inductive: bool = False,
        categorical_obs: bool = False,
        preprocess_fn: Optional[Callable] = None,
        action_selection: str = "deterministic",
        sampling_mode: str = "full",
        inference_algo: str = "fpi",
        inference_horizon: Optional[int] = None,
        num_iter: int = 16,
        batch_size: int = 1,
        learning_mode: str = "online", # TODO: or should this be an argument to `self.infer_parameters()` or even `env/rollout.py:rollout()`
        learn_A: bool = False,
        learn_B: bool = False,
        learn_C: bool = False,
        learn_D: bool = False,
        learn_E: bool = False,
    ) -> None:
        if B_action_dependencies is not None:
            assert num_controls is not None, "Please specify num_controls if you're also using complex action dependencies"

        if learn_A:
            assert pA is not None, "pA is required for A learning"

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

        if pA is not None:
            pA = [jnp.array(pa) for pa in pA]

        if pB is not None:
            pB = [jnp.array(pb) for pb in pB]

        self.batch_size = batch_size

        # flatten B action dims for multiple action dependencies
        self.action_maps = None
        if (
            policies is None and B_action_dependencies is not None
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
        self.num_states = self._get_num_states_from_B(B, self.B_dependencies)
        self.num_controls = [b_f.shape[-1] for b_f in B] # dimensions of control states for each hidden state factor

        # check that batch_size is consistent with shapes of given A and B
        for m, a_m in enumerate(A):
            a_m_state_factors = tuple([self.num_states[f] for f in self.A_dependencies[m]])
            if a_m.ndim > (len(a_m_state_factors) + 1): # this indicates there's a leading batch dimension
                if a_m.shape[0] == 1 and batch_size > 1:
                    A[m] = jnp.broadcast_to(a_m, (batch_size,) + a_m.shape[1:])
                    if pA is not None:
                        pA[m] = jnp.broadcast_to(pA[m], (batch_size,) + a_m.shape[1:])
                    if C is not None:
                        C[m] = jnp.broadcast_to(C[m], (batch_size,) + C[m].shape[1:])
                elif a_m.shape[0] != batch_size:
                    raise ValueError(
                        f"Batch size {batch_size} does not match the first dimension of A[{m}] with shape {a_m.shape}"
                    )
            elif a_m.ndim == (len(a_m_state_factors) + 1):  # this indicates no leading batch dimension
                A[m] = jnp.broadcast_to(a_m, (batch_size,) + a_m.shape)
                if pA is not None:
                    pA[m] = jnp.broadcast_to(pA[m], (batch_size,) + a_m.shape)
                if C is not None:
                    C[m] = jnp.broadcast_to(C[m], (batch_size,) + C[m].shape)
        
        for f, b_f in enumerate(B):
            b_f_state_factors = tuple([self.num_states[f] for f in self.B_dependencies[f]])
            if b_f.ndim > (len(b_f_state_factors) + 2):  # this indicates there's a leading batch dimension
                if b_f.shape[0] == 1 and batch_size > 1:
                    B[f] = jnp.broadcast_to(b_f, (batch_size,) + b_f.shape[1:])
                    if pB is not None:
                        pB[f] = jnp.broadcast_to(pB[f], (batch_size,) + b_f.shape[1:])
                    if D is not None:
                        D[f] = jnp.broadcast_to(D[f], (batch_size,) + D[f].shape[1:])
                elif b_f.shape[0] != batch_size:
                    raise ValueError(
                        f"Batch size {batch_size} does not match the first dimension of B[{f}] with shape {b_f.shape}"
                    )
            elif b_f.ndim == (len(b_f_state_factors) + 2):  # this indicates no leading batch dimension
                B[f] = jnp.broadcast_to(b_f, (batch_size,) + b_f.shape)
                if pB is not None:
                    pB[f] = jnp.broadcast_to(pB[f], (batch_size,) + b_f.shape)
                if D is not None:
                    D[f] = jnp.broadcast_to(D[f], (batch_size,) + D[f].shape)

        # now that shapes of A/B have been made consistent and have (batch_size,)-sized leading dimension applied, we can infer num_obs from the first dimension of A
        self.num_obs = [a_m.shape[1] for a_m in A] # dimensions of observations for each modality

        # static parameters
        self.num_iter = num_iter
        self.inference_algo = inference_algo.lower() if isinstance(inference_algo, str) else inference_algo
        self.inference_horizon = inference_horizon
        self.inductive_depth = inductive_depth

        if self.inference_horizon is not None and self.inference_horizon < 1:
            raise ValueError("`inference_horizon` must be >= 1 when provided")

        # policy parameters
        self.policy_len = policy_len
        self.action_selection = action_selection
        self.sampling_mode = sampling_mode
        self.use_utility = use_utility
        self.use_states_info_gain = use_states_info_gain
        self.use_param_info_gain = use_param_info_gain
        self.use_inductive = use_inductive

        # learning parameters
        self.learning_mode = learning_mode
        self.learn_A = learn_A
        self.learn_B = learn_B
        self.learn_C = learn_C
        self.learn_D = learn_D
        self.learn_E = learn_E

        # construct control factor indices
        if control_fac_idx is None:
            self.control_fac_idx = [f for f in range(self.num_factors) if self.num_controls[f] > 1]
        else:
            msg = "Check control_fac_idx - must be consistent with `num_states` and `num_factors`..."
            assert max(control_fac_idx) <= (self.num_factors - 1), msg
            self.control_fac_idx = control_fac_idx

        # construct policies
        if policies is None:
            policies_array = control.construct_policies(
                self.num_states,
                self.num_controls,
                self.policy_len,
                self.control_fac_idx,
            )
            self.policies = control.Policies(policies_array)
        else:
            if not isinstance(policies, control.Policies):
                self.policies = control.Policies(jnp.array(policies))
            else:
                self.policies = policies

        if C is None:
            C = [jnp.ones((self.batch_size, self.num_obs[m])) / self.num_obs[m] for m in range(self.num_modalities)]
    
        if D is None:
            D = [jnp.ones((self.batch_size, self.num_states[f])) / self.num_states[f] for f in range(self.num_factors)]

        if E is None:
            E = jnp.ones((self.batch_size, self.policies.num_policies)) / self.policies.num_policies
        else:
            if E.ndim > 1:
                if E.shape[0] == 1 and batch_size > 1:
                    E = jnp.broadcast_to(E, (batch_size,) + E.shape[1:])
                elif E.shape[0] != batch_size:
                    raise ValueError(
                        f"Batch size {batch_size} does not match the first dimension of E with shape {E.shape}"
                    )
            elif E.ndim == 1:
                E = jnp.broadcast_to(E, (self.batch_size,) + E.shape)

        if H is not None:
            for f, h_f in enumerate(H):
                if h_f.ndim > 1:
                    if h_f.shape[0] == 1 and batch_size > 1:
                        H[f] = jnp.broadcast_to(h_f, (batch_size,) + h_f.shape[1:])
                    elif h_f.shape[0] != batch_size:
                        raise ValueError(
                            f"Batch size {batch_size} does not match the first dimension of H[{f}] with shape {h_f.shape}"
                        )
                elif h_f.ndim == 1:
                    H[f] = jnp.broadcast_to(h_f, (self.batch_size,) + h_f.shape)

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.H = H
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
        self.I = I

        self.categorical_obs = categorical_obs
        self.preprocess_fn = preprocess_fn
        if (self.preprocess_fn is not None) and (self.categorical_obs is False):
            warnings.warn(
                "preprocess_fn is set while categorical_obs=False. If your preprocess_fn returns "
                "categorical distributions, set categorical_obs=True so that learning/planning can "
                "interpret observations correctly.",
                UserWarning,
                stacklevel=2,
            )

        # validate model
        self._validate()

    @property
    def unique_multiactions(self) -> Array:
        size = pymath.prod(self.num_controls)
        return jnp.unique(self.policies.policy_arr[:, 0], axis=0, size=size, fill_value=-1)

    def _get_num_states_from_B(self, B: list[Array], B_dependencies: list[list[int]]) -> list[int]:
        """ Use the shapes of B and the B_dependencies to determine the number of states for each factor."""
  
        num_states = []
        for (f, B_deps_f) in enumerate(B_dependencies):  
            if f in B_deps_f:
                self_factor_index = B_deps_f.index(f)
            else:
                raise ValueError(f"num_states cannot be inferred from B, B_dependencies if the dynamics of hidden state {f} is not conditioned on itself")
            
            num_states_f = B[f].shape[-(len(B_deps_f) + 1 - self_factor_index)]
            num_states.append(num_states_f)

        return num_states

    def infer_parameters(
        self,
        beliefs_A: list[Array],
        observations: list[Array] | None = None,
        actions: Array | None = None,
        beliefs_B: list[Array] | None = None,
        lr_pA: float = 1.0,
        lr_pB: float = 1.0,
        **kwargs: Any,
    ) -> "Agent":
        """Update Dirichlet parameters for `A` and/or `B` models from data.

        Parameters
        ----------
        beliefs_A: list[Array]
            Marginal state beliefs used when updating the observation model
            parameters.
        observations: list[Array]
            Observation histories for each modality.
        actions: Array or None
            Action history aligned to time. For multi-action agents this should be
            shaped `(batch, T, num_factors)`.
        beliefs_B: list[Array] | None, optional
            Optional sequence of beliefs used for transition updates. If `None`,
            transition updates are skipped.
        lr_pA: float, default=1.0
            Learning-rate multiplier for `A` updates.
        lr_pB: float, default=1.0
            Learning-rate multiplier for `B` updates.
        **kwargs: Any
            Reserved for future/compatibility arguments.

        Returns
        -------
        Agent
            Agent instance with updated `pA`, `A`, `pB`, and `B` where learning is
            enabled.
        """

        agent = self
        if observations is None:
            observations = kwargs.pop("outcomes", None)
            if observations is not None:
                warnings.warn(
                    "`outcomes` is deprecated in `infer_parameters`; use `observations`.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        elif "outcomes" in kwargs:
            raise ValueError("Pass either `observations` or `outcomes`, not both.")

        if observations is None:
            raise ValueError("`observations` must be provided to `infer_parameters`.")

        # ------------------------------------------------------------------
        # Prepare the sequences we'll use for A- and B- learning
        # ------------------------------------------------------------------
        # For updating A we use 'marginal_beliefs' (possibly smoothed under OVF).
        # For updating B we need a *time* sequence to construct joints or run smoothing.
        seq_beliefs = beliefs_A if beliefs_B is None else beliefs_B  # (list over factors) each (B, T, Ns_f)

        # Normalize action shape to include time if needed
        if actions is not None and actions.ndim == 2:
            # (B, Nu) -> (B, 1, Nu) so we can slice to T-1 below
            actions = jnp.expand_dims(actions, 1)
        
        # Infer sequence length (T) from beliefs provided for B learning
        # If learn_B is False we don't need a time sequence, but computing T is cheap and safe
        T = seq_beliefs[0].shape[1]  # each factor has (B, T, Ns_f)

        # Make actions time-conformant: always slice to T-1 along time
        if actions is not None and actions.ndim == 3:
            actions_Tm1 = actions[:, :max(T - 1, 0), :]    # (B, max(T-1, 0), Nu)
        else:
            actions_Tm1 = actions  # None or already time-matched

        if actions_Tm1 is not None:
            valid_transition_mask = jnp.all(actions_Tm1 >= 0, axis=-1)
        else:
            valid_transition_mask = None
        
        # A handy predicate: can we (meaningfully) apply sequence-based smoothing or
        # B-joint construction for this update window?
        # We need at least one valid transition.
        can_update_Beliefs = (
            actions_Tm1 is not None
            and (T > 1)
            and (actions_Tm1.shape[1] == (T - 1))
            and jnp.any(valid_transition_mask)
        )

        # Full B-parameter update additionally requires `self.learn_B`.
        can_update_B = (
            self.learn_B
            and can_update_Beliefs
        )

        def _apply_transition_mask(joint_beliefs: list[Array] | None) -> list[Array] | None:
            if valid_transition_mask is None:
                return joint_beliefs

            def _mask_joint(x: Array) -> Array:
                mask = valid_transition_mask.astype(x.dtype).reshape(
                    valid_transition_mask.shape + (1,) * (x.ndim - 2)
                )
                return x * mask

            return jtu.tree_map(_mask_joint, joint_beliefs)

        def _empty_joint_beliefs() -> list[Array]:
            return [
                jnp.zeros(
                    (
                        self.batch_size,
                        seq_beliefs[0].shape[1] - 1,
                        self.num_states[f],
                        *[self.num_states[dep] for dep in self.B_dependencies[f]],
                    )
                )
                for f in range(self.num_factors)
            ]

        def _build_outer_product_joints() -> list[list[Array]]:
            return [
                [seq_beliefs[f][:, 1:]] + [seq_beliefs[f_idx][:, :-1] for f_idx in self.B_dependencies[f]]
                for f in range(self.num_factors)
            ]

        def _smooth_or_fallback(smoothing_fn: Callable[..., Any]) -> tuple[list[Array], list[Array]]:
            def update_with_smoothing(_: Any) -> tuple[list[Array], list[Array]]:
                # Use the *sequence* of filtered beliefs (seq_beliefs) for smoothing
                #   vmap runs over batch: each call sees (T, Ns_f) and (T-1, Nu)
                smoothed_marginals_and_joints = vmap(smoothing_fn)(
                    seq_beliefs, self.B, actions_Tm1
                )
                marginal_beliefs = smoothed_marginals_and_joints[0]  # list[f] -> (B, T, Ns_f)
                joint_beliefs = _apply_transition_mask(smoothed_marginals_and_joints[1])  # list[f] -> (B, T-1, ...)
                return marginal_beliefs, joint_beliefs

            def use_filtered_beliefs(_: Any) -> tuple[list[Array], list[Array]]:
                # No valid transition yet (or sentinel action found):
                # - Use filtered beliefs for A-learning,
                # - and either skip B-learning or fall back to the two-frame outer-product joint.
                marginal_beliefs = seq_beliefs
                # Create empty joint_beliefs with same structure as the true branch would return
                joint_beliefs = _apply_transition_mask(_empty_joint_beliefs())
                return marginal_beliefs, joint_beliefs

            return lax.cond(
                can_update_Beliefs,
                update_with_smoothing,
                use_filtered_beliefs,
                operand=None
            )

        if self.inference_algo in inference.SMOOTHING_METHODS:
            smoothing_fn = inference.smoothing_ovf
            if self.inference_algo == inference.EXACT_METHOD:
                # Exact backward smoothing from online filtering history for single-factor HMMs.
                smoothing_fn = inference.smoothing_exact
            marginal_beliefs, joint_beliefs = _smooth_or_fallback(smoothing_fn)
        else:
            # Non-OVF: keep existing behavior (use filtered marginals and build joints from t and t-1)
            marginal_beliefs = beliefs_A
            joint_beliefs = _apply_transition_mask(_build_outer_product_joints()) if self.learn_B else None

        if self.learn_A:
            update_A = partial(
                learning.update_obs_likelihood_dirichlet,
                A_dependencies=self.A_dependencies,
                num_obs=self.num_obs,
                categorical_obs=self.categorical_obs,
            )
            
            lr = jnp.broadcast_to(lr_pA, (self.batch_size,))
            qA, E_qA = vmap(update_A)(
                self.pA,
                self.A,
                observations,
                marginal_beliefs,
                lr=lr,
            )
            
            agent = tree_at(lambda x: (x.A, x.pA), agent, (E_qA, qA))
            
        if self.learn_B:
            update_B = partial(learning.update_state_transition_dirichlet, num_controls=self.num_controls)
            lrB = jnp.broadcast_to(lr_pB, (self.batch_size,))

            def update_B_step(_: Any) -> tuple[list[Array], list[Array]]:
                qB, E_qB = vmap(update_B)(
                    self.pB,
                    self.B,
                    joint_beliefs,
                    actions_Tm1,   # time-aligned actions
                    lr=lrB
                )
                return qB, E_qB

            def skip_B_step(_: Any) -> tuple[list[Array], list[Array]]:
                return self.pB, self.B

            qB, E_qB = lax.cond(can_update_B, update_B_step, skip_B_step, operand=None)

            if self.use_inductive and self.H is not None:
                I_updated = lax.cond(
                    can_update_B,
                    lambda _: vmap(partial(control.generate_I_matrix, depth=self.inductive_depth))(
                        self.H, E_qB, self.inductive_threshold
                    ),
                    lambda _: self.I,
                    operand=None,
                )
                agent = tree_at(lambda x: (x.B, x.pB, x.I), agent, (E_qB, qB, I_updated))
            else:
                agent = tree_at(lambda x: (x.B, x.pB), agent, (E_qB, qB))

        return agent

    def process_obs(self, observations: list[Array] | list[int]) -> list[Array]:
        """
        Preprocess observations into the distributional format expected by the inference routines.

        Parameters
        ----------
        observations: list[Array] or list[int]
            The observation input. Format depends on the default preprocessing:

            - If `self.categorical_obs=False` (default): Each entry `observations[m]` is an integer
              index representing the discrete observation for modality `m`.

            - If `self.categorical_obs=True`: Each entry `observations[m]` is a 1D array representing
              a probability distribution over observations for modality `m`.

        Returns
        -------
        o_vec: list[Array]
            Observations in distributional form (one-hot vectors or categorical distributions).

        Notes
        -----
        If `self.preprocess_fn` is set on the agent, it takes precedence over the default
        categorical/discrete handling and will be used instead of the logic based on
        `self.categorical_obs`. This override only affects preprocessing; `self.categorical_obs`
        is still used by learning and planning code paths that consume raw observations.
        Ensure `self.categorical_obs` matches the output format of your preprocessing
        (or per-call `preprocess_fn`) to keep those paths consistent.
        """
        if self.preprocess_fn is not None:
            return self.preprocess_fn(observations)

        if self.categorical_obs:
            return observations

        return self.make_categorical(observations)

    def make_categorical(self, observations: list[Array] | list[int]) -> list[Array]:
        """
        Convert discrete index observations into one-hot categorical distributions.

        Parameters
        ----------
        observations: list[Array] or list[int]
            Each entry `observations[m]` is an integer index for modality `m`.

        Returns
        -------
        o_vec: list
            One-hot categorical distributions for each modality.
        """
        return [nn.one_hot(o, self.num_obs[m]) for m, o in enumerate(observations)]

    def infer_states(
        self,
        observations: list[Array] | list[int],
        empirical_prior: list[Array],
        *,
        past_actions: Array | None = None,
        qs_hist: list[Array] | None = None,
        valid_steps: int | Array | None = None,
        mask: list[Array] | None = None,
        preprocess_fn: Callable | None = None,
    ) -> list[Array]:
        """
        Update approximate posterior over hidden states by solving variational inference problem, given an observation.

        Parameters
        ----------
        observations: list[Array] | list[int]
            Observation input in one of two formats:

            - Discrete observations (default): each `observations[m]` is an
              integer index for modality `m`.
            - Categorical observations: each `observations[m]` is a probability
              vector over observations for modality `m`.

            If `preprocess_fn` is provided, it should map the raw input to
            categorical observations and takes precedence over default handling.

        empirical_prior: list[Array] or tuple[Array]
            Empirical prior beliefs over hidden states. Depending on the inference algorithm chosen,
            the resulting `empirical_prior` variable may be a matrix (or list[Array]).
            of additional dimensions to encode extra conditioning variables like timepoint and policy.

        past_actions: list[int] or tuple[int], optional
            The action input. Each entry `past_actions[f]` stores indices representing the actions
            for control factor `f`.

        qs_hist: list[Array] or tuple[Array], optional
            History of posterior beliefs over hidden states.

        valid_steps: Array or int, optional
            Number of valid (unpadded) timesteps when using fixed-size sequence windows.
            If provided, sequence inference methods (`mmp`, `vmp`) ignore padded prefix
            timesteps and transitions.

        mask: list[Array] or tuple[Array], optional
            Mask for observations.

        preprocess_fn: callable, optional
            Optional preprocessing function to convert observations into distributional form.
            If None, defaults to `self.process_obs`. The callable should accept
            `observations` and return distributional observations.

        Notes
        -----
        `categorical_obs` is no longer an argument to `infer_states`. Set it when
        constructing the agent or supply a `preprocess_fn`. If you provide a custom
        preprocessing function, ensure `self.categorical_obs` matches the output format,
        since it is still used by learning and planning code paths that consume raw observations.

        Returns
        -------
        qs: list[Array]
            Posterior beliefs over hidden states. Depending on the inference algorithm chosen,
            the resulting `qs` variable will have additional sub-structure to reflect whether
            beliefs are additionally conditioned on timepoint and policy.
            For example, in case the `self.inference_algo == 'MMP'` indexing structure is
            policy->timepoint->factor, so that `qs[p_idx][t_idx][f_idx]` refers to beliefs
            about marginal factor `f_idx` expected under policy `p_idx` at timepoint `t_idx`.

        Examples
        --------
        Discrete observations:

        >>> obs = [0, 1]  # Modality 0 observed observation 0, modality 1 observed observation 1
        >>> qs = agent.infer_states(obs, prior)

        Categorical observations:

        >>> obs = [
        ...     jnp.array([0.7, 0.2, 0.1]),  # Peaked belief distribution for observation 0
        ...     jnp.array([0.5, 0.5])        # Flat belief distribution for observation 1
        ... ]
        >>> agent_cat = Agent(..., categorical_obs=True)
        >>> qs = agent_cat.infer_states(obs, prior)
        """

        if preprocess_fn is None:
            o_vec = self.process_obs(observations)
        else:
            o_vec = preprocess_fn(observations)

        if not isinstance(empirical_prior, (list, tuple)):
            raise ValueError(
                "`empirical_prior` must be a list/tuple with one entry per hidden-state factor."
            )
        if len(empirical_prior) != self.num_factors:
            raise ValueError(
                f"`empirical_prior` has {len(empirical_prior)} factor(s), expected {self.num_factors}"
            )

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
            distr_obs=True,  # Always True because o_vec is expected to be distributional
            inference_horizon=self.inference_horizon,
        )

        if valid_steps is not None:
            valid_steps = jnp.asarray(valid_steps, dtype=jnp.int32)
            if valid_steps.ndim == 0:
                valid_steps = jnp.broadcast_to(valid_steps, (self.batch_size,))
            elif valid_steps.ndim != 1 or valid_steps.shape[0] != self.batch_size:
                raise ValueError(
                    "`valid_steps` must be a scalar or have shape `(batch_size,)`"
                )
        
        output = vmap(infer_states)(
            A,
            self.B,
            o_vec,
            past_actions,
            prior=empirical_prior,
            qs_hist=qs_hist,
            valid_steps=valid_steps,
        )

        return output

    def update_empirical_prior(self, action: Array, qs: list[Array]) -> list[Array]:
        """
        Compute the empirical prior used for the next state-inference step.

        Parameters
        ----------
        action: Array
            Action sampled at the current timestep for each control factor.
        qs: list[Array]
            Posterior beliefs over hidden states for the current timestep/history.

        Returns
        -------
        pred: list[Array]
            Predicted prior over hidden states for the next inference step.
            For sequence methods (`mmp`, `vmp`), this returns `self.D` to preserve sequence-inference semantics.
        """
        # this computation of the predictive prior is correct only for fully factorised Bs.
        if self.inference_algo in inference.SEQUENCE_METHODS:
            # in the case of the 'mmp' or 'vmp' we have to use D as prior parameter for infer states
            pred = self.D
        else:
            qs_last = jtu.tree_map( lambda x: x[:, -1], qs)
            propagate_beliefs = partial(control.compute_expected_state, B_dependencies=self.B_dependencies)
            pred = vmap(propagate_beliefs)(qs_last, self.B, action)
        
        return pred

    def infer_policies(self, qs: list[Array]) -> tuple[Array, Array]:
        """
        Perform policy inference by optimizing a posterior (categorical) distribution over policies.
        This distribution is computed as the softmax of `G * gamma + lnE` where `G` is the negative expected
        free energy of policies, `gamma` is a policy precision and `lnE` is the (log) prior probability of policies.
        This function returns the posterior over policies as well as the negative expected free energy of each policy.

        Parameters
        ----------
        qs: list[Array]
            Posterior beliefs over hidden states (typically output of
            `infer_states`), including the most recent timestep.

        Returns
        ----------
        q_pi: Array
            Posterior beliefs over policies with shape
            `(batch_size, num_policies)`.
        G: Array
            Negative expected free energies of policies with shape
            `(batch_size, num_policies)`.
        """

        latest_belief = jtu.tree_map(lambda x: x[:, -1], qs) # only get the posterior belief held at the current timepoint
        infer_policies = partial(
            control.update_posterior_policies_inductive,
            self.policies.policy_arr,
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
    
    def multiaction_probabilities(self, q_pi: Array) -> Array:
        """
        Compute probabilities of unique multi-actions from the posterior over policies.

        Parameters
        ----------
        q_pi: Array
            Posterior beliefs over policies for one batch element.

        Returns
        ----------
        Array
            Probability vector over unique multi-actions.
        """

        if self.sampling_mode == "marginal":
            get_marginals = partial(control.get_marginals, policies=self.policies.policy_arr, num_controls=self.num_controls)
            marginals = get_marginals(q_pi)
            outer = lambda a, b: jnp.outer(a, b).reshape(-1)
            marginals = jtu.tree_reduce(outer, marginals)

        elif self.sampling_mode == "full":
            locs = jnp.all(
                self.policies.policy_arr[:, 0] == jnp.expand_dims(self.unique_multiactions, -2),
                -1,
            )
            get_marginals = lambda x: jnp.where(locs, x, 0.).sum(-1)
            marginals = vmap(get_marginals)(q_pi)

        return marginals

    def sample_action(self, q_pi: Array, rng_key: Array | None = None) -> Array:
        """
        Sample or select a discrete action from the posterior over control states.
        
        Parameters
        ----------
        q_pi: Array
            Posterior over policies for each batch element (usually from
            `infer_policies`).
        rng_key: Array or sequence of keys, optional
            Required for stochastic action selection. For batched agents, pass a
            key array with one key per batch element.

        Returns
        ----------
        action: Array
            Action indices per batch element and control factor.
        """
        if (rng_key is None) and (self.action_selection == "stochastic"):
            raise ValueError("Please provide a random number generator key to sample actions stochastically")

        if self.sampling_mode == "marginal":
            sample_action = partial(control.sample_action, self.policies.policy_arr, self.num_controls, action_selection=self.action_selection)
            action = vmap(sample_action)(q_pi, alpha=self.alpha, rng_key=rng_key)
        elif self.sampling_mode == "full":
            sample_policy = partial(control.sample_policy, self.policies.policy_arr, action_selection=self.action_selection)
            action = vmap(sample_policy)(q_pi, alpha=self.alpha, rng_key=rng_key)

        return action
    
    def decode_multi_actions(self, action: Array) -> Array:
        """Decode flattened multi-actions back to factor-wise actions.

        Parameters
        ----------
        action: Array
            Flattened multi-action indices.

        Returns
        -------
        Array
            Array of shape `(batch_size, num_controls_multi)` containing decoded
            actions per control factor.
        """
        if self.action_maps is None:
            return action

        action_multi = jnp.zeros((self.batch_size, len(self.num_controls_multi))).astype(action.dtype)
        for f, action_map in enumerate(self.action_maps):
            if action_map["multi_dependency"] == []:
                continue

            action_multi_f = utils.index_to_combination(action[..., f], action_map["multi_dims"])
            action_multi = action_multi.at[..., action_map["multi_dependency"]].set(action_multi_f)
        return action_multi

    def encode_multi_actions(self, action_multi: Array) -> Array:
        """Encode factor-wise multi-actions into flattened actions.

        Parameters
        ----------
        action_multi: Array
            Array of actions per control factor.

        Returns
        -------
        Array
            Flattened action indices with shape `(batch_size, num_controls)`.
        """
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

    def get_model_dimensions(self) -> dict[str, Any]:
        """
        Collect key model dimensions in a single object.

        Returns
        -------
        dict[str, Any]
            Dictionary containing model shape metadata. Includes:

            - `num_obs`: list[int]
            - `num_states`: list[int]
            - `num_controls`: list[int]
            - `num_modalities`: int
            - `num_factors`: int
            - `num_policies`: int
            - `policy_len`: int
            - `inference_horizon`: int | None
            - `A_dependencies`: list[list[int]]
            - `B_dependencies`: list[list[int]]
        """
        return {
            "num_obs": self.num_obs,
            "num_states": self.num_states,
            "num_controls": self.num_controls,
            "num_modalities": self.num_modalities,
            "num_factors": self.num_factors,
            "num_policies": self.policies.num_policies,
            "policy_len": self.policy_len,
            "inference_horizon": self.inference_horizon,
            "A_dependencies": self.A_dependencies,
            "B_dependencies": self.B_dependencies,
        }

    def _construct_dependencies(
        self,
        A_dependencies: list[list[int]] | None,
        B_dependencies: list[list[int]] | None,
        B_action_dependencies: list[list[int]] | None,
        A: list[Array] | list[Distribution],
        B: list[Array] | list[Distribution],
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
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

    def _flatten_B_action_dims(
        self,
        B: list[Array],
        pB: list[Array] | None,
        B_action_dependencies: list[list[int]],
    ) -> tuple[list[Array], list[Array] | None, list[dict[str, Any]]]:
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

    def _construct_flattend_policies(self, policies: Array, action_maps: list[dict[str, Any]]) -> Array:
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

    def _get_default_params(self) -> dict[str, Any] | None:
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

    def _validate(self) -> None:
        for m in range(self.num_modalities):
            factor_dims = tuple([self.num_states[f] for f in self.A_dependencies[m]])
            assert (
                self.A[m].shape[2:] == factor_dims
            ), f"Please input an `A_dependencies` whose {m}-th indices correspond to the hidden state factors that line up with lagging dimensions of A[{m}]..."
            
            # validate A tensor is normalised
            utils.validate_normalization(self.A[m], axis=1, tensor_name=f"A[{m}]")
            
            if self.pA is not None:
                assert (
                    self.pA[m].shape[2:] == factor_dims if self.pA[m] is not None else True
                ), f"Please input an `A_dependencies` whose {m}-th indices correspond to the hidden state factors that line up with lagging dimensions of pA[{m}]..."
                    
            assert max(self.A_dependencies[m]) <= (
                self.num_factors - 1
            ), f"Check modality {m} of `A_dependencies` - must be consistent with `num_states` and `num_factors`..."

        for f in range(self.num_factors):
            factor_dims = tuple([self.num_states[f] for f in self.B_dependencies[f]])
            assert (
                self.B[f].shape[2:-1] == factor_dims
            ), f"Please input a `B_dependencies` whose {f}-th indices pick out the hidden state factors that line up with the all-but-final lagging dimensions of B[{f}]..."
            
            # validate B tensor is normalised
            utils.validate_normalization(self.B[f], axis=1, tensor_name=f"B[{f}]")
            
            if self.pB is not None:
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

        if self.inference_algo == inference.EXACT_METHOD:
            if self.num_factors != 1:
                raise ValueError("`exact` inference currently supports only a single hidden-state factor")
            if len(self.B_dependencies) != 1 or list(self.B_dependencies[0]) != [0]:
                raise ValueError(
                    "Exact inference requires single-factor self-dynamics only "
                    "(i.e. B_dependencies == [[0]])"
                )
            if any(list(deps) != [0] for deps in self.A_dependencies):
                raise ValueError(
                    "Exact inference requires each observation modality to depend only on factor 0 "
                    "(i.e. A_dependencies[m] == [0])"
                )
