#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-an-iterable
"""Policy construction, expected free energy, and action sampling utilities."""

import itertools
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Sequence
from functools import partial
from jax import lax, vmap, nn
from jax import random as jr
from jaxtyping import Array

import equinox as eqx

from pymdp.maths import factor_dot, log_stable, stable_entropy, stable_xlogx, spm_wnorm

class Policies(eqx.Module):
    """ 
    A class for storing an array of policies and its properties
    
    """
    policy_arr: Array
    horizon: int = eqx.field(static=True)
    num_policies: int = eqx.field(static=True)

    def __init__(self, policy_arr: Array) -> None:
        self.num_policies = policy_arr.shape[0]
        self.horizon = policy_arr.shape[1]
        self.policy_arr = policy_arr
    
    def __getitem__(self, idx: int) -> Array:
        return self.policy_arr[idx]
    
    def __len__(self) -> int:
        return self.num_policies
    
def get_marginals(q_pi: Array, policies: Array, num_controls: Sequence[int]) -> list[Array]:
    """
    Computes the marginal posterior(s) over actions by integrating their posterior probability under the policies that they appear within.

    Parameters
    ----------
    q_pi: 1D Array
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
    policies: Array
        Policy matrix with shape `(num_policies, policy_len, num_factors)`.
    num_controls: Sequence[int]
        Dimensionalities of each control state factor.
    
    Returns
    ----------
    action_marginals: list[Array]
        Marginal posterior over actions for each control factor.
    """
    num_factors = len(num_controls)    

    action_marginals = []
    for factor_i in range(num_factors):
        actions = jnp.arange(num_controls[factor_i])[:, None]
        action_marginals.append(jnp.where(actions==policies[:, 0, factor_i], q_pi, 0).sum(-1))
    
    return action_marginals

def sample_action(
    policies: Array,
    num_controls: Sequence[int],
    q_pi: Array,
    action_selection: str = "deterministic",
    alpha: float = 16.0,
    rng_key: Array | None = None,
) -> Array:
    """
    Samples an action from posterior marginals, one action per control factor.

    Parameters
    ----------
    q_pi: 1D Array
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
    policies: Array
        Policy matrix with shape `(num_policies, policy_len, num_factors)`.
    num_controls: Sequence[int]
        Dimensionalities of each control state factor.
    action_selection: string, default "deterministic"
        String indicating whether whether the selected action is chosen as the maximum of the posterior over actions,
        or whether it's sampled from the posterior marginal over actions
    alpha: float, default 16.0
        Action selection precision -- the inverse temperature of the softmax that is used to scale the 
        action marginals before sampling. This is only used if `action_selection` is `"stochastic"`.
    rng_key: Array | None, optional
        PRNG key required when `action_selection='stochastic'`.

    Returns
    ----------
    selected_policy: 1D Array
        Vector containing the indices of the actions for each control factor
    """

    marginal = get_marginals(q_pi, policies, num_controls)
    
    if action_selection == 'deterministic':
        selected_policy = jtu.tree_map(lambda x: jnp.argmax(x, -1), marginal)
    elif action_selection == 'stochastic':
        logits = lambda x: alpha * log_stable(x)
        selected_policy = jtu.tree_map(lambda x: jr.categorical(rng_key, logits(x)), marginal)
    else:
        raise NotImplementedError

    return jnp.array(selected_policy)

def sample_policy(
    policies: Array,
    q_pi: Array,
    action_selection: str = "deterministic",
    alpha: float = 16.0,
    rng_key: Array | None = None,
) -> Array:
    """Select or sample a policy, then return its first-step multi-action.

    Parameters
    ----------
    policies : Array
        Policy matrix with shape `(num_policies, policy_len, num_factors)`.
    q_pi : Array
        Posterior over policies for one batch element.
    action_selection : {"deterministic", "stochastic"}, default="deterministic"
        Selection mode for choosing a policy.
    alpha : float, default=16.0
        Precision (inverse temperature) used for stochastic sampling.
    rng_key : Array | None, optional
        PRNG key required for `action_selection='stochastic'`.

    Returns
    -------
    Array
        First-step action vector for all control factors.
    """

    if action_selection == "deterministic":
        policy_idx = jnp.argmax(q_pi)
    elif action_selection == "stochastic":
        log_p_policies = log_stable(q_pi) * alpha
        policy_idx = jr.categorical(rng_key, log_p_policies)

    selected_multiaction = policies[policy_idx, 0]
    return selected_multiaction

def construct_policies(
    num_states: Sequence[int],
    num_controls: Sequence[int] | None = None,
    policy_len: int = 1,
    control_fac_idx: Sequence[int] | None = None,
) -> Array:
    """
    Generate an exhaustive policy matrix for the specified planning horizon.

    Parameters
    ----------
    num_states: Sequence[int]
        Dimensionalities of each hidden state factor.
    num_controls: Sequence[int], default None
        Dimensionalities of each control state factor. If `None`, this is
        computed from controllable state factors.
    policy_len: int, default 1
        temporal depth ("planning horizon") of policies
    control_fac_idx: Sequence[int]
        Indices of controllable hidden state factors (factors `i` where `num_controls[i] > 1`).

    Returns
    ----------
    policies: Array
        Policy matrix with shape `(num_policies, policy_len, num_factors)`.
    """

    num_factors = len(num_states)
    if control_fac_idx is None:
        if num_controls is not None:
            control_fac_idx = [f for f, n_c in enumerate(num_controls) if n_c > 1]
        else:
            control_fac_idx = list(range(num_factors))

    if num_controls is None:
        num_controls = [num_states[c_idx] if c_idx in control_fac_idx else 1 for c_idx in range(num_factors)]
        
    x = num_controls * policy_len
    policies = list(itertools.product(*[list(range(i)) for i in x]))
    
    for pol_i in range(len(policies)):
        policies[pol_i] = jnp.array(policies[pol_i]).reshape(policy_len, num_factors)

    return jnp.stack(policies)


def update_posterior_policies(
    policy_matrix: Array,
    qs_init: list[Array],
    A: list[Array],
    B: list[Array],
    C: list[Array],
    E: Array,
    pA: list[Array] | None,
    pB: list[Array] | None,
    A_dependencies: list[list[int]],
    B_dependencies: list[list[int]],
    gamma: float = 16.0,
    use_utility: bool = True,
    use_states_info_gain: bool = True,
    use_param_info_gain: bool = False,
) -> tuple[Array, Array]:
    """Compute posterior over policies and policy-wise negative expected free energy.

    Notes
    -----
    The returned policy score is `neg_efe = -EFE`. In SPM-style notation this
    same quantity is often denoted by `G`.

    Parameters
    ----------
    policy_matrix: Array
        Policy tensor with shape `(num_policies, policy_len, num_factors)`.
    qs_init: list[Array]
        Current marginal beliefs over hidden states.
    A: list[Array]
        Observation likelihood tensors.
    B: list[Array]
        Transition tensors.
    C: list[Array]
        Prior preferences over observations.
    E: Array
        Prior over policies.
    pA: list[Array] | None
        Posterior Dirichlet parameters for `A` (required when `use_param_info_gain=True`).
    pB: list[Array] | None
        Posterior Dirichlet parameters for `B` (required when `use_param_info_gain=True`).
    A_dependencies: list[list[int]]
        Observation dependencies between modalities and hidden-state factors.
    B_dependencies: list[list[int]]
        Transition dependencies between hidden-state factors.
    gamma: float, default=16.0
        Policy precision parameter.
    use_utility: bool, default=True
        Whether to include expected utility in EFE.
    use_states_info_gain: bool, default=True
        Whether to include state epistemic value.
    use_param_info_gain: bool, default=False
        Whether to include parameter epistemic value.

    Returns
    -------
    tuple[Array, Array]
        `(q_pi, neg_efe_all_policies)` where `q_pi` is the posterior over policies.
    """
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    ## vmap across policies
    compute_neg_efe_fixed_states = partial(
        compute_neg_efe_policy,
        qs_init,
        A,
        B,
        C,
        pA,
        pB,
        A_dependencies,
        B_dependencies,
        use_utility=use_utility,
        use_states_info_gain=use_states_info_gain,
        use_param_info_gain=use_param_info_gain,
    )

    # only in the case of policy-dependent qs_inits
    # in_axes_list = (1,) * n_factors
    # all_efe_of_policies = vmap(compute_neg_efe_policy, in_axes=(in_axes_list, 0))(qs_init_pi, policy_matrix)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    neg_efe_all_policies = vmap(compute_neg_efe_fixed_states)(policy_matrix)

    return nn.softmax(gamma * neg_efe_all_policies + log_stable(E)), neg_efe_all_policies

def compute_expected_state(
    qs_prior: list[Array],
    B: list[Array],
    u_t: Array | Sequence[int],
    B_dependencies: list[list[int]] | None = None,
) -> list[Array]:
    """
    Compute posterior over next state, given belief about previous state, transition model and action...

    Parameters
    ----------
    qs_prior: list[Array]
        Marginal beliefs over hidden states at time `t`.
    B: list[Array]
        Transition model tensors.
    u_t: Array | Sequence[int]
        Action indices for each control factor.
    B_dependencies: list[list[int]], optional
        Optional dependencies used to marginalize transition tensors. If `None`,
        defaults to factor-local transitions.

    Returns
    -------
    list[Array]
        Marginal beliefs over next-time hidden states.
    """
    #Note: this algorithm is only correct if each factor depends only on itself. For any interactions, 
    # we will have empirical priors with codependent factors. 
    assert len(u_t) == len(B)
    if B_dependencies is None:
        B_dependencies = [[f] for f in range(len(B))]
    qs_next = []
    for B_f, u_f, deps in zip(B, u_t, B_dependencies):
        relevant_factors = [qs_prior[idx] for idx in deps]
        qs_next_f = factor_dot(B_f[...,u_f], relevant_factors, keep_dims=(0,))
        qs_next.append(qs_next_f)
        
    # P(s'|s, u) = \sum_{s, u} P(s'|s) P(s|u) P(u|pi)P(pi) because u </-> pi
    return qs_next

def compute_expected_state_and_Bs(
    qs_prior: list[Array], B: list[Array], u_t: Array | Sequence[int]
) -> tuple[list[Array], list[Array]]:
    """Compute one-step predictive states and selected transition matrices.

    Parameters
    ----------
    qs_prior: list[Array]
        Marginal beliefs over hidden states at time `t`.
    B: list[Array]
        Transition model tensors for each hidden-state factor.
    u_t: Array | Sequence[int]
        Action indices for each control factor at time `t`.

    Returns
    -------
    tuple[list[Array], list[Array]]
        `(qs_next, Bs)` where `qs_next` are next-state marginals and `Bs` are
        the action-conditioned transition slices used for each factor.
    """
    assert len(u_t) == len(B)  
    qs_next = []
    Bs = []
    for qs_f, B_f, u_f in zip(qs_prior, B, u_t):
        qs_next.append( B_f[..., u_f].dot(qs_f) )
        Bs.append(B_f[..., u_f])
    
    return qs_next, Bs

def compute_expected_obs(
    qs: list[Array], A: list[Array], A_dependencies: list[list[int]]
) -> list[Array]:
    """
    New version of expected observation (computation of Q(o|pi)) that takes into account sparse dependencies between observation
    modalities and hidden state factors

    Parameters
    ----------
    qs: list[Array]
        Beliefs over hidden states.
    A: list[Array]
        Observation likelihood models.
    A_dependencies: list[list[int]]
        Observation dependencies between modalities and state factors.

    Returns
    -------
    list[Array]
        Predictive beliefs over observations for each modality.
    """
        
    def compute_expected_obs_modality(A_m: Array, m: int) -> Array:
        deps = A_dependencies[m]
        relevant_factors = [qs[idx] for idx in deps]
        return factor_dot(A_m, relevant_factors, keep_dims=(0,))

    return jtu.tree_map(compute_expected_obs_modality, A, list(range(len(A))))

def compute_info_gain(
    qs: list[Array], qo: list[Array], A: list[Array], A_dependencies: list[list[int]]
) -> Array:
    """Compute expected state-information gain term of expected free energy.

    Parameters
    ----------
    qs: list[Array]
        Predicted hidden-state beliefs.
    qo: list[Array]
        Predicted observation beliefs.
    A: list[Array]
        Observation likelihood tensors.
    A_dependencies: list[list[int]]
        Observation dependencies between modalities and hidden-state factors.

    Returns
    -------
    Array
        Scalar epistemic value from expected information gain.
    """

    def compute_info_gain_for_modality(qo_m: Array, A_m: Array, m: int) -> Array:
        H_qo = stable_entropy(qo_m)
        H_A_m = - stable_xlogx(A_m).sum(0)
        deps = A_dependencies[m]
        relevant_factors = [qs[idx] for idx in deps]
        qs_H_A_m = factor_dot(H_A_m, relevant_factors)
        return H_qo - qs_H_A_m
    
    info_gains_per_modality = jtu.tree_map(compute_info_gain_for_modality, qo, A, list(range(len(A))))
        
    return jtu.tree_reduce(lambda x,y: x+y, info_gains_per_modality)

def compute_expected_utility(qo: list[Array], C: list[Array], t: int = 0) -> Array:
    """Compute expected utility from predictive observations and preferences.

    Parameters
    ----------
    qo: list[Array]
        Predicted observations for each modality.
    C: list[Array]
        Prior preferences per modality. Each modality can be static `(num_obs,)`
        or time-indexed `(policy_len, num_obs)`.
    t: int, default=0
        Planning timestep used when `C[m]` is time-indexed.

    Returns
    -------
    Array
        Scalar expected utility contribution.
    """

    util = 0.
    for o_m, C_m in zip(qo, C):
        if C_m.ndim > 1:
            util += (o_m * C_m[t]).sum()
        else:
            util += (o_m * C_m).sum()
    
    return util

def calc_pA_info_gain(
    pA: list[Array], qo: list[Array], qs: list[Array], A_dependencies: list[list[int]]
) -> Array:
    """
    Compute expected Dirichlet information gain about parameters `pA` for a given posterior predictive distribution over observations `qo` and states `qs`.

    Parameters
    ----------
    pA: list[Array]
        Dirichlet parameters over observation model (same shape as `A`)
    qo: list[Array]
        Predictive posterior beliefs over observations; stores the beliefs about
        observations expected under the policy at some arbitrary time `t`
    qs: list[Array]
        Predictive posterior beliefs over hidden states, stores the beliefs about
        hidden states expected under the policy at some arbitrary time `t`

    Returns
    -------
    infogain_pA: float
        Surprise (about Dirichlet parameters) expected for the pair of posterior predictive distributions `qo` and `qs`
    """

    def infogain_per_modality(pa_m: Array, qo_m: Array, m: int) -> Array:
        wa_m = spm_wnorm(pa_m) * (pa_m > 0.)
        fd = factor_dot(wa_m, [s for f, s in enumerate(qs) if f in A_dependencies[m]], keep_dims=(0,))[..., None]
        return qo_m.dot(fd)

    pA_infogain_per_modality = jtu.tree_map(
        infogain_per_modality, pA, qo, list(range(len(qo)))
    )
    
    infogain_pA = jtu.tree_reduce(lambda x, y: x + y, pA_infogain_per_modality)
    return infogain_pA.squeeze(-1)

def calc_pB_info_gain(
    pB: list[Array],
    qs_t: list[Array],
    qs_t_minus_1: list[Array],
    B_dependencies: list[list[int]],
    u_t_minus_1: Array | Sequence[int],
) -> Array:
    """
    Compute expected Dirichlet information gain about parameters `pB` under a given policy

    Parameters
    ----------
    pB: list[Array]
        Dirichlet parameters over transition model (same shape as `B`)
    qs_t: list[Array]
        Predictive posterior beliefs over hidden states expected under the policy at time `t`
    qs_t_minus_1: list[Array]
        Posterior over hidden states at time `t-1` (before receiving observations)
    u_t_minus_1: Array | Sequence[int]
        Actions in time step t-1 for each factor

    Returns
    -------
    infogain_pB: float
        Surprise (about Dirichlet parameters) expected under the policy in question
    """
    
    wB = lambda pb:  spm_wnorm(pb) * (pb > 0.)
    fd = lambda x, i: factor_dot(x, [s for f, s in enumerate(qs_t_minus_1) if f in B_dependencies[i]], keep_dims=(0,))[..., None]
    
    pB_infogain_per_factor = jtu.tree_map(lambda pb, qs, f: qs.dot(fd(wB(pb[..., u_t_minus_1[f]]), f)), pB, qs_t, list(range(len(qs_t))))
    infogain_pB = jtu.tree_reduce(lambda x, y: x + y, pB_infogain_per_factor)[0]
    return infogain_pB

def compute_neg_efe_policy(
    qs_init: list[Array],
    A: list[Array],
    B: list[Array],
    C: list[Array],
    pA: list[Array] | None,
    pB: list[Array] | None,
    A_dependencies: list[list[int]],
    B_dependencies: list[list[int]],
    policy_i: Array,
    use_utility: bool = True,
    use_states_info_gain: bool = True,
    use_param_info_gain: bool = False,
) -> Array:
    """Compute policy-wise negative expected free energy for one policy.

    Notes
    -----
    This function computes `neg_efe = -EFE` for a single policy. In SPM-style
    notation, this policy score (`neg_efe`) is commonly denoted `G`.

    Parameters
    ----------
    qs_init: list[Array]
        Initial hidden-state marginals at current timestep.
    A: list[Array]
        Observation likelihood tensors.
    B: list[Array]
        Transition tensors.
    C: list[Array]
        Prior preferences over observations.
    pA: list[Array] | None
        Posterior Dirichlet parameters for `A`.
    pB: list[Array] | None
        Posterior Dirichlet parameters for `B`.
    A_dependencies: list[list[int]]
        Observation dependencies between modalities and hidden-state factors.
    B_dependencies: list[list[int]]
        Transition dependencies between hidden-state factors.
    policy_i: Array
        Single policy trajectory with shape `(policy_len, num_factors)`.
    use_utility: bool, default=True
        Include expected utility term.
    use_states_info_gain: bool, default=True
        Include state-information-gain term.
    use_param_info_gain: bool, default=False
        Include parameter-information-gain term.

    Returns
    -------
    Array
        Scalar negative expected free energy for `policy_i`.
    """

    def scan_body(carry: tuple[list[Array], Array], t: Array) -> tuple[tuple[list[Array], Array], None]:

        qs, neg_efe = carry

        qs_next = compute_expected_state(qs, B, policy_i[t], B_dependencies)

        qo = compute_expected_obs(qs_next, A, A_dependencies)

        info_gain = compute_info_gain(qs_next, qo, A, A_dependencies) if use_states_info_gain else 0.

        utility = compute_expected_utility(qo, C, t) if use_utility else 0.

        param_info_gain = calc_pA_info_gain(pA, qo, qs_next, A_dependencies) if use_param_info_gain else 0.
        param_info_gain += calc_pB_info_gain(pB, qs_next, qs, B_dependencies, policy_i[t]) if use_param_info_gain else 0.

        neg_efe += info_gain + utility + param_info_gain

        return (qs_next, neg_efe), None

    qs = qs_init
    neg_efe = 0.
    final_state, _ = lax.scan(scan_body, (qs, neg_efe), jnp.arange(policy_i.shape[0]))
    qs_final, neg_efe = final_state
    return neg_efe

def compute_neg_efe_policy_inductive(
    qs_init: list[Array],
    A: list[Array],
    B: list[Array],
    C: list[Array],
    pA: list[Array] | None,
    pB: list[Array] | None,
    A_dependencies: list[list[int]],
    B_dependencies: list[list[int]],
    I: list[Array],
    policy_i: Array,
    inductive_epsilon: float = 1e-3,
    use_utility: bool = True,
    use_states_info_gain: bool = True,
    use_param_info_gain: bool = False,
    use_inductive: bool = False,
) -> Array:
    """
    Compute policy-wise negative expected free energy with inductive planning.

    Notes
    -----
    This function computes `neg_efe = -EFE` for a single policy with optional
    inductive-value terms. In SPM-style notation, this score is commonly
    denoted `G`, so here `G = neg_efe = -EFE`.

    Parameters
    ----------
    qs_init: list[Array]
        Initial hidden-state marginals at current timestep.
    A: list[Array]
        Observation likelihood tensors.
    B: list[Array]
        Transition tensors.
    C: list[Array]
        Prior preferences over observations.
    pA: list[Array] | None
        Posterior Dirichlet parameters for `A`.
    pB: list[Array] | None
        Posterior Dirichlet parameters for `B`.
    A_dependencies: list[list[int]]
        Observation dependencies between modalities and hidden-state factors.
    B_dependencies: list[list[int]]
        Transition dependencies between hidden-state factors.
    I: list[Array]
        Inductive reachability matrices.
    policy_i: Array
        Single policy trajectory with shape `(policy_len, num_factors)`.
    inductive_epsilon: float, default=1e-3
        Scale of the inductive-value contribution.
    use_utility: bool, default=True
        Include expected utility term.
    use_states_info_gain: bool, default=True
        Include state-information-gain term.
    use_param_info_gain: bool, default=False
        Include parameter-information-gain term.
    use_inductive: bool, default=False
        Include inductive-value term.

    Returns
    -------
    Array
        Scalar negative expected free energy for `policy_i`.
    """

    def scan_body(carry: tuple[list[Array], Array], t: Array) -> tuple[tuple[list[Array], Array], None]:

        qs, neg_efe = carry

        qs_next = compute_expected_state(qs, B, policy_i[t], B_dependencies)

        qo = compute_expected_obs(qs_next, A, A_dependencies)

        info_gain = compute_info_gain(qs_next, qo, A, A_dependencies) if use_states_info_gain else 0.

        utility = compute_expected_utility(qo, C, t) if use_utility else 0.

        inductive_value = calc_inductive_value_t(qs_init, qs_next, I, epsilon=inductive_epsilon) if use_inductive else 0.

        param_info_gain = 0.
        if pA is not None:
            param_info_gain += calc_pA_info_gain(pA, qo, qs_next, A_dependencies) if use_param_info_gain else 0.
        if pB is not None:
            param_info_gain += calc_pB_info_gain(pB, qs_next, qs, B_dependencies, policy_i[t]) if use_param_info_gain else 0.

        neg_efe += info_gain + utility - param_info_gain + inductive_value

        return (qs_next, neg_efe), None

    qs = qs_init
    neg_efe = 0.
    final_state, _ = lax.scan(scan_body, (qs, neg_efe), jnp.arange(policy_i.shape[0]))
    _, neg_efe = final_state
    return neg_efe

def update_posterior_policies_inductive(
    policy_matrix: Array,
    qs_init: list[Array],
    A: list[Array],
    B: list[Array],
    C: list[Array],
    E: Array,
    pA: list[Array] | None,
    pB: list[Array] | None,
    A_dependencies: list[list[int]],
    B_dependencies: list[list[int]],
    I: list[Array],
    gamma: float = 16.0,
    inductive_epsilon: float = 1e-3,
    use_utility: bool = True,
    use_states_info_gain: bool = True,
    use_param_info_gain: bool = False,
    use_inductive: bool = True,
) -> tuple[Array, Array]:
    """
    Compute policy posterior and negative expected free energy with optional
    inductive terms.

    Notes
    -----
    The returned policy score is `neg_efe = -EFE`. In SPM-style notation this
    same quantity is often denoted by `G`.

    Parameters
    ----------
    policy_matrix: Array
        Policy tensor with shape `(num_policies, policy_len, num_factors)`.
    qs_init: list[Array]
        Current marginal state beliefs.
    A: list[Array]
        Observation likelihood models.
    B: list[Array]
        Transition models.
    C: list[Array]
        Prior preference vectors.
    E: Array
        Policy prior over the policy space.
    pA: list[Array] | None
        Optional posterior Dirichlet parameters for `A`.
    pB: list[Array] | None
        Optional posterior Dirichlet parameters for `B`.
    A_dependencies: list[list[int]]
        Observation dependencies between modalities and state factors.
    B_dependencies: list[list[int]]
        Transition dependencies between hidden-state factors and control factors.
    I: list[Array]
        Inductive planning matrices.
    gamma: float = 16.0
        Policy precision for softmax policy posterior.
    inductive_epsilon: float = 1e-3
        Inductive value scale factor.
    use_utility: bool = True
        Include utility term in expected free energy.
    use_states_info_gain: bool = True
        Include epistemic state-information gain term.
    use_param_info_gain: bool = False
        Include epistemic parameter-information gain term.
    use_inductive: bool = True
        Include inductive value term.

    Returns
    -------
    q_pi: Array
        Posterior over policies.
    neg_efe_all_policies: Array
        Policy-wise negative expected free energies.
    """
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    ## vmap across policies
    compute_neg_efe_fixed_states = partial(
        compute_neg_efe_policy_inductive,
        qs_init,
        A,
        B,
        C,
        pA,
        pB,
        A_dependencies,
        B_dependencies,
        I,
        inductive_epsilon=inductive_epsilon,
        use_utility=use_utility,
        use_states_info_gain=use_states_info_gain,
        use_param_info_gain=use_param_info_gain,
        use_inductive=use_inductive,
    )

    # only in the case of policy-dependent qs_inits
    # in_axes_list = (1,) * n_factors
    # all_efe_of_policies = vmap(compute_neg_efe_policy, in_axes=(in_axes_list, 0))(qs_init_pi, policy_matrix)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    neg_efe_all_policies = vmap(compute_neg_efe_fixed_states)(policy_matrix)

    return nn.softmax(gamma * neg_efe_all_policies + log_stable(E)), neg_efe_all_policies

def generate_I_matrix(H: list[Array], B: list[Array], threshold: float, depth: int) -> list[Array]:
    """ 
    Generates the `I` matrices used in inductive planning. These matrices stores the probability of reaching the goal state backwards from state j (columns) after i (rows) steps.
    Parameters
    ----------    
    H: list[Array]
        Constraints over desired states (1 if you want to reach that state, 0 otherwise)
    B: list[Array]
        Dynamics likelihood mapping or transition model, mapping from hidden states at `t` to hidden states at `t+1`, given some control state `u`.
        Each element `B[f]` stores a 3-D tensor for hidden state factor `f`, whose entries `B[f][s, v, u]` store the probability
        of hidden state level `s` at the current time, given hidden state level `v` and action `u` at the previous time.
    threshold: float
        The threshold for pruning transitions that are below a certain probability
    depth: int
        The temporal depth of the backward induction

    Returns
    ----------
    I: list[Array]
        For each state factor, contains a 2D array whose element i,j yields the probability
        of reaching the goal state backwards from state j after i steps.
    """
    
    num_factors = len(H)
    I = []
    for f in range(num_factors):
        """
        For each factor, we need to compute the probability of reaching the goal state
        """

        # If there exists an action that allows transitioning 
        # from state to next_state, with probability larger than threshold
        # set b_reachable[current_state, previous_state] to 1
        b_reachable = jnp.where(B[f] > threshold, 1.0, 0.0).sum(axis=-1)
        b_reachable = jnp.where(b_reachable > 0., 1.0, 0.0)

        def step_fn(carry: Array, i: Array) -> tuple[Array, Array]:
            I_prev = carry
            I_next = jnp.dot(b_reachable, I_prev)
            I_next = jnp.where(I_next > 0.1, 1.0, 0.0) # clamp I_next to 1.0 if it's above 0.1, 0 otherwise
            return I_next, I_next
    
        _, I_f = lax.scan(step_fn, H[f], jnp.arange(depth-1))
        I_f = jnp.concatenate([H[f][None,...], I_f], axis=0)

        I.append(I_f)
    
    return I

def calc_inductive_value_t(
    qs: list[Array], qs_next: list[Array], I: list[Array], epsilon: float = 1e-3
) -> Array:
    """
    Computes the inductive value of a state at a particular time (translation of @tverbele's `numpy` implementation of inductive planning, formerly
    called `calc_inductive_cost`).

    Parameters
    ----------
    qs: list[Array]
        Marginal posterior beliefs over hidden states at a given timepoint.
    qs_next: list[Array]
        Predictive posterior beliefs over hidden states expected under the policy.
    I: list[Array]
        For each state factor, contains a 2D array whose element i,j yields the probability
        of reaching the goal state backwards from state j after i steps.
    epsilon: float
        Value that tunes the strength of the inductive value (how much it contributes to the expected free energy of policies)

    Returns
    -------
    inductive_val: float
        Value (negative inductive cost) of visiting this state using backwards induction under the policy in question
    """
    
    # initialise inductive value
    inductive_val = 0.

    log_eps = log_stable(epsilon)
    for f in range(len(qs)):
        # we also assume precise beliefs here?!
        idx = jnp.argmax(qs[f])
        # m = arg max_n p_n < sup p

        # i.e. find first entry at which I_idx equals 1, and then m is the index before that
        m = jnp.maximum(jnp.argmax(I[f][:, idx])-1, 0)
        I_m = (1. - I[f][m, :]) * log_eps
        path_available = jnp.clip(I[f][:, idx].sum(0), min=0, max=1) # if there are any 1's at all in that column of I, then this == 1, otherwise 0
        inductive_val += path_available * I_m.dot(qs_next[f]) # scaling by path_available will nullify the addition of inductive value in the case we find no path to goal (i.e. when no goal specified)

    return inductive_val

# if __name__ == '__main__':

#     from jax import random as jr
#     key = jr.PRNGKey(1)
#     num_obs = [3, 4]

#     A = [jr.uniform(key, shape = (no, 2, 2)) for no in num_obs]
#     B = [jr.uniform(key, shape = (2, 2, 2)), jr.uniform(key, shape = (2, 2, 2))]
#     C = [log_stable(jnp.array([0.8, 0.1, 0.1])), log_stable(jnp.ones(4)/4)]
#     policy_1 = jnp.array([[0, 1],
#                          [1, 1]])
#     policy_2 = jnp.array([[1, 0],
#                          [0, 0]])
#     policy_matrix = jnp.stack([policy_1, policy_2]) # 2 x 2 x 2 tensor
    
#     qs_init = [jnp.ones(2)/2, jnp.ones(2)/2]
#     neg_efe_all_policies = jit(update_posterior_policies)(policy_matrix, qs_init, A, B, C)
#     print(neg_efe_all_policies)
