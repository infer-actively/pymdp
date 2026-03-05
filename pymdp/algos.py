"""Core variational-inference and exact-HMM algorithm implementations.

This module contains lower-level routines used by high-level inference/control
APIs. Public entry points include fixed-point iteration, message-passing over
sequences, and exact single-factor scan-based HMM smoothing.
"""

import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Any, Callable, NamedTuple, Tuple
from jax import jit, vmap, grad, lax, nn
from jaxtyping import Array
# from jax.config import config
# config.update("jax_enable_x64", True)

from pymdp.maths import compute_log_likelihood, compute_log_likelihood_per_modality, log_stable, factor_dot, factor_dot_flex, MINVAL

def add(x: Array, y: Array) -> Array:
    return x + y

def marginal_log_likelihood(qs: list[Array], log_likelihood: Array, i: int) -> Array:
    xs = [q for j, q in enumerate(qs) if j != i]
    return factor_dot(log_likelihood, xs, keep_dims=(i,))

def all_marginal_log_likelihood(
    qs: list[Array], log_likelihoods: list[Array], all_factor_lists: list[list[int]]
) -> list[Array]:
    qL_marginals = jtu.tree_map(lambda ll_m, factor_list_m: mll_factors(qs, ll_m, factor_list_m), log_likelihoods, all_factor_lists)
    
    num_factors = len(qs)

    # insted of a double loop we could have a list defining m to f mapping
    # which could be resolved with a single tree_map cast
    qL_all = [jnp.zeros(1)] * num_factors
    for m, factor_list_m in enumerate(all_factor_lists):
        for l, f in enumerate(factor_list_m):
            qL_all[f] += qL_marginals[m][l]

    return qL_all

def mll_factors(qs: list[Array], ll_m: Array, factor_list_m: list[int]) -> list[Array]:
    relevant_factors = [qs[f] for f in factor_list_m]
    marginal_ll_f = jtu.Partial(marginal_log_likelihood, relevant_factors, ll_m)
    loc_nf = len(factor_list_m)
    loc_factors = list(range(loc_nf))
    return jtu.tree_map(marginal_ll_f, loc_factors)

def run_vanilla_fpi(
    A: list[Array],
    obs: list[Array],
    prior: list[Array],
    num_iter: int = 1,
    distr_obs: bool = True,
) -> list[Array]:
    """ Vanilla fixed point iteration (jaxified) """

    nf = len(prior)
    factors = list(range(nf))
    # Step 1: Compute log likelihoods for each factor
    ll = compute_log_likelihood(obs, A, distr_obs=distr_obs)
    # log_likelihoods = [ll] * nf

    # Step 2: Map prior to log space and create initial log-posterior
    log_prior = jtu.tree_map(log_stable, prior)
    log_q = jtu.tree_map(jnp.zeros_like, prior)

    # Step 3: Iterate until convergence
    def scan_fn(carry: list[Array], t: Array) -> tuple[list[Array], None]:
        log_q = carry
        q = jtu.tree_map(nn.softmax, log_q)
        mll = jtu.Partial(marginal_log_likelihood, q, ll)
        marginal_ll = jtu.tree_map(mll, factors)
        log_q = jtu.tree_map(add, marginal_ll, log_prior)

        return log_q, None

    res, _ = lax.scan(scan_fn, log_q, jnp.arange(num_iter))

    # Step 4: Map result to factorised posterior
    qs = jtu.tree_map(nn.softmax, res)
    return qs

def run_factorized_fpi(
    A: list[Array],
    obs: list[Array],
    prior: list[Array],
    A_dependencies: list[list[int]],
    num_iter: int = 1,
    distr_obs: bool = True,
) -> list[Array]:
    """
    Run the fixed point iteration algorithm with sparse dependencies between factors and observations (stored in `A_dependencies`)
    """

    # Exact one-pass update for single-factor models.
    if len(prior) == 1:
        log_likelihood = compute_log_likelihood(obs, A, distr_obs=distr_obs)
        log_q = log_likelihood + log_stable(prior[0])
        return [nn.softmax(log_q, axis=-1)]

    # Step 1: Compute log likelihoods for each factor
    log_likelihoods = compute_log_likelihood_per_modality(obs, A, distr_obs=distr_obs)

    # Step 2: Map prior to log space and create initial log-posterior
    log_prior = jtu.tree_map(log_stable, prior)
    log_q = jtu.tree_map(jnp.zeros_like, prior)

    # Step 3: Iterate until convergence
    def scan_fn(carry: list[Array], t: Array) -> tuple[list[Array], None]:
        log_q = carry
        q = jtu.tree_map(nn.softmax, log_q)
        marginal_ll = all_marginal_log_likelihood(q, log_likelihoods, A_dependencies)
        log_q = jtu.tree_map(add, marginal_ll, log_prior)

        return log_q, None

    res, _ = lax.scan(scan_fn, log_q, jnp.arange(num_iter))

    # Step 4: Map result to factorised posterior
    qs = jtu.tree_map(nn.softmax, res)
    return qs

def mirror_gradient_descent_step(
    tau: float, ln_A: Array, lnB_past: Array, lnB_future: Array, ln_qs: Array
) -> Array:
    """
    u_{k+1} = u_{k} - \nabla_p F_k
    p_k = softmax(u_k)
    """
    err = ln_A - ln_qs + lnB_past + lnB_future
    ln_qs = ln_qs + tau * err
    qs = nn.softmax(ln_qs - ln_qs.mean(axis=-1, keepdims=True))

    return qs

def update_marginals(
    get_messages: Callable[..., tuple[list[Array], list[Array]]],
    obs: list[Array],
    A: list[Array],
    B: list[Array] | None,
    prior: list[Array],
    A_dependencies: list[list[int]],
    B_dependencies: list[list[int]],
    num_iter: int = 1,
    tau: float = 1.0,
    distr_obs: bool = True,
    obs_valid_mask: Array | None = None,
    transition_valid_mask: Array | None = None,
) -> list[Array]:
    """ Version of marginal update that uses a sparse dependency matrix for A """

    T = obs[0].shape[0]
    ln_B = None if B is None else jtu.tree_map(log_stable, B)
    # log likelihoods -> $\ln(A)$ for all time steps
    # for $k > t$ we have $\ln(A) = 0$

    def get_log_likelihood(obs_t: list[Array], A: list[Array]) -> list[Array]:
       # # mapping over batch dimension
       # return vmap(compute_log_likelihood_per_modality)(obs_t, A)
       return compute_log_likelihood_per_modality(obs_t, A, distr_obs=distr_obs)

    # mapping over time dimension of obs array
    log_likelihoods = vmap(get_log_likelihood, (0, None))(obs, A) # this gives a sequence of log-likelihoods (one for each `t`)

    if obs_valid_mask is not None:
        obs_valid_mask = obs_valid_mask.astype(log_likelihoods[0].dtype)
        log_likelihoods = jtu.tree_map(
            lambda ll: ll * obs_valid_mask.reshape((T,) + (1,) * (ll.ndim - 1)),
            log_likelihoods,
        )

    # log marginals -> $\ln(q(s_t))$ for all time steps and factors
    ln_qs = jtu.tree_map( lambda p: jnp.broadcast_to(jnp.zeros_like(p), (T,) + p.shape), prior)

    # log prior -> $\ln(p(s_t))$ for all factors
    ln_prior = jtu.tree_map(log_stable, prior)

    qs = jtu.tree_map(nn.softmax, ln_qs)

    def scan_fn(carry: list[Array], iter: Array) -> tuple[list[Array], None]:
        qs = carry

        ln_qs = jtu.tree_map(log_stable, qs)
        # messages from future $m_+(s_t)$ and past $m_-(s_t)$ for all time steps and factors. For t = T we have that $m_+(s_T) = 0$
        
        lnB_future, lnB_past = get_messages(
            ln_B,
            B,
            qs,
            ln_prior,
            B_dependencies,
            transition_valid_mask=transition_valid_mask,
        )

        mgds = jtu.Partial(mirror_gradient_descent_step, tau)

        ln_As = vmap(all_marginal_log_likelihood, in_axes=(0, 0, None))(qs, log_likelihoods, A_dependencies)

        qs = jtu.tree_map(mgds, ln_As, lnB_past, lnB_future, ln_qs)

        return qs, None

    qs, _ = lax.scan(scan_fn, qs, jnp.arange(num_iter))

    return qs

def variational_filtering_step(
    prior: list[Array], Bs: list[Array], ln_As: list[Array], A_dependencies: list[list[int]]
) -> tuple[list[Array], list[Array], list[Array]]:

    ln_prior = jtu.tree_map(log_stable, prior)
    
    #TODO: put this inside scan
    ####
    marg_ln_As = all_marginal_log_likelihood(prior, ln_As, A_dependencies)

    # compute posterior q(z_t) -> n x 1 x d
    post = jtu.tree_map( 
            lambda x, y: nn.softmax(x + y, -1), marg_ln_As, ln_prior 
        )
    ####

    # compute prediction p(z_{t+1}) = \int p(z_{t+1}|z_t) q(z_t) -> n x d x 1
    pred = jtu.tree_map(
            lambda x, y: jnp.sum(x * jnp.expand_dims(y, -2), -1), Bs, post
        )
    
    # compute reverse conditional distribution q(z_t|z_{t+1})
    cond = jtu.tree_map(
        lambda x, y, z: x * jnp.expand_dims(y, -2) / jnp.expand_dims(z, -1),
        Bs,
        post, 
        pred
    )

    return post, pred, cond

def update_variational_filtering(
    obs: list[Array],
    A: list[Array],
    B: list[Array],
    prior: list[Array],
    A_dependencies: list[list[int]],
    **kwargs: Any,
) -> tuple[list[Array], list[Array], list[Array]]:
    """Online variational filtering belief update that uses a sparse dependency matrix for A"""

    obs[0].shape[0]
    def pad(x: Array) -> Array:
        npad = [(0, 0)] * jnp.ndim(x)
        npad[0] = (0, 1)
        return jnp.pad(x, npad, constant_values=1.)
    
    B = jtu.tree_map(pad, B)
 
    def get_log_likelihood(obs_t: list[Array], A: list[Array]) -> list[Array]:
        # mapping over batch dimension
        return vmap(compute_log_likelihood_per_modality)(obs_t, A)

    # mapping over time dimension of obs array
    log_likelihoods = vmap(get_log_likelihood, (0, None))(obs, A) # this gives a sequence of log-likelihoods (one for each `t`)
    
    def scan_fn(carry: tuple[list[Array], list[Array]], iter: tuple[list[Array], list[Array]]) -> tuple[tuple[list[Array], list[Array]], list[Array]]:
        _, prior = carry
        Bs, ln_As = iter

        post, pred, cond = variational_filtering_step(prior, Bs, ln_As, A_dependencies)
        
        return (post, pred), cond

    init = (prior, prior)
    iterator = (B, log_likelihoods)
    # get q_T(s_t), p_T(s_{t+1}) and the history q_{T}(s_{t}|s_{t+1})q_{T-1}(s_{t-1}|s_{t}) ...
    (qs, ps), qss = lax.scan(scan_fn, init, iterator)

    return qs, ps, qss

def get_vmp_messages(
    ln_B: list[Array] | None,
    B: list[Array] | None,
    qs: list[Array],
    ln_prior: list[Array],
    B_dependencies: list[list[int]],
    transition_valid_mask: Array | None = None,
) -> tuple[list[Array], list[Array]]:
    
    num_factors = len(qs)
    factors = list(range(num_factors))
    get_deps = lambda x, f_idx: [x[f] for f in f_idx] # function that effectively "slices" a list with a set of indices `f_idx`

    # make a list of lists, where each list contains all dependencies of a factor except itself
    all_deps_except_f = jtu.tree_map( 
        lambda f: [d for d in B_dependencies[f] if d != f], 
        factors
    )

    # make list of integers, where each integer is the position of the self-factor in its dependencies list
    position = jtu.tree_map(
        lambda f: B_dependencies[f].index(f),
        factors
    )

    if ln_B is not None:
        ln_B_marg = jtu.tree_map( # this is a list of matrices, where each matrix is the marginal transition tensor for factor f
            lambda b, f: factor_dot(b, get_deps(qs, all_deps_except_f[f]), keep_dims=(0, 1, 2 + position[f])), 
            ln_B, 
            factors
        )  # shape = (T, states_f_{t+1}, states_f_{t})
    else:
        ln_B_marg = None

    T = qs[0].shape[0]
    if transition_valid_mask is None:
        transition_valid_mask = jnp.ones((max(T - 1, 0),), dtype=bool)
    transition_valid_mask = transition_valid_mask.astype(bool)
    start_mask = jnp.concatenate(
        [jnp.ones((1,), dtype=bool), jnp.logical_not(transition_valid_mask)],
        axis=0,
    )

    def forward(ln_b: Array, q: Array, ln_prior: Array) -> Array:
        msg = vmap(lambda x, y: y @ x)(q[:-1], ln_b) # ln_b has shape (num_states, num_states) qs[:-1] has shape (T-1, num_states)
        msg = jnp.where(transition_valid_mask[:, None], msg, 0.0)
        # Append the prior as the t=0 forward message so the result has length T.
        # With window masks, start_mask also resets masked boundary timesteps to the prior.
        msg = jnp.concatenate([jnp.expand_dims(ln_prior, 0), msg], axis=0)
        prior_msg = jnp.broadcast_to(jnp.expand_dims(ln_prior, 0), msg.shape)
        return jnp.where(start_mask[:, None], prior_msg, msg)
    
    def backward(ln_b: Array, q: Array) -> Array:
        # q_i B_ij
        msg = vmap(lambda x, y: x @ y)(q[1:], ln_b)
        msg = jnp.where(transition_valid_mask[:, None], msg, 0.0)
        return jnp.pad(msg, ((0, 1), (0, 0)))

    if ln_B_marg is not None:
        lnB_future = jtu.tree_map(forward, ln_B_marg, qs, ln_prior)
        lnB_past = jtu.tree_map(backward, ln_B_marg, qs)
    else:
        lnB_future = jtu.tree_map(lambda x: 0., qs)
        lnB_past = jtu.tree_map(lambda x: 0., qs)
    
    return lnB_future, lnB_past 

def run_vmp(
    A: list[Array],
    B: list[Array] | None,
    obs: list[Array],
    prior: list[Array],
    A_dependencies: list[list[int]],
    B_dependencies: list[list[int]],
    num_iter: int = 1,
    tau: float = 1.0,
    distr_obs: bool = True,
    obs_valid_mask: Array | None = None,
    transition_valid_mask: Array | None = None,
) -> list[Array]:
    """Run variational message passing over a sequence window.

    Parameters are identical to :func:`run_mmp`.

    Returns
    -------
    list[Array]
        Sequence posterior beliefs per hidden-state factor (same structure as
        :func:`run_mmp`).
    """

    qs = update_marginals(
        get_vmp_messages,
        obs,
        A,
        B,
        prior,
        A_dependencies,
        B_dependencies,
        num_iter=num_iter,
        tau=tau,
        distr_obs=distr_obs,
        obs_valid_mask=obs_valid_mask,
        transition_valid_mask=transition_valid_mask,
    )
    return qs

def get_mmp_messages(
    ln_B: list[Array] | None,
    B: list[Array] | None,
    qs: list[Array],
    ln_prior: list[Array],
    B_deps: list[list[int]],
    transition_valid_mask: Array | None = None,
) -> tuple[list[Array], list[Array]]:
    
    num_factors = len(qs)
    factors = list(range(num_factors))

    get_deps_forw = lambda x, f_idx: [x[f][:-1] for f in f_idx]
    get_deps_back = lambda x, f_idx: [x[f][1:] for f in f_idx]

    T = qs[0].shape[0]
    if transition_valid_mask is None:
        transition_valid_mask = jnp.ones((max(T - 1, 0),), dtype=bool)
    transition_valid_mask = transition_valid_mask.astype(bool)
    start_mask = jnp.concatenate(
        [jnp.ones((1,), dtype=bool), jnp.logical_not(transition_valid_mask)],
        axis=0,
    )
    outgoing_valid = jnp.concatenate(
        [transition_valid_mask, jnp.zeros((1,), dtype=bool)],
        axis=0,
    )

    def forward(b: Array, ln_prior: Array, f: int) -> Array:
        xs = get_deps_forw(qs, B_deps[f])
        dims = tuple((0, 2 + i) for i in range(len(B_deps[f])))
        msg = log_stable(factor_dot_flex(b, xs, dims, keep_dims=(0, 1) ))
        msg = jnp.where(transition_valid_mask[:, None], msg, 0.0)
        # Append the prior as the t=0 forward message so the result has length T.
        # With window masks, start_mask also resets masked boundary timesteps to the prior.
        msg = jnp.concatenate([jnp.expand_dims(ln_prior, 0), msg], axis=0)
        prior_msg = jnp.broadcast_to(jnp.expand_dims(ln_prior, 0), msg.shape)
        msg = jnp.where(start_mask[:, None], prior_msg, msg)
        # In MMP each valid transition contributes via both forward and backward terms.
        # We assign half-weight here (and half in backward below) so each edge is counted once.
        msg = msg * jnp.where(outgoing_valid[:, None], 0.5, 1.0)
        return msg
    
    def backward(Bs: list[Array], xs: list[Array]) -> Array:
        msg = 0.
        for i, b in enumerate(Bs):
            b_norm = b / jnp.clip(b.sum(-1, keepdims=True), min=MINVAL)
            # Complementary half-weight to the forward pass above.
            msg += log_stable(vmap(lambda x, y: y @ x)(b_norm, xs[i])) * .5

        msg = jnp.where(transition_valid_mask[:, None], msg, 0.0)
        return jnp.pad(msg, ((0, 1), (0, 0)))

    def marg(inv_deps: list[int], f: int) -> list[Array]:
        B_marg = []
        for i in inv_deps:
            b = B[i]
            keep_dims = (0, 1, 2 + B_deps[i].index(f))
            dims = []
            idxs = []
            for j, d in enumerate(B_deps[i]):
                if f != d:
                    dims.append((0, 2 + j))
                    idxs.append(d)
            xs = get_deps_forw(qs, idxs)
            B_marg.append( factor_dot_flex(b, xs, tuple(dims), keep_dims=keep_dims) )
        
        return B_marg

    if B is not None:
        inv_B_deps = [[i for i, d in enumerate(B_deps) if f in d] for f in factors]
        B_marg = jtu.tree_map(lambda f: marg(inv_B_deps[f], f), factors)
        lnB_future = jtu.tree_map(forward, B, ln_prior, factors) 
        lnB_past = jtu.tree_map(lambda f: backward(B_marg[f], get_deps_back(qs, inv_B_deps[f])), factors)
    else: 
        lnB_future = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), ln_prior)
        lnB_past = jtu.tree_map(lambda x: 0., qs)

    return lnB_future, lnB_past

def run_mmp(
    A: list[Array],
    B: list[Array] | None,
    obs: list[Array],
    prior: list[Array],
    A_dependencies: list[list[int]],
    B_dependencies: list[list[int]],
    num_iter: int = 1,
    tau: float = 1.0,
    distr_obs: bool = True,
    obs_valid_mask: Array | None = None,
    transition_valid_mask: Array | None = None,
) -> list[Array]:
    """Run marginal message passing over a sequence window.

    Parameters
    ----------
    A : list[Array]
        Model likelihood tensors.
    B : list[Array] | None
        Transition tensors (or `None` for static-state models).
    obs : list[Array]
        Observation sequence per modality.
    prior : list[Array]
        Sequence prior over hidden states.
    A_dependencies : list[list[int]]
        Sparse observation dependencies per modality.
    B_dependencies : list[list[int]]
        Sparse transition dependencies per factor.
    num_iter : int, default=1
        Number of variational update iterations.
    tau : float, default=1.0
        Mirror-descent step size.
    distr_obs : bool, default=True
        Whether observations are already distributional.
    obs_valid_mask : Array | None, optional
        Optional validity mask for padded observation windows.
    transition_valid_mask : Array | None, optional
        Optional validity mask for transitions in padded windows.

    Returns
    -------
    list[Array]
        Sequence posterior beliefs per hidden-state factor.
    """
    qs = update_marginals(
        get_mmp_messages,
        obs,
        A,
        B,
        prior,
        A_dependencies,
        B_dependencies,
        num_iter=num_iter,
        tau=tau,
        distr_obs=distr_obs,
        obs_valid_mask=obs_valid_mask,
        transition_valid_mask=transition_valid_mask,
    )
    return qs

def run_online_filtering(
    A: list[Array],
    B: list[Array],
    obs: list[Array],
    prior: list[Array],
    A_dependencies: list[list[int]],
    num_iter: int = 1,
    tau: float = 1.0,
) -> list[Array]:
    """Runs online filtering (HAVE TO REPLACE WITH OVF CODE)"""
    qs = update_marginals(get_mmp_messages, obs, A, B, prior, A_dependencies, num_iter=num_iter, tau=tau)
    return qs 


# Infer states hybrid & hybrid block

def run_factorized_fpi_hybrid(
    log_likelihoods: list[Array],
    prior: list[Array],
    A_dependencies: list[list[int]],
    num_iter: int,
) -> list[Array]:
    
    log_prior = jtu.tree_map(log_stable, prior)
    log_q = jtu.tree_map(jnp.zeros_like, prior)

    def scan_fn(carry: list[Array], t: Array) -> tuple[list[Array], None]:
        log_q = carry
        q = jtu.tree_map(nn.softmax, log_q)
        marginal_ll = all_marginal_log_likelihood(q, log_likelihoods, A_dependencies)
        log_q = jtu.tree_map(add, marginal_ll, log_prior)
        return log_q, None

    res, _ = lax.scan(scan_fn, log_q, jnp.arange(num_iter))
    qs = jtu.tree_map(nn.softmax, res)
    
    return jtu.tree_map(lambda x: jnp.expand_dims(x, 0), qs)

# Infer states end2end padded

def get_qs_padded(qs: list[Array], max_state_dim: int) -> list[Array]:
    qs_padded = []
    for q in qs:
        qs_padded.append(jnp.zeros((q.shape[0], max_state_dim)).at[:, 0:q.shape[1]].set(q))
    return qs_padded

def compute_qL_marginals(
    lls_padded: Array, qs_padded: list[Array], A_dependencies: list[list[int]], max_state_dim: int
) -> list[list[Array]]:

    dp_dict = {}
    qL_marginals_padded = [[] for i in range(len(A_dependencies))]
    
    for i, A_dep in enumerate(A_dependencies):
    
        if len(A_dep) == 1:
            
            qL_marginals_padded[i].append(lls_padded[i])
    
        else:
            
            for j in range(len(A_dep)):
    
                axes_factors = tuple((axis, factor) for axis, factor in enumerate(A_dep) if j != axis)
    
                existing_key = None
                for sublen in range(len(axes_factors), 0, -1):
                    if axes_factors[:sublen] in dp_dict:
                        existing_key = axes_factors[:sublen]
                        break
    
                new_key = () if existing_key is None else existing_key
                offset = 0 if existing_key is None else len(existing_key)
                curr_prod = lls_padded if existing_key is None else dp_dict[existing_key]
                
                for axis, factor in axes_factors[offset:]:
                    new_key = new_key + ((axis, factor),)
                    q_reshaped = qs_padded[factor].reshape(
                        [1, lls_padded.shape[1]] + [max_state_dim if k == axis else 1 for k in range(lls_padded.ndim - 2)]
                    )
                    curr_prod = curr_prod * q_reshaped
                    dp_dict[new_key] = curr_prod

                qL_marginals_padded[i].append(
                    dp_dict[axes_factors][i].sum(
                        axis=[axis + 1 for axis, _ in axes_factors]
                    )
                )

    return qL_marginals_padded

def qL_flatten(qL_marginals_padded: list[list[Array]]) -> list[list[Array]]:
    
    qL_marginals = []
    for _qs in qL_marginals_padded:
        qL_marginals.append([])
        for _q in _qs:
            idx = (slice(0, _q.shape[0]), slice(0, _q.shape[1])) + tuple(0 for _ in range(_q.ndim - 2))
            qL_marginals[-1].append(_q[idx])
            
    return qL_marginals

def compute_qL_all(
    qL_marginals: list[list[Array]], A_dependencies: list[list[int]], num_factors: int
) -> list[Array]:

    qL_all = [jnp.zeros_like(qL_marginals[0][0])] * num_factors
    
    for m, factor_list_m in enumerate(A_dependencies):
        for l, f in enumerate(factor_list_m):
            qL_all[f] += qL_marginals[m][l]

    return qL_all

def run_factorized_fpi_end2end_padded(
    lls_padded: Array,
    prior: list[Array],
    A_dependencies: list[list[int]],
    max_obs_dim: int,
    max_state_dim: int,
    num_iter: int,
) -> list[Array]:
    
    log_prior = jtu.tree_map(log_stable, prior)
    log_q = jtu.tree_map(jnp.zeros_like, prior)

    def scan_fn(carry: list[Array], t: Array) -> tuple[list[Array], None]:
        
        log_q = carry
        q = jtu.tree_map(nn.softmax, log_q)
        
        qs_padded = get_qs_padded(q, max_state_dim)
        qL_marginals_padded = compute_qL_marginals(lls_padded, qs_padded, A_dependencies, max_state_dim)
        qL_flat = qL_flatten(qL_marginals_padded)
        qL_all = compute_qL_all(qL_flat, A_dependencies, len(q))
        
        log_q = jtu.tree_map(lambda q, lp: q[:, 0:lp.shape[1]] + lp, qL_all, log_prior)
        
        return log_q, None

    res, _ = lax.scan(scan_fn, log_q, jnp.arange(num_iter))
    qs = jtu.tree_map(nn.softmax, res)

    return jtu.tree_map(lambda x: jnp.expand_dims(x, 1), qs)

class FilterMessage(NamedTuple):
    # A: conditional transition-like matrix for the segment
    A: Array      # (..., K, K)
    # log_b: log normalizer term (per left-boundary state)
    log_b: Array  # (..., K)


def _normalize_preserve_zeros(u: Array,
                              axis: int = -1,
                              eps: float = 1e-15) -> Tuple[Array, Array]:
    """
    Normalize along `axis` while preserving exact structural zeros and flooring tiny
    positive values to improve gradient stability in near-degenerate regimes.

    Returns normalized tensor and the (safe) normalization factor with `axis` squeezed.
    """
    eps = jnp.asarray(eps, dtype=u.dtype)
    u_safe = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
    c = jnp.sum(u_safe, axis=axis, keepdims=True)
    c_safe = jnp.where(c == 0, jnp.ones_like(c), c)
    return u_safe / c_safe, jnp.squeeze(c_safe, axis=axis)


def _log_predictive_normalizer(predicted_probs: Array,
                               log_likelihoods: Array) -> Array:
    """
    Compute log c_t = log p(x_t | x_{1:t-1}) in a stable way for each timestep.
    """
    ll_max = jnp.max(log_likelihoods, axis=-1, keepdims=True)
    stable_weights = jnp.exp(log_likelihoods - ll_max)
    return log_stable(jnp.sum(predicted_probs * stable_weights, axis=-1)) + ll_max.squeeze(-1)

def _condition_on(A: Array,
                  ll: Array,
                  axis: int = -1) -> Tuple[Array, Array]:
    """
    Condition a transition-like object A on log-likelihood ll in a numerically stable way.

    Works for:
      - A shape (K,) with ll shape (K,)   (initial probs)
      - A shape (K,K) with ll shape (K,)  (transition rows/cols)
      - and batched versions via broadcasting when ll is 1D.
    """
    ll = jnp.asarray(ll)
    ll_max = jnp.max(ll, axis=-1)
    w = jnp.exp(ll - ll_max)

    if A.ndim == 1:
        A_cond, norm = _normalize_preserve_zeros(A * w, axis=0)
        return A_cond, log_stable(norm) + ll_max

    axis = axis if axis >= 0 else A.ndim + axis
    shape = [1] * A.ndim
    shape[axis] = w.shape[-1]
    w_b = w.reshape(shape)

    A_cond, norm = _normalize_preserve_zeros(A * w_b, axis=axis)
    return A_cond, log_stable(norm) + ll_max


def _hmm_filter_scan_row_oriented(
    initial_probs: Array,
    transition_mats: Array,
    log_likelihoods: Array,
) -> tuple[Array, Array, Array]:
    """
    Core row-oriented HMM filtering via associative scan.

    Parameters
    ----------
    initial_probs:
        Initial state distribution `p(s_0)` with shape `(K,)`.
    transition_mats:
        Transition tensors with row orientation `p(s_{t+1}=j | s_t=i)`.
        Can be `(K, K)` or `(T-1, K, K)`.
    log_likelihoods:
        Log likelihoods `log p(o_t | s_t)` with shape `(T, K)`.

    Returns
    -------
    marginal_loglik:
        Scalar `log p(o_{1:T})`.
    filtered_probs:
        `(T, K)` filtered beliefs `p(s_t | o_{1:t})`.
    predicted_probs:
        `(T, K)` one-step-ahead predictive prior
        `p(s_t | o_{1:t-1})` with `predicted_probs[0] = initial_probs`.
    """
    T, K = log_likelihoods.shape
    if transition_mats.ndim == 2:
        # Use identical transition matrix for all timesteps when stationary.
        transition_mats = jnp.broadcast_to(transition_mats, (T - 1, K, K))

    @vmap
    def _marginalize(
        m_ij: FilterMessage, m_jk: FilterMessage
    ) -> FilterMessage:
        # Combine segment i->j with segment j->k by summing out the intermediate state.
        A_ij_cond, lognorm = _condition_on(m_ij.A, m_jk.log_b)
        A_ik = A_ij_cond @ m_jk.A
        log_b_ik = m_ij.log_b + lognorm
        return FilterMessage(A=A_ik, log_b=log_b_ik)

    A0_vec, log_b0_scalar = _condition_on(initial_probs, log_likelihoods[0])
    A0 = jnp.broadcast_to(A0_vec, (K, K))
    log_b0 = jnp.broadcast_to(log_b0_scalar, (K,))
    A1T, log_b1T = vmap(_condition_on, in_axes=(0, 0))(transition_mats, log_likelihoods[1:])

    partial = lax.associative_scan(
        _marginalize,
        FilterMessage(
            A=jnp.concatenate([A0[None, :, :], A1T], axis=0),
            log_b=jnp.concatenate([log_b0[None, :], log_b1T], axis=0),
        ),
    )

    marginal_loglik = partial.log_b[-1, 0]
    filtered_probs = partial.A[:, 0, :]

    if T == 1:
        predicted_probs = initial_probs[None, :]
    else:
        pred_next = vmap(lambda q, A: q @ A)(filtered_probs[:-1], transition_mats)
        predicted_probs = jnp.concatenate([initial_probs[None, :], pred_next], axis=0)

    return marginal_loglik, filtered_probs, predicted_probs


def hmm_filter_scan_rowstoch(
    initial_probs: Array,
    transition_mats: Array,
    log_likelihoods: Array,
) -> tuple[Array, Array, Array]:
    """
    Exact HMM filtering via `lax.associative_scan` for row-stochastic transitions.

    This variant uses the standard row-stochastic convention
    `transition_mats[t, i, j] = p(z_{t+1}=j | z_t=i)`.

    Parameters
    ------
    initial_probs:
        Initial state distribution `p(s_0)`.
    transition_mats:
        Row-stochastic transitions, stationary `(K, K)` or time-varying
        `(T-1, K, K)`.
    log_likelihoods:
        Sequence log likelihoods with shape `(T, K)`.

    Returns
    -------
    (marginal_loglik, filtered_probs, predicted_probs)
    """
    return _hmm_filter_scan_row_oriented(initial_probs, transition_mats, log_likelihoods)


def _hmm_smoother_scan_row_oriented(
    initial_probs: Array,
    transition_mats: Array,
    log_likelihoods: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """
    Core row-oriented HMM filtering + smoothing via associative scans.

    Computes:
    - filtered marginals using the forward scan
    - smoothed marginals with a suffix scan
    - pairwise transition posteriors `p(s_t, s_{t+1} | o_{1:T})`
    - `p(s_t | s_{t+1}, o_{1:T})` in pymdp-style conditional orientation
    """
    T, K = log_likelihoods.shape
    marginal_loglik, filtered, predicted = _hmm_filter_scan_row_oriented(
        initial_probs, transition_mats, log_likelihoods
    )

    # log-normaliser c_t = p(o_t | o_{1:t-1}), used to stabilize backward factors.
    log_c = _log_predictive_normalizer(predicted, log_likelihoods)
    if T == 1:
        beta = jnp.ones((1, K))
    else:
        # Build backward matrices M_t = A_t * diag(exp(ll_{t+1} - c_{t+1}))
        # that propagate the vector beta_{t+1} -> beta_t.
        col_scale = jnp.exp(log_likelihoods[1:] - log_c[1:][:, None])
        M = transition_mats * col_scale[:, None, :]

        def op(x: Array, y: Array) -> Array:
            # batched matmul works; associative_scan may call this with leading batch dims
            return y @ x

        # suffix-style associative scan via reverse + reverse-back
        beta = lax.associative_scan(op, M[::-1])[::-1]
        beta = vmap(lambda P_t: P_t @ jnp.ones((K,)))(beta)
        beta = jnp.concatenate([beta, jnp.ones((1, K))], axis=0)

    # p(s_t | o_{1:T}) = p(s_t | o_{1:t}) * beta_t
    smoothed = filtered * beta
    smoothed = smoothed / jnp.clip(jnp.sum(smoothed, axis=-1, keepdims=True), min=MINVAL)

    if T == 1:
        trans_probs = jnp.zeros((0, K, K))
        cond_probs = jnp.zeros((0, K, K))
    else:
        xi = (
            filtered[:-1, :, None] *
            transition_mats *
            col_scale[:, None, :] *
            beta[1:, None, :]
        )
        xi = xi / jnp.clip(jnp.sum(xi, axis=(1, 2), keepdims=True), min=MINVAL)
        trans_probs = xi
        cond_probs = xi.transpose(0, 2, 1) / jnp.clip(smoothed[1:, :, None], min=MINVAL)

    return marginal_loglik, filtered, predicted, smoothed, trans_probs, cond_probs


def hmm_smoother_scan_rowstoch(
    initial_probs: Array,
    transition_mats: Array,
    log_likelihoods: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """
    Exact HMM filtering + smoothing via associative scans for row-stochastic transitions.

    Parameters
    ------
    initial_probs:
        Initial distribution `p(s_0)`.
    transition_mats:
        Row-stochastic transitions, stationary `(K, K)` or time-varying `(T-1, K, K)`.
    log_likelihoods:
        `(T, K)` log-likelihood matrix.

    Returns
    -------
    (marginal_loglik, filtered_probs, predicted_probs, smoothed_probs, trans_probs, cond_probs)
    """
    return _hmm_smoother_scan_row_oriented(
        initial_probs, transition_mats, log_likelihoods
    )


def hmm_filter_scan_colstoch(
    initial_probs: Array,
    B_mats: Array,
    log_likelihoods: Array,
) -> tuple[Array, Array, Array]:
    """
    Exact HMM filtering via `lax.associative_scan` for column-stochastic transitions.

    This is the pymdp-native transition orientation:
    `B_mats[t, j, i] = p(z_{t+1}=j | z_t=i)`.

    Parameters
    ------
    initial_probs:
        Initial distribution `p(s_0)`.
    B_mats:
        Column-stochastic transitions, stationary `(K, K)` or time-varying `(T-1, K, K)`.
    log_likelihoods:
        `(T, K)` log-likelihood matrix.

    Returns
    -------
    (marginal_loglik, filtered_probs, predicted_probs)
    """
    return _hmm_filter_scan_row_oriented(
        initial_probs,
        jnp.swapaxes(B_mats, -1, -2),
        log_likelihoods,
    )


def hmm_smoother_scan_colstoch(
    initial_probs: Array,
    B_mats: Array,
    log_likelihoods: Array,
    return_trans_probs: bool = False,
) -> tuple[Any, ...]:
    """
    Exact HMM filtering + smoothing via associative scans for column-stochastic transitions.

    Parameters
    ------
    initial_probs:
        Initial distribution `p(s_0)`.
    B_mats:
        Column-stochastic transitions, stationary `(K, K)` or time-varying `(T-1, K, K)`.
        In this orientation `B_mats[t, j, i] = p(s_{t+1}=j | s_t=i)`.
    log_likelihoods:
        `(T, K)` log-likelihood matrix.
    return_trans_probs:
        If `True`, includes pairwise transition posterior
        `p(s_t, s_{t+1} | o_{1:T})`.

    Returns
    -------
    tuple
        If `return_trans_probs=False`, returns
        `(marginal_loglik, filtered_probs, predicted_probs, smoothed_probs, cond_probs)`.
        If `True`, returns
        `(marginal_loglik, filtered_probs, predicted_probs, smoothed_probs, trans_probs, cond_probs)`.
    """
    smoothed = _hmm_smoother_scan_row_oriented(
        initial_probs,
        jnp.swapaxes(B_mats, -1, -2),
        log_likelihoods,
    )

    if return_trans_probs:
        return smoothed
    return (
        smoothed[0],
        smoothed[1],
        smoothed[2],
        smoothed[3],
        smoothed[5],
    )


def hmm_smoother_from_filtered_colstoch(
    filtered_probs: Array,
    B_mats: Array,
    return_trans_probs: bool = False,
) -> tuple[Any, ...]:
    """
    Exact HMM smoothing from precomputed filtering marginals for column-stochastic transitions.

    This routine is intended for online agent loops where filtering is computed one step at a
    time during observe-infer-act cycles, and backward smoothing is only needed at learning time.

    Args
    ----
    filtered_probs:
        `(T, K)` filtering marginals with `filtered_probs[t] = p(z_t | x_{1:t})`.
    B_mats:
        - `(K_next, K_curr)` stationary column-stochastic transitions, or
        - `(T-1, K_next, K_curr)` time-varying transitions.
    return_trans_probs:
        If `True`, also return pairwise posteriors in `(T-1, K_curr, K_next)` orientation.

    Returns
    -------
    Always returned:
        smoothed_probs: `(T, K)` with `p(z_t | x_{1:T})`.
        joint_next_curr: `(T-1, K_next, K_curr)` with
            `joint_next_curr[t, j, i] = p(z_{t+1}=j, z_t=i | x_{1:T})`.
        cond_probs: `(T-1, K_next, K_curr)` with
            `cond_probs[t, j, i] = p(z_t=i | z_{t+1}=j, x_{1:T})`.
    Additionally returned when `return_trans_probs=True`:
        trans_probs: `(T-1, K_curr, K_next)` with
            `trans_probs[t, i, j] = p(z_t=i, z_{t+1}=j | x_{1:T})`.
    """
    T, K = filtered_probs.shape

    if B_mats.ndim == 2:
        B_mats = jnp.broadcast_to(B_mats, (max(T - 1, 0), K, K))
    elif B_mats.ndim == 3:
        expected_steps = max(T - 1, 0)
        if B_mats.shape[0] != expected_steps:
            raise ValueError(
                f"B_mats has leading dimension {B_mats.shape[0]}, expected {expected_steps} for T={T}"
            )
    else:
        raise ValueError("B_mats must be 2D or 3D")

    if T == 1:
        smoothed = filtered_probs
        joint_next_curr = jnp.zeros((0, K, K))
        cond_probs = jnp.zeros((0, K, K))
        trans_probs = jnp.zeros((0, K, K))
        if return_trans_probs:
            return smoothed, joint_next_curr, cond_probs, trans_probs
        return smoothed, joint_next_curr, cond_probs

    predicted_next = vmap(lambda B_t, q_t: B_t @ q_t)(B_mats, filtered_probs[:-1])  # (T-1, K_next)

    cond_probs = (
        B_mats * filtered_probs[:-1][:, None, :]
    ) / jnp.clip(predicted_next[:, :, None], min=MINVAL)  # (T-1, K_next, K_curr)

    # smoothed_t = cond_t^T @ smoothed_{t+1}
    M = cond_probs.transpose(0, 2, 1)  # (T-1, K_curr, K_next)
    R = M[::-1]

    def op(x: Array, y: Array) -> Array:
        return y @ x

    P_rev = lax.associative_scan(op, R)  # cumulative products from the end
    P = P_rev[::-1]

    q_last = filtered_probs[-1]
    smoothed_prefix = vmap(lambda P_t: P_t @ q_last)(P)  # (T-1, K)
    smoothed = jnp.concatenate([smoothed_prefix, q_last[None, :]], axis=0)
    smoothed = smoothed / jnp.clip(jnp.sum(smoothed, axis=-1, keepdims=True), min=MINVAL)

    joint_next_curr = cond_probs * smoothed[1:, :, None]  # (T-1, K_next, K_curr)
    trans_probs = joint_next_curr.transpose(0, 2, 1)  # (T-1, K_curr, K_next)

    if return_trans_probs:
        return smoothed, joint_next_curr, cond_probs, trans_probs
    return smoothed, joint_next_curr, cond_probs

def run_exact_single_factor_hmm_scan(
    obs: list[Array],
    A: list[Array],
    B: list[Array],
    prior: list[Array],
    actions: Array | None = None,
    distr_obs: bool = True,
) -> tuple[Array, list[Array], list[Array], list[Array]]:
    """
    pymdp-style single-factor wrapper around the column-stochastic scan smoother.

    Notes
    -----
    - `A`, `B`, and `prior` are expected as singleton lists (one hidden-state factor).
    - `B[0]` must be in pymdp-native column-stochastic orientation:
      `(K_next, K_curr[, n_actions])` (no transpose required).
    - Returns `(mll, qs, ps, qss)` with `qss[0]` equal to
      `p(z_t | z_{t+1}, x_{1:T})` in `(T-1, K_next, K_curr)` orientation.
    """
    if len(prior) != 1 or len(B) != 1:
        raise ValueError("run_exact_single_factor_hmm_scan expects singleton lists for prior and B")

    T = obs[0].shape[0]
    pi = prior[0]              # (K,)
    B0 = B[0]                  # (K_next, K_curr[, n_actions]) or (T-1, K_next, K_curr)
    K = pi.shape[0]

    if B0.ndim == 2:
        B_seq = B0
    elif B0.ndim == 3:
        looks_time_varying = B0.shape[0] == max(T - 1, 0) and B0.shape[1] == K and B0.shape[2] == K
        if actions is not None:
            actions = jnp.asarray(actions)
            if actions.ndim == 2:
                if actions.shape[-1] == 1:
                    actions = actions[:, 0]
                else:
                    raise ValueError("single-factor scan expects actions with shape (T-1,) or (T-1, 1)")
            if actions.ndim != 1:
                raise ValueError("single-factor scan expects actions with shape (T-1,) or (T-1, 1)")
            if actions.shape[0] != max(T - 1, 0):
                raise ValueError(
                    f"actions has length {actions.shape[0]} but expected {max(T - 1, 0)} for T={T}"
                )
            B_seq = vmap(lambda u: B0[:, :, u])(actions.astype(jnp.int32))  # (T-1, K_next, K_curr)
        elif looks_time_varying:
            B_seq = B0
        elif T <= 1:
            # No transition is needed when there is only one observation.
            B_seq = B0[:, :, 0]
        else:
            raise ValueError(
                "actions must be provided when B has an action dimension and T > 1"
            )
    else:
        raise ValueError("B[0] must be 2D or 3D")

    # log_likelihoods[t] = log p(o_t | z_t)  => (T, K)
    # vmap over time dimension of obs pytree
    ll = vmap(lambda obs_t: compute_log_likelihood(obs_t, A, distr_obs=distr_obs))(obs)

    mll, q_filt, q_pred, q_smooth, q_t_given_next = hmm_smoother_scan_colstoch(pi, B_seq, ll)

    # Return in pymdp-ish packaging (list per factor)
    qs = [q_smooth]
    ps = [q_pred]
    qss = [q_t_given_next]   # shape (T-1, K_next, K_curr)

    return mll, qs, ps, qss


if __name__ == "__main__":
    prior = [jnp.ones(2)/2, jnp.ones(2)/2, nn.softmax(jnp.array([0, -80., -80., -80, -80.]))]
    obs = [nn.one_hot(0, 5), nn.one_hot(5, 10)]
    A = [jnp.ones((5, 2, 2, 5))/5, jnp.ones((10, 2, 2, 5))/10]
    
    qs = jit(run_vanilla_fpi)(A, obs, prior)

    # test if differentiable

    def sum_prod(prior: list[Array]) -> Array:
        qs = jnp.concatenate(run_vanilla_fpi(A, obs, prior))
        return (qs * log_stable(qs)).sum()

    print(jit(grad(sum_prod))(prior))

    # def sum_prod(precision):
    #     # prior = [jnp.ones(2)/2, jnp.ones(2)/2, nn.softmax(log_prior)]
    #     prior = [jnp.ones(2)/2, jnp.ones(2)/2, nn.softmax(precision*nn.one_hot(0, 5))]
    #     qs = jnp.concatenate(run_vanilla_fpi(A, obs, prior))
    #     return (qs * log_stable(qs)).sum()

    # precis_to_test = 1.
    # print(jit(grad(sum_prod))(precis_to_test))

    # log_prior = jnp.array([0, -80., -80., -80, -80.])
    # print(jit(grad(sum_prod))(log_prior))
