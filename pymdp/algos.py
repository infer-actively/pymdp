import jax.numpy as jnp
import jax.tree_util as jtu

from jax import jit, vmap, grad, lax, nn
# from jax.config import config
# config.update("jax_enable_x64", True)

from pymdp.maths import compute_log_likelihood, compute_log_likelihood_per_modality, log_stable, factor_dot, factor_dot_flex
from typing import List

def add(x, y):
    return x + y

def marginal_log_likelihood(qs, log_likelihood, i):
    xs = [q for j, q in enumerate(qs) if j != i]
    return factor_dot(log_likelihood, xs, keep_dims=(i,))

def all_marginal_log_likelihood(qs, log_likelihoods, all_factor_lists):
    qL_marginals = jtu.tree_map(lambda ll_m, factor_list_m: mll_factors(qs, ll_m, factor_list_m), log_likelihoods, all_factor_lists)
    
    num_factors = len(qs)

    # insted of a double loop we could have a list defining m to f mapping
    # which could be resolved with a single tree_map cast
    qL_all = [jnp.zeros(1)] * num_factors
    for m, factor_list_m in enumerate(all_factor_lists):
        for l, f in enumerate(factor_list_m):
            qL_all[f] += qL_marginals[m][l]

    return qL_all

def mll_factors(qs, ll_m, factor_list_m) -> List:
    relevant_factors = [qs[f] for f in factor_list_m]
    marginal_ll_f = jtu.Partial(marginal_log_likelihood, relevant_factors, ll_m)
    loc_nf = len(factor_list_m)
    loc_factors = list(range(loc_nf))
    return jtu.tree_map(marginal_ll_f, loc_factors)

def run_vanilla_fpi(A, obs, prior, num_iter=1, distr_obs=True):
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
    def scan_fn(carry, t):
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

def run_factorized_fpi(A, obs, prior, A_dependencies, num_iter=1):
    """
    Run the fixed point iteration algorithm with sparse dependencies between factors and outcomes (stored in `A_dependencies`)
    """

    # Step 1: Compute log likelihoods for each factor
    log_likelihoods = compute_log_likelihood_per_modality(obs, A)

    # Step 2: Map prior to log space and create initial log-posterior
    log_prior = jtu.tree_map(log_stable, prior)
    log_q = jtu.tree_map(jnp.zeros_like, prior)

    # Step 3: Iterate until convergence
    def scan_fn(carry, t):
        log_q = carry
        q = jtu.tree_map(nn.softmax, log_q)
        marginal_ll = all_marginal_log_likelihood(q, log_likelihoods, A_dependencies)
        log_q = jtu.tree_map(add, marginal_ll, log_prior)

        return log_q, None

    res, _ = lax.scan(scan_fn, log_q, jnp.arange(num_iter))

    # Step 4: Map result to factorised posterior
    qs = jtu.tree_map(nn.softmax, res)
    return qs

def mirror_gradient_descent_step(tau, ln_A, lnB_past, lnB_future, ln_qs):
    """
    u_{k+1} = u_{k} - \nabla_p F_k
    p_k = softmax(u_k)
    """
    err = ln_A - ln_qs + lnB_past + lnB_future
    ln_qs = ln_qs + tau * err
    qs = nn.softmax(ln_qs - ln_qs.mean(axis=-1, keepdims=True))

    return qs

def update_marginals(get_messages, obs, A, B, prior, A_dependencies, B_dependencies, num_iter=1, tau=1.,):
    """ Version of marginal update that uses a sparse dependency matrix for A """

    T = obs[0].shape[0]
    ln_B = jtu.tree_map(log_stable, B)
    # log likelihoods -> $\ln(A)$ for all time steps
    # for $k > t$ we have $\ln(A) = 0$

    def get_log_likelihood(obs_t, A):
       # # mapping over batch dimension
       # return vmap(compute_log_likelihood_per_modality)(obs_t, A)
       return compute_log_likelihood_per_modality(obs_t, A)

    # mapping over time dimension of obs array
    log_likelihoods = vmap(get_log_likelihood, (0, None))(obs, A) # this gives a sequence of log-likelihoods (one for each `t`)

    # log marginals -> $\ln(q(s_t))$ for all time steps and factors
    ln_qs = jtu.tree_map( lambda p: jnp.broadcast_to(jnp.zeros_like(p), (T,) + p.shape), prior)

    # log prior -> $\ln(p(s_t))$ for all factors
    ln_prior = jtu.tree_map(log_stable, prior)

    qs = jtu.tree_map(nn.softmax, ln_qs)

    def scan_fn(carry, iter):
        qs = carry

        ln_qs = jtu.tree_map(log_stable, qs)
        # messages from future $m_+(s_t)$ and past $m_-(s_t)$ for all time steps and factors. For t = T we have that $m_+(s_T) = 0$
        
        lnB_past, lnB_future = get_messages(ln_B, B, qs, ln_prior, B_dependencies)

        mgds = jtu.Partial(mirror_gradient_descent_step, tau)

        ln_As = vmap(all_marginal_log_likelihood, in_axes=(0, 0, None))(qs, log_likelihoods, A_dependencies)

        qs = jtu.tree_map(mgds, ln_As, lnB_past, lnB_future, ln_qs)

        return qs, None

    qs, _ = lax.scan(scan_fn, qs, jnp.arange(num_iter))

    return qs

def variational_filtering_step(prior, Bs, ln_As, A_dependencies):

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

def update_variational_filtering(obs, A, B, prior, A_dependencies, **kwargs):
    """Online variational filtering belief update that uses a sparse dependency matrix for A"""

    obs[0].shape[0]
    def pad(x):
        npad = [(0, 0)] * jnp.ndim(x)
        npad[0] = (0, 1)
        return jnp.pad(x, npad, constant_values=1.)
    
    B = jtu.tree_map(pad, B)
 
    def get_log_likelihood(obs_t, A):
        # mapping over batch dimension
        return vmap(compute_log_likelihood_per_modality)(obs_t, A)

    # mapping over time dimension of obs array
    log_likelihoods = vmap(get_log_likelihood, (0, None))(obs, A) # this gives a sequence of log-likelihoods (one for each `t`)
    
    def scan_fn(carry, iter):
        _, prior = carry
        Bs, ln_As = iter

        post, pred, cond = variational_filtering_step(prior, Bs, ln_As, A_dependencies)
        
        return (post, pred), cond

    init = (prior, prior)
    iterator = (B, log_likelihoods)
    # get q_T(s_t), p_T(s_{t+1}) and the history q_{T}(s_{t}|s_{t+1})q_{T-1}(s_{t-1}|s_{t}) ...
    (qs, ps), qss = lax.scan(scan_fn, init, iterator)

    return qs, ps, qss

def get_vmp_messages(ln_B, B, qs, ln_prior, B_dependencies):
    
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

    def forward(ln_b, q, ln_prior):
        msg = vmap(lambda x, y: y @ x)(q[:-1], ln_b) # ln_b has shape (num_states, num_states) qs[:-1] has shape (T-1, num_states)
        return jnp.concatenate([jnp.expand_dims(ln_prior, 0), msg], axis=0)
    
    def backward(ln_b, q):
        # q_i B_ij
        msg = vmap(lambda x, y: x @ y)(q[1:], ln_b)
        return jnp.pad(msg, ((0, 1), (0, 0)))

    if ln_B_marg is not None:
        lnB_future = jtu.tree_map(forward, ln_B_marg, qs, ln_prior)
        lnB_past = jtu.tree_map(backward, ln_B_marg, qs)
    else:
        lnB_future = jtu.tree_map(lambda x: 0., qs)
        lnB_past = jtu.tree_map(lambda x: 0., qs)
    
    return lnB_future, lnB_past 

def run_vmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=1, tau=1.):
    '''
    Run variational message passing (VMP) on a sequence of observations
    '''

    qs = update_marginals(
        get_vmp_messages, 
        obs, 
        A, 
        B, 
        prior, 
        A_dependencies, 
        B_dependencies, 
        num_iter=num_iter, 
        tau=tau
    )
    return qs

def get_mmp_messages(ln_B, B, qs, ln_prior, B_deps):
    
    num_factors = len(qs)
    factors = list(range(num_factors))

    get_deps_forw = lambda x, f_idx: [x[f][:-1] for f in f_idx]
    get_deps_back = lambda x, f_idx: [x[f][1:] for f in f_idx]

    def forward(b, ln_prior, f):
        xs = get_deps_forw(qs, B_deps[f])
        dims = tuple((0, 2 + i) for i in range(len(B_deps[f])))
        msg = log_stable(factor_dot_flex(b, xs, dims, keep_dims=(0, 1) ))
        # append log_prior as a first message 
        msg = jnp.concatenate([jnp.expand_dims(ln_prior, 0), msg], axis=0)
        # mutliply with 1/2 all but the last msg
        T = len(msg)
        if T > 1:
            msg = msg * jnp.pad( 0.5 * jnp.ones(T - 1), (0, 1), constant_values=1.)[:, None]

        return msg
    
    def backward(Bs, xs):
        msg = 0.
        for i, b in enumerate(Bs):
            b_norm = b / (b.sum(-1, keepdims=True) + 1e-16)
            msg += log_stable(vmap(lambda x, y: y @ x)(b_norm, xs[i])) * .5
        
        return jnp.pad(msg, ((0, 1), (0, 0)))

    def marg(inv_deps, f):
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

def run_mmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=1, tau=1.):
    qs = update_marginals(
        get_mmp_messages, 
        obs, 
        A, 
        B, 
        prior, 
        A_dependencies, 
        B_dependencies, 
        num_iter=num_iter, 
        tau=tau
    )
    return qs

def run_online_filtering(A, B, obs, prior, A_dependencies, num_iter=1, tau=1.):
    """Runs online filtering (HAVE TO REPLACE WITH OVF CODE)"""
    qs = update_marginals(get_mmp_messages, obs, A, B, prior, A_dependencies, num_iter=num_iter, tau=tau)
    return qs 

if __name__ == "__main__":
    prior = [jnp.ones(2)/2, jnp.ones(2)/2, nn.softmax(jnp.array([0, -80., -80., -80, -80.]))]
    obs = [nn.one_hot(0, 5), nn.one_hot(5, 10)]
    A = [jnp.ones((5, 2, 2, 5))/5, jnp.ones((10, 2, 2, 5))/10]
    
    qs = jit(run_vanilla_fpi)(A, obs, prior)

    # test if differentiable

    def sum_prod(prior):
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
