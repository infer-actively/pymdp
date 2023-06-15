import jax.numpy as jnp
from jax import jit, vmap, grad, lax, nn
import jax.tree_util as jtu
# from jax.config import config
# config.update("jax_enable_x64", True)

from pymdp.jax.maths import compute_log_likelihood, compute_log_likelihood_per_modality, log_stable, MINVAL
from typing import Any, List

def add(x, y):
    return x + y

def marginal_log_likelihood(qs, log_likelihood, i):
    if i == 0:
        x = jnp.ones_like(qs[0])
    else: 
        x = qs[0]

    parallel_ndim = len(x.shape[:-1])

    for (f, q) in enumerate(qs[1:]):
        if (f + 1) != i:
            x = jnp.expand_dims(x, -1) * q
        else:
            x = jnp.expand_dims(x, -1) * jnp.ones_like(q)
    
    joint = log_likelihood * x
    dims = (f + parallel_ndim for f in range(len(qs)) if f != i)
    return joint.sum(dims)

def all_marginal_log_likelihood(qs, log_likelihoods, all_factor_lists):
    qL_marginals = jtu.tree_map(lambda ll_m, factor_list_m: mll_factors(qs, ll_m, factor_list_m), log_likelihoods, all_factor_lists)
    
    num_factors = len(qs)

    qL_all = [0.] * num_factors
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

def run_vanilla_fpi(A, obs, prior, num_iter=1):
    """ Vanilla fixed point iteration (jaxified) """

    nf = len(prior)
    factors = list(range(nf))
    # Step 1: Compute log likelihoods for each factor
    ll = compute_log_likelihood(obs, A)
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

def run_factorized_fpi(A, obs, prior, factor_lists, num_iter=1):
    """
    @TODO: Run the sparsity-leveraging fixed point iteration algorithm (jaxified)
    """

    nf = len(prior)
    factors = list(range(nf))
    # Step 1: Compute log likelihoods for each factor
    log_likelihoods = compute_log_likelihood_per_modality(obs, A)

    # Step 2: Map prior to log space and create initial log-posterior
    log_prior = jtu.tree_map(log_stable, prior)
    log_q = jtu.tree_map(jnp.zeros_like, prior)

    # Step 3: Iterate until convergence
    def scan_fn(carry, t):
        log_q = carry
        q = jtu.tree_map(nn.softmax, log_q)
        marginal_ll = all_marginal_log_likelihood(q, log_likelihoods, factor_lists)
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

    err = ln_A + lnB_past + lnB_future - ln_qs
    ln_qs = ln_qs + tau * err
    qs = nn.softmax(ln_qs - ln_qs.mean(axis=-1, keepdims=True))

    return qs

def update_marginals(get_messages, obs, A, B, prior, num_iter=1, tau=1.):

    nf = len(prior)
    T = obs[0].shape[0]
    factors = list(range(nf))
    ln_B = jtu.tree_map(log_stable, B)
    # log likelihoods -> $\ln(A)$ for all time steps
    # for $k > t$ we have $\ln(A) = 0$

    def get_log_likelihood(obs_t, A):

        # mapping over batch dimension
        return vmap(compute_log_likelihood)(obs_t, A)

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
        lnB_past, lnB_future = get_messages(ln_B, B, qs, ln_prior)

        mgds = jtu.Partial(mirror_gradient_descent_step, tau)

        # @TODO: Change to allow factorized updates
        mll = jtu.Partial(marginal_log_likelihood, qs, log_likelihoods)
        ln_As = jtu.tree_map(mll, factors)

        qs = jtu.tree_map(mgds, ln_As, lnB_past, lnB_future, ln_qs)

        return qs, None

    qs, _ = lax.scan(scan_fn, qs, jnp.arange(num_iter))

    return qs

def get_vmp_messages(ln_B, B, qs, ln_prior):
    
    # @vmap(in_axes=(0, 1, 0), out_axes=1)
    def forward(ln_b, q, ln_prior):
        msg = lax.batch_matmul(q[:-1, None], ln_b.transpose(0, 2, 1)).squeeze()
        return jnp.concatenate([jnp.expand_dims(ln_prior, 0), msg], axis=0)
    
    fwd = vmap(forward, in_axes=(0, 1, 0), out_axes=1)

    # @vmap(in_axes=(0, 1), out_axes=1)
    def backward(ln_b, q):
        # q_i B_ij
        msg = lax.batch_matmul(q[1:, None], ln_b).squeeze()
        return jnp.pad(msg, ((0, 1), (0, 0)))
    bkwd = vmap(backward, in_axes=(0, 1), out_axes=1)

    lnB_future = jtu.tree_map(fwd, ln_B, qs, ln_prior)
    lnB_past = jtu.tree_map(bkwd, ln_B, qs)

    return lnB_future, lnB_past

def run_vmp(A, B, obs, prior, blanket_dict, num_iter=1, tau=1.):
    qs = update_marginals(get_vmp_messages, obs, A, B, prior, num_iter=num_iter, tau=tau)
    return qs

def get_mmp_messages(ln_B, B, qs, ln_prior):
    
    def forward(b, q, ln_prior):
        if len(q) > 1:
            msg = lax.batch_matmul(q[:-1, None], b.transpose(0, 2, 1)).squeeze()
            msg = log_stable(msg)
            n = len(msg) 
            if n > 1: # this is the case where there are at least 3 observations. If you have two observations, then you weight the single past message from t = 0 by 1.0
                msg = msg * jnp.pad( 0.5 * jnp.ones(n-1), (0, 1), constant_values=1.)[:, None]
            return jnp.concatenate([jnp.expand_dims(ln_prior, 0), msg], axis=0) # @TODO: look up whether we want to decrease influence of prior by half as well
        else: # this is case where this is a single observation / single-timestep posterior
            return jnp.expand_dims(ln_prior, 0)

    fwd = vmap(forward, in_axes=(0, 1, 0), out_axes=1)
        
    def backward(b, q):
        msg = lax.batch_matmul(q[:-1, None], b.transpose(0, 2, 1)).squeeze()
        msg = log_stable(msg) * 0.5
        return jnp.pad(msg, ((0, 1), (0, 0)))

    bkwd = vmap(backward, in_axes=(0, 1), out_axes=1)

    lnB_future = jtu.tree_map(fwd, B, qs, ln_prior)
    lnB_past = jtu.tree_map(bkwd, B, qs)

    return lnB_future, lnB_past

def run_mmp(A, B, obs, prior, blanket_dict, num_iter=1, tau=1.):
    qs = update_marginals(get_mmp_messages, obs, A, B, prior, num_iter=num_iter, tau=tau)
    return qs

if __name__ == "__main__":
    prior = [jnp.ones(2)/2, jnp.ones(2)/2, nn.softmax(jnp.array([0, -80., -80., -80, -80.]))]
    obs = [nn.one_hot(0, 5), nn.one_hot(5, 10)]
    A = [jnp.ones((5, 2, 2, 5))/5, jnp.ones((10, 2, 2, 5))/10]
    
    qs = jit(run_vanilla_fpi)(A, obs, prior)
    print(qs)

    # test if differentiable
    from functools import partial

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

