import jax.numpy as jnp
from jax import jit, grad, lax, nn
import jax.tree_util as jtu
# from jax.config import config
# config.update("jax_enable_x64", True)

from pymdp.jax.maths import compute_log_likelihood, compute_log_likelihood_per_modality, log_stable, MINVAL

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
        # marginal_ll = jtu.tree_map(mll, log_likelihoods, factors)
        log_q = jtu.tree_map(add, marginal_ll, log_prior)

        return log_q, None

    res, _ = lax.scan(scan_fn, log_q, jnp.arange(num_iter))

    # Step 4: Map result to factorised posterior
    qs = jtu.tree_map(nn.softmax, res)
    return qs

def run_factorized_fpi(A, obs, prior, blanket_dict, num_iter=1):
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
        mll = jtu.Partial(marginal_log_likelihood, q)
        marginal_ll = jtu.tree_map(mll, log_likelihoods, factors)

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
    T = obs.shape[0]
    factors = list(range(nf))
    ln_B = jtu.tree_map(log_stable, B)
    # log likelihoods -> $\ln(A)$ for all time steps
    # for $k > t$ we have $\ln(A) = 0$

    log_likelihoods = vmap(compute_log_likelihood, (0, None))(obs, A) # this gives a sequence of log-likelihoods (one for each `t`)

    # log marginals -> $\ln(q(s_t))$ for all time steps and factors
    ln_qs = jtu.tree_map( lambda p: jnp.broadcast_to(jnp.zeros_like(p), (T,) + p.shape), prior)

    qs = jtu.tree_map(nn.softmax, ln_qs)

    def scan_fn(carry, iter):
        qs = carry

        ln_qs = jtu.tree_map(log_stable, qs)
        # messages from future $m_+(s_t)$ and past $m_-(s_t)$ for all time steps and factors. For t = T we have that $m_+(s_T) = 0$
        lnB_past, lnB_future = get_messages(ln_B, B, qs)

        mgds = partial(mirror_gradient_descent_step, tau)

        mll = vmap(jtu.Partial(marginal_log_likelihood, qs, log_likelihoods), ((None, 0, 1, None), 0)) 
        ln_As = jtu.tree_map(mll, factors)

        qs = jtu.tree_map(mgds, ln_As, lnB_past, lnB_future, ln_qs)

        return qs, None

    qs, _ = lax.scan(scan_fn, qs, jnp.arange(num_iter))

    # Step 4: Map result to factorised posterior
    # qs = jtu.tree_map(nn.softmax, res)
    return qs

def run_vmp(A, obs, prior, blanket_dict, num_iter=1):

    qs = update_marginals(get_vmp_messages, num_iter=num_iter)

def run_mmp(A, obs, prior, blanket_dict, num_iter=1):

    qs = update_marginals(get_mmp_messages, num_iter=num_iter)

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

