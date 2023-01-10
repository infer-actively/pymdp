import jax.numpy as jnp
from jax import tree_util, jit, grad, lax, nn

from pymdp.jax.maths import compute_log_likelihood, log_stable, MINVAL

def add(x, y):
    return x + y

def marginal_log_likelihood(qs, log_likelihood, i):
    x = qs[0]
    for q in qs[1:]:
        x = jnp.expand_dims(x, -1) * q

    joint = log_likelihood * x
    dims = (f for f in range(len(qs)) if f != i)
    marg = joint.sum(dims)
    return marg/jnp.clip(qs[i], a_min=MINVAL)

def run_vanilla_fpi(A, obs, prior, num_iter=1):
    """ Vanilla fixed point iteration (jaxified) """

    nf = len(prior)
    factors = list(range(nf))
    # Step 1: Compute log likelihoods for each factor
    ll = compute_log_likelihood(obs, A)
    log_likelihoods = [ll] * nf

    # Step 2: Map prior to log space and create initial log-posterior
    log_prior = tree_util.tree_map(log_stable, prior)
    log_q = tree_util.tree_map(jnp.zeros_like, prior)

    # Step 3: Iterate until convergence
    def scan_fn(carry, t):
        log_q = carry
        q = tree_util.tree_map(nn.softmax, log_q)
        mll = tree_util.Partial(marginal_log_likelihood, q)
        marginal_ll = tree_util.tree_map(mll, log_likelihoods, factors)

        log_q = tree_util.tree_map(add, marginal_ll, log_prior)

        return log_q, None

    res, _ = lax.scan(scan_fn, log_q, jnp.arange(num_iter))

    # Step 4: Map result to factorised posterior
    qs = tree_util.tree_map(nn.softmax, res)
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

