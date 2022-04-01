import jax.numpy as jnp
from jax import tree_util, jit, lax, nn

from pymdp.jax.maths import compute_log_likelihood, log_stable

def add(x, y):
    return x + y

def marginal_log_likelihood(qs, log_likelihood, i):
    x = qs[0]
    for q in qs[1:]:
        x = x[:, None] * q

    joint = log_likelihood * x
    dims = (f for f in range(len(qs)) if f != i)
    return joint.sum(dims)/qs[i]

def run_vanilla_fpi(A, obs, prior, num_iter=16):
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
    obs = [0, 1, 2]
    A = [jnp.ones((3, 2, 2))/3] * 3
    prior = [jnp.ones(2)/2] * 2
    obs_vec = [nn.one_hot(o_m, A[m].shape[0]) for m, o_m in enumerate(obs)]
    print(jit(run_vanilla_fpi)(A, obs_vec, prior))
    # print(jit(run_vanilla_fpi)(A, obs, prior))

