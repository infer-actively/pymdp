from jax import tree_util, nn, jit
import jax.numpy as jnp

MINVAL = jnp.finfo(float).eps

def log_stable(x):
    return jnp.log(jnp.clip(x, a_min=MINVAL))

def compute_log_likelihood_single_modality(o_m, A_m, distr_obs=True):
    """ Compute observation likelihood for a single modality (observation and likelihood)"""
    if distr_obs:
        expanded_obs = jnp.expand_dims(o_m, tuple(range(1, A_m.ndim)))
        likelihood = (expanded_obs * A_m).sum(axis=0)
    else:
        likelihood = A_m[o_m]
    
    return log_stable(likelihood)

def compute_log_likelihood(obs, A, distr_obs=True):
    """ Compute likelihood over hidden states across observations from different modalities """
    result = tree_util.tree_map(lambda o, a: compute_log_likelihood_single_modality(o, a, distr_obs=distr_obs), obs, A)
    ll = jnp.sum(jnp.stack(result), 0)

    return ll

def compute_log_likelihood_per_modality(obs, A, distr_obs=True):
    """ Compute likelihood over hidden states across observations from different modalities, and return them per modality """
    ll_all = tree_util.tree_map(lambda o, a: compute_log_likelihood_single_modality(o, a, distr_obs=distr_obs), obs, A)

    return ll_all

def compute_accuracy(qs, obs, A):
    """ Compute the accuracy portion of the variational free energy (expected log likelihood under the variational posterior) """

    ll = compute_log_likelihood(obs, A)

    x = qs[0]
    for q in qs[1:]:
        x = jnp.expand_dims(x, -1) * q

    joint = log_likelihood * x
    return joint.sum()

def compute_free_energy(qs, prior, obs, A):
    """ 
    Calculate variational free energy by breaking its computation down into three steps:
    1. computation of the negative entropy of the posterior -H[Q(s)]
    2. computation of the cross entropy of the posterior with the prior H_{Q(s)}[P(s)]
    3. computation of the accuracy E_{Q(s)}[lnP(o|s)] 
    
    Then add them all together -- except subtract the accuracy
    """

    vfe = 0.0 # initialize variational free energy
    for q, p in zip(qs, prior):
        negH_qs = q.dot(log_stable(q))
        xH_qp = -q.dot(log_stable(p))
        vfe += (negH_qs + xH_qp)
    
    vfe -= compute_accuracy(qs, obs, A)

    return vfe

def multidimensional_outer(arrs):
    """ Compute the outer product of a list of arrays by iteratively expanding the first array and multiplying it with the next array """

    x = arrs[0]
    for q in arrs[1:]:
        x = jnp.expand_dims(x, -1) * q

    return x

def spm_wnorm(A):
    """ 
    Returns Expectation of logarithm of Dirichlet parameters over a set of 
    Categorical distributions, stored in the columns of A.
    """
    A = jnp.clip(A, a_min=MINVAL)
    norm = 1. / A.sum(axis=0)
    avg = 1. / A
    wA = norm - avg
    return wA

def dirichlet_expected_value(dir_arr):
    """ 
    Returns Expectation of Dirichlet parameters over a set of 
    Categorical distributions, stored in the columns of A.
    """
    dir_arr = jnp.clip(dir_arr, a_min=MINVAL)
    expected_val = jnp.divide(dir_arr, dir_arr.sum(axis=0, keepdims=True))
    return expected_val

if __name__ == '__main__':
    obs = [0, 1, 2]
    obs_vec = [ nn.one_hot(o, 3) for o in obs]
    A = [jnp.ones((3, 2)) / 3] * 3
    res = jit(compute_log_likelihood)(obs_vec, A)
    
    print(res)