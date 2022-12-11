from jax import tree_util, nn, jit
import jax.numpy as jnp

MIN_VAL = 1e-16 # to debug weird inference with FPI, which we encountered with the T-Maze, try uncommenting this / commenting out the 1e-32 below
# MIN_VAL = 1e-32

def log_stable(x):

    return jnp.log(jnp.where(x >= MIN_VAL, x, MIN_VAL))

def compute_log_likelihood_single_modality(o_m, A_m):
    """ Compute observation likelihood for a single modality (observation and likelihood)"""
    expanded_obs = jnp.expand_dims(o_m, tuple(range(1, A_m.ndim)))
    likelihood = (expanded_obs * A_m).sum(axis=0, keepdims=True).squeeze()
    
    return log_stable(likelihood)

def compute_log_likelihood(obs, A):
    """ Compute likelihood over hidden states across observations from different modalities """
    result = tree_util.tree_map(compute_log_likelihood_single_modality, obs, A)

    ll = jnp.sum(jnp.stack(result), 0)

    return ll

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

if __name__ == '__main__':
    obs = [0, 1, 2]
    obs_vec = [ nn.one_hot(o, 3) for o in obs]
    A = [jnp.ones((3, 2)) / 3] * 3
    res = jit(compute_log_likelihood)(obs_vec, A)
    
    print(res)