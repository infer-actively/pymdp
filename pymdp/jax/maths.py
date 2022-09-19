from jax import tree_util, nn, jit
import jax.numpy as jnp

MIN_VAL = 1e-32

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

if __name__ == '__main__':
    obs = [0, 1, 2]
    obs_vec = [ nn.one_hot(o, 3) for o in obs]
    A = [jnp.ones((3, 2)) / 3] * 3
    res = jit(compute_log_likelihood)(obs_vec, A)
    
    print(res)