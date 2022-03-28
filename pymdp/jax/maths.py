from jax import tree_util, nn, jit
import jax.numpy as jnp

MIN_VAL = -100

def log(x):

    return jnp.where(x > 0, jnp.log(x), MIN_VAL)

def compute_log_likelihood_single_modality(o_m, A_m):
    """ Compute observation likelihood for a single modality (observation and likelihood)"""
    expanded_obs = jnp.expand_dims(nn.one_hot(o_m, A_m.shape[0]), tuple(range(1,A_m.ndim)))
    likelihood = (expanded_obs * A_m).sum(axis=0, keepdims=True).squeeze()
    return log(likelihood)

def compute_log_likelihood(obs, A):
    """ Compute likelihood over hidden states across observations from different modalities """
    result = tree_util.tree_map(compute_log_likelihood_single_modality, obs, A)

    ll = jnp.sum(jnp.stack(result), 0) # if no-logging

    return ll

if __name__ == '__main__':
    obs = [0, 1, 2]
    A = [jnp.ones((3, 2)) / 3] * 3
    res = jit(compute_log_likelihood)(obs, A)
    print(res)