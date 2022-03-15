from jax import tree_util
import jax.numpy as jnp

def compute_likelihood_single_modality(o_m, A_m):
    """ Compute observation likelihood for a single modality (observation and likelihood)"""
    expanded_obs = jnp.expand_dims(o_m, tuple(range(1,A_m.ndim)))
    likelihood = (expanded_obs * A_m).sum(axis=0, keepdims=True).squeeze()
    # return jnp.log(likelihood)
    return likelihood

def compute_likelihood(obs, A):
    """ Compute likelihood over hidden states across observations from different modalities """
    result = tree_util.tree_map(compute_likelihood_single_modality, obs, A)

    # log_likelihood = jnp.stack(result, axis = 0).sum(axis=0) # if likelihoods were already logged at the single modality level
    likelihood = jnp.prod(jnp.stack(result, axis = 0), axis=0) # if no-logging

    return likelihood
