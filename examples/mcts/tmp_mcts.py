from functools import partial
from jax import vmap, nn, random as jr, tree_util as jtu
from pymdp.jax.control import compute_expected_state, compute_expected_obs, compute_info_gain, compute_expected_utility

import mctx
import jax.numpy as jnp

@vmap
def compute_neg_efe(agent, qs, action):
  qs_next_pi = compute_expected_state(qs, agent.B, action, B_dependencies=agent.B_dependencies)
  qo_next_pi = compute_expected_obs(qs_next_pi, agent.A, agent.A_dependencies)
  if agent.use_states_info_gain:
    exp_info_gain = compute_info_gain(qs_next_pi, qo_next_pi, agent.A, agent.A_dependencies)
  else:
    exp_info_gain = 0.
  
  if agent.use_utility:
    exp_utility = compute_expected_utility(qo_next_pi, agent.C)
  else:
    exp_utility = 0.
  
  return exp_utility + exp_info_gain, qs_next_pi, qo_next_pi

@partial(vmap, in_axes=(0, 0, None))
def get_prob_single_modality(o_m, po_m, distr_obs):
    """ Compute observation likelihood for a single modality (observation and likelihood)"""
    return jnp.inner(o_m, po_m) if distr_obs else po_m[o_m]
    
def make_aif_recurrent_fn():
  """Returns a recurrent_fn for an AIF agent."""

  def recurrent_fn(agent, rng_key, action, embedding):
    multi_action = agent.policies[action, 0]
    qs = embedding
    neg_efe, qs_next_pi, qo_next_pi = compute_neg_efe(agent, qs, multi_action)

    # recursively branch the policy + outcome tree
    choice = lambda key, po: jr.categorical(key, logits=jnp.log(po))
    if agent.onehot_obs:
      sample = lambda key, po, no: nn.one_hot(choice(key, po), no)
    else:
      sample = lambda key, po, no: choice(key, po)
    
    # set discount to outcome probabilities
    discount = 1.
    obs = []
    for no_m, qo_m in zip(agent.num_obs, qo_next_pi):
      rng_key, key = jr.split(rng_key)
      o_m = sample(key, qo_m, no_m)
      discount *= get_prob_single_modality(o_m, qo_m, agent.onehot_obs)
      obs.append(jnp.expand_dims(o_m, 1))    
    
    qs_next_posterior = agent.infer_states(obs, qs_next_pi)
    # remove time dimension
    # TODO: update infer_states to not expand along time dimension when needed
    qs_next_posterior = jtu.tree_map(lambda x: x.squeeze(1), qs_next_posterior)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=neg_efe,
        discount=discount,
        prior_logits=jnp.log(agent.E),
        value=jnp.zeros_like(neg_efe)
      )

    return recurrent_fn_output, qs_next_posterior

  return recurrent_fn