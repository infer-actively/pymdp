# Quickstart (JAX)

```python
from jax import random as jr
from pymdp import utils
from pymdp.agent import Agent

key = jr.PRNGKey(0)
keys = jr.split(key, 3)

num_obs = [3, 5]
num_states = [3, 2]
num_controls = [3, 1]

A = utils.random_A_array(keys[0], num_obs, num_states)
B = utils.random_B_array(keys[1], num_states, num_controls)
C = utils.list_array_uniform([[no] for no in num_obs])

agent = Agent(A=A, B=B, C=C)

# Discrete observation indices for each modality
obs = [1, 2]

# Use agent.D as the initial empirical prior
qs = agent.infer_states(obs, empirical_prior=agent.D)
q_pi, G = agent.infer_policies(qs)

sample_keys = jr.split(keys[2], agent.batch_size + 1)
action = agent.sample_action(q_pi, rng_key=sample_keys[1:])
```

For sequence/looped execution, see the dedicated [`rollout()` guide](../guides/rollout-active-inference-loop.md).
