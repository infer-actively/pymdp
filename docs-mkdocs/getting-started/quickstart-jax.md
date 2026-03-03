# Quickstart (JAX)

Here are a couple of lines to quickly build an active inference agent and run state inference, policy inference (planning), and action selection. The agent is created with a random generative model comprising two observation modalities and two hidden state factors.
The first of the two state factors is controllable via a 3 dimensional action.

The agent also has uniform preferences (a flat `C` vector) over the observation modalities.

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

You can also run a quick agent-environment loop with `rollout()` and
`PymdpEnv`. For repeated calls, we recommend wrapping `rollout` in `jit`:

```python
from jax import jit
from jax import random as jr
from pymdp import utils
from pymdp.agent import Agent
from pymdp.envs.env import make
from pymdp.envs.rollout import rollout

key = jr.PRNGKey(1)
key_A, key_B, key_rollout = jr.split(key, 3)

num_obs = [3, 5]
num_states = [3, 2]
num_controls = [3, 1]

A = utils.random_A_array(key_A, num_obs, num_states)
B = utils.random_B_array(key_B, num_states, num_controls)
C = utils.list_array_uniform([[no] for no in num_obs])
D = utils.list_array_uniform([[ns] for ns in num_states])

agent = Agent(A=A, B=B, C=C, D=D, batch_size=1)
env, _ = make(A=A, B=B, D=D)

rollout_jit = jit(rollout, static_argnums=[1, 2])  # env and num_timesteps are static
last, info = rollout_jit(
    agent,
    env,
    20,
    key_rollout,
)

actions = info["action"]
```

For a more in-depth guide to compiled active inference loops, see the dedicated
[`rollout()` guide](../guides/rollout-active-inference-loop.md).
