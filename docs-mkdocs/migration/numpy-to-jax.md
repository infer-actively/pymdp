# NumPy/legacy to JAX Migration

This guide is for users moving from `pymdp.legacy` (NumPy/object-array style) to modern JAX-first `pymdp`.

## Key concept shifts

| Legacy/NumPy style | Modern JAX style |
|---|---|
| `numpy.ndarray(dtype=object)` collections | pytrees/lists of `jax.Array` |
| `np.random` global RNG | explicit `jr.PRNGKey` threading |
| stateful loops with implicit mutation | functional updates with explicit returns |
| `pymdp.legacy.*` modules | `pymdp.*` modern modules |

## API differences to update

1. Policy inference now uses current beliefs:
```python
# legacy
q_pi, G = agent.infer_policies()

# modern
q_pi, G = agent.infer_policies(qs)
```

2. Stochastic functions require explicit random keys (not just action sampling):
```python
# legacy
action = agent.sample_action()

# modern (stochastic mode)
keys = jr.split(rng_key, agent.batch_size + 1)
action = agent.sample_action(q_pi, rng_key=keys[1:])
```

Common examples in `pymdp` include:
- action/policy sampling (for example `agent.sample_action(..., rng_key=...)`)
- random generative model initialization utilities such as
  `utils.random_A_array(...)`, `utils.random_B_array(...)`, and
  `utils.random_factorized_categorical(...)`
- rollout execution (`rollout(..., rng_key=...)`)
- stochastic environment methods (`env.reset(key, ...)`, `env.step(key, ...)`)

3. Keep observation preprocessing consistent:
- If `categorical_obs=False`, pass discrete indices.
- If `categorical_obs=True`, pass normalized categorical vectors.

## Randomness migration

Use explicit key flow everywhere:

```python
rng_key = jr.PRNGKey(0)
rng_key, key_infer, key_action = jr.split(rng_key, 3)

qs = agent.infer_states(obs, empirical_prior=agent.D)
q_pi, _ = agent.infer_policies(qs)
keys = jr.split(key_action, agent.batch_size + 1)
action = agent.sample_action(q_pi, rng_key=keys[1:])
```

Other frequently used stochastic APIs that also require explicit keys:
- `utils.random_A_array(...)`
- `utils.random_B_array(...)`
- `utils.random_factorized_categorical(...)`
- `utils.generate_agent_spec(..., key=...)`
- `rollout(..., rng_key=...)`
- stochastic `env.reset(key, ...)` / `env.step(key, ...)`

Avoid `np.random` in modern paths.

## Legacy-to-modern worked conversion

```python
# LEGACY
from pymdp.legacy.agent import Agent as LegacyAgent
from pymdp.legacy import utils as legacy_utils
A = legacy_utils.random_A_matrix([3], [2])
B = legacy_utils.random_B_matrix([2], [2])
agent = LegacyAgent(A=A, B=B)
obs = [1]
qs = agent.infer_states(obs)
q_pi, G = agent.infer_policies()
action = agent.sample_action()
```

```python
# MODERN JAX
from jax import random as jr
from pymdp.agent import Agent
from pymdp import utils

key = jr.PRNGKey(0)
keys = jr.split(key, 3)
A = utils.random_A_array(keys[0], [3], [2])
B = utils.random_B_array(keys[1], [2], [2])
agent = Agent(A=A, B=B)
obs = [1]
qs = agent.infer_states(obs, empirical_prior=agent.D)
q_pi, G = agent.infer_policies(qs)
action_keys = jr.split(keys[2], agent.batch_size + 1)
action = agent.sample_action(q_pi, rng_key=action_keys[1:])
```

## Common migration pitfalls

1. Missing `empirical_prior` argument in `infer_states`.
2. Missing `rng_key`/`key` in stochastic calls (for example `sample_action`,
   random model constructors, `rollout`, or stochastic `env.reset`/`env.step`).
3. Mixing `numpy.ndarray` into JAX-only paths.
4. Forgetting sequence action-history shape `(T-1, num_factors)`.

## Migration done checklist

1. No imports from `pymdp.legacy` in active scripts/notebooks.
2. Randomness uses explicit `jr.PRNGKey` flow.
3. `infer_policies(qs)` and `sample_action(q_pi, rng_key=...)` usage updated.
4. Modern tests pass (`pytest test`).
5. Notebook/docs examples run with modern API.
