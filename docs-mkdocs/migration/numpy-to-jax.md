# NumPy/legacy to JAX Migration

This guide is for users moving from `pymdp.legacy` (NumPy/object-array style) to modern JAX-first `pymdp`.

## Key concept shifts

| Legacy/NumPy style | Modern JAX style |
|---|---|
| `numpy.ndarray(dtype=object)` collections | pytrees/lists of `jax.Array` |
| `np.random` global RNG | explicit `jr.PRNGKey` threading |
| stateful loops with implicit mutation | functional updates with explicit returns |
| mutable attribute assignment (`agent.x = ...`) | functional pytree updates (`eqx.tree_at(...)`) |
| `pymdp.legacy.*` modules | `pymdp.*` modern modules |

## API differences to update

1. Policy inference now takes current beliefs as inputs, since they are no longer stored internally on the agent.

        # legacy
        q_pi, G = agent.infer_policies()

        # modern
        q_pi, G = agent.infer_policies(qs)

2. Stochastic functions require explicit random keys. Common examples in
   `pymdp` include action/policy sampling (for example
   `agent.sample_action(..., rng_key=...)`), random generative model
   initialization utilities such as `utils.random_A_array(...)`,
   `utils.random_B_array(...)`, and `utils.random_factorized_categorical(...)`,
   rollout execution (`rollout(..., rng_key=...)`), and stochastic
   environment methods (`env.reset(key, ...)`, `env.step(key, ...)`).

        # legacy
        action = agent.sample_action()

        # modern (stochastic mode)
        keys = jr.split(rng_key, agent.batch_size + 1)
        action = agent.sample_action(q_pi, rng_key=keys[1:])

3. Keep observation preprocessing consistent.
    - If `categorical_obs=False`, pass discrete indices.
    - If `categorical_obs=True`, pass normalized categorical vectors.

## Batching and `batch_size`

Most of `pymdp`'s JAX APIs expecting a leading `batch_size` dimension to most input arrays (e.g., arrays of state and parameter posteriors and priors). This allows for easy parallelization using [`jax.vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html) to automatically run multiple active inference agents/processes in parallel.

When migrating from legacy single-agent code, keep these points in mind:

- Multi-agent simulations (this could also be running multiple parameterizations of one agent, in parallel) requires batched per-agent information (observations, beliefs, parameters, preferences, some hyperparameters).
- If you pass unbatched model arrays to `Agent(..., batch_size=N)`, they are
  broadcast internally to `N` copies of the input model parameters. 
- Even in the default single-agent case (`batch_size=1`), most arrays and
  outputs are represented with a leading singleton dimension.
- This is why you will often see shapes that begin with `(1, ...)` in the JAX-based
  code paths.

Typical multi-agent setup pattern:

```python
batch_size = 3
agent = Agent(A=A, B=B, C=C, D=D, batch_size=batch_size)

# One discrete modality for 3 parallel agents: shape (batch_size, 1)
obs = [jnp.array([[0], [2], [3]])]
qs = agent.infer_states(obs, empirical_prior=agent.D)
```

For environment-side batching, either:
- keep a single environment and run batched agents against shared `env.A/B/D`,
  or
- pass batched `env_params` (for example via
  `env.generate_env_params(batch_size=...)`) when using `rollout()`. The parameters inside `env_params` must also carry a leading `batch_size` dimension that is matched to each set of parameters in the `Agent` class.

## Updating `Agent` fields in JAX mode

The `Agent` class is an [Equinox](https://github.com/patrick-kidger/equinox)
module. In practice, this means you should avoid mutable-style setter updates
like:

```python
agent.A = new_A
agent.B = new_B
```

Instead, use Equinox-style functional updates with `eqx.tree_at(...)`:

```python
import equinox as eqx

agent = eqx.tree_at(lambda x: (x.A,), agent, (new_A,))
agent = eqx.tree_at(lambda x: (x.B,), agent, (new_B,))
```

This is the pattern used near the end of
`infer_parameters()` in `pymdp/agent.py`, where after a learning update, the new `A`/`B` (and, when
enabled, `I`) arrays are written back into a new `Agent` instance.

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
