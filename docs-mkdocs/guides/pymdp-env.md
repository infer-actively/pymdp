# `PymdpEnv`: Building JAX Environments from Generative Models

`pymdp` environments implement the `Env` interface:

- `reset(key, state=None, env_params=None) -> (obs, state)`
- `step(key, state, action, env_params=None) -> (obs, next_state)`

If these methods are JAX/JIT compatible, the environment can be used directly
with [`rollout()`](rollout-active-inference-loop.md) for fast compiled
agent-environment loops.

## Two ways to build an environment

1. Subclass `Env` for fully custom dynamics and observation logic.
2. Use `PymdpEnv` when your environment is a discrete POMDP generative process
   defined by a discrete observation model `A`, discrete transition model `B`, and
   discrete distribution over initial states `D`.

## Using `PymdpEnv`

`PymdpEnv` represents environments with:

- `A`: categorical emission model(s), `P(obs_m | hidden states)`
- `B`: categorical transition model(s), `P(s_{t+1} | s_t, action)`
- `D`: initial hidden-state priors

Sparse structure is controlled with:

- `A_dependencies[m]`
- `B_dependencies[f]`

Minimal construction:

```python
from pymdp.envs.env import make

env, env_params = make(
    A=A,
    B=B,
    D=D,
    A_dependencies=A_dependencies,
    B_dependencies=B_dependencies,
    make_env_params=True,
)
```

`env_params` can be passed to `reset`/`step` for batched or per-run parameter
control, while the `env` instance keeps default `A`, `B`, and `D`.

## Writing a custom `Env`

For non-POMDP or richer simulators, subclass `Env` and implement `reset` and
`step` directly. Keep all randomness explicit through JAX keys (`jax.random`)
and avoid hidden stateful randomness so behavior remains JIT-safe and
reproducible.

## Practical notes

- Prefer array shapes and dependency indices that match your agent model.
- Use `env.generate_env_params(batch_size=...)` when you need batched
  environment parameters.
- If your environment methods are not JIT-compatible, use manual loops instead
  of `rollout()`.
