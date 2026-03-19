# Using `rollout()` for compiled active inference loops

`rollout()` runs repeated inference, planning, action sampling, and environment stepping for a fixed horizon.
Internally, it uses `jax.lax.scan`, so direct calls are valid and JAX-friendly.

## When to use it
- The `.step()` and `.reset()` methods of your `Env` can be JIT-compiled.
  This includes compatibility with `pymdp`'s native `PymdpEnv`, as well as existing JAX RL environment frameworks (see for example [gymnax](https://github.com/RobertTLange/gymnax), [jumanji](https://github.com/instadeepai/jumanji), and [navix](https://github.com/epignatelli/navix)).
- You want high-throughput simulations by compiling the full closed-loop interaction once and executing it efficiently across many rollouts.
- You want a single, consistent API for multi-step active inference rollouts with explicit PRNG key threading.

## Required inputs
- `agent`: `pymdp.agent.Agent`
- `env`: `pymdp.envs.env.Env`-compatible object
- `num_timesteps`: integer horizon
- `rng_key`: JAX PRNG key

## Optional inputs
- `initial_carry`: override initial rollout carry
- `policy_search`: custom policy search function
- `env_params`: batched environment parameters

## Canonical usage

```python
from jax import random as jr
from pymdp.envs.rollout import rollout

rng_key = jr.PRNGKey(0)
last, info = rollout(agent, env, 20, rng_key)
```

For repeated calls with fixed `env` and `num_timesteps`, we recommend wrapping
`rollout` with `jit` so XLA can cache the compiled program:

```python
from jax import jit
from jax import random as jr
from pymdp.envs.rollout import rollout

rng_key = jr.PRNGKey(0)
rollout_jit = jit(rollout, static_argnums=[1, 2])  # env and num_timesteps are static

last, info = rollout_jit(
    agent,
    env,
    20,
    rng_key,
)
```

## Reproducible key flow

`rollout()` internally splits keys per step and per batch. For deterministic re-runs:

1. pass in the same `rng_key` seed,
2. keep environment params/initial carry identical,
3. avoid hidden non-JAX randomness inside your environment's `.step()` or `.reset()` methods.

## Batched runs and carry

- `agent.batch_size` controls parallel batch dimension.
- `initial_carry` can override auto-initialized state (for warm-starting).

## Relationship to manual loops

`rollout()` repeatedly applies the one-step helper `infer_and_plan` internally
using `jax.lax.scan`.

Use manual loops when:

- your environment's `.step()` and `.reset()` methods cannot be JITTed.
- you need custom per-step side effects that don't respect the active inference logic of `infer_and_plan`.

For JAX-based environments, we recommend using `rollout()`, as it's usually simpler and less error-prone.

## Debugging checklist

1. Shape mismatch: check observation/action histories and factor dimensions.
2. Stochastic sampling errors: ensure valid RNG keys are threaded.
3. Sequence methods (`mmp`, `vmp`): ensure `past_actions` and valid windows are correct.
4. Learning updates: verify `learn_A`/`learn_B` flags and action alignment.

## Cross-links
- [Quickstart (JAX)](../getting-started/quickstart-jax.md)
- [API: `pymdp.envs.rollout`](../api/envs-rollout.md)
