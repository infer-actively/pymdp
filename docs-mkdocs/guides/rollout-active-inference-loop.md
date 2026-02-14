# Building an Active Inference Loop with `rollout()`

`rollout()` runs repeated inference, planning, action sampling, and environment stepping for a fixed horizon.

## When to use it
- You have a JAX-compatible `Env` and `Agent`.
- You want reproducible multi-step simulations with explicit key management.
- You want optional online learning integrated into the loop.

## Required inputs
- `agent`: `pymdp.agent.Agent`
- `env`: `pymdp.envs.env.Env`-compatible object
- `num_timesteps`: integer horizon
- `rng_key`: JAX PRNG key

Optional:
- `initial_carry`: override initial rollout carry
- `policy_search`: custom policy search function
- `env_params`: batched environment parameters

## Canonical usage

```python
from jax import random as jr
from pymdp.envs.rollout import rollout

rng_key = jr.PRNGKey(0)
last, info = rollout(
    agent,
    env,
    num_timesteps=20,
    rng_key=rng_key,
)
```

## Reproducible key flow

`rollout()` internally splits keys per step and per batch. For deterministic re-runs:
1. keep the same `rng_key` seed,
2. keep environment params/carry identical,
3. avoid hidden non-JAX randomness.

## Batched runs and carry

- `agent.batch_size` controls parallel batch dimension.
- `initial_carry` can override auto-initialized state (for warm-starting).

## Relationship to manual loops

`rollout()` repeatedly applies the one-step helper `infer_and_plan` internally.

Use manual loops when:
- your environment is not JAX-friendly,
- you need custom per-step side effects.

For JAX environments, `rollout()` is usually simpler and less error-prone.

## Debugging checklist

1. Shape mismatch: check observation/action histories and factor dimensions.
2. Stochastic sampling errors: ensure valid RNG keys are threaded.
3. Sequence methods (`mmp`, `vmp`): ensure `past_actions` and valid windows are correct.
4. Learning updates: verify `learn_A`/`learn_B` flags and action alignment.

## Cross-links
- [Quickstart (JAX)](../getting-started/quickstart-jax.md)
- [API: `pymdp.envs.rollout`](../api/envs-rollout.md)
