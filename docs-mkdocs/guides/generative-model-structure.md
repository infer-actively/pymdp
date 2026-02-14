# Thinking in Generative Models in `pymdp`

This guide explains how to map Active Inference concepts onto the list-based
model representation used in modern `pymdp`.

## Mental model

Use two independent indexing systems:

1. **Observation modalities** (`m`) index sensory channels.
2. **Hidden-state factors** (`f`) index latent causes.

In code, this means:

- modality-indexed lists: `A[m]`, `observations[m]`, `C[m]`, `pA[m]`
- factor-indexed lists: `D[f]`, `qs[f]`, `B[f]`, `pB[f]`

## Observation modalities: `A[m]` and `observations[m]`

Each modality gets its own likelihood tensor:

- `A[m] = P(o_m | s_dependencies_for_m)`
- `observations[m]` is the actual observation for modality `m`

So if you have two modalities (for example, location and reward), you should
expect `len(A) == 2` and your observation input to have two modality entries.

## Hidden-state factors: `D[f]`, `qs[f]`, `B[f]`

Each hidden-state factor gets its own prior, posterior, and transition model:

- `D[f]`: prior over factor-`f` states
- `qs[f]`: posterior over factor-`f` states
- `B[f] = P(s_f,t+1 | s_dependencies_for_f,t, u_f,t)`

This means you can reason about each factor semantically (for example, context,
location, or object identity) and keep dimensions aligned by factor index.

## Dependencies connect modalities and factors

`pymdp` lets you be explicit about sparse structure:

- `A_dependencies[m]`: which factors modality `m` depends on
- `B_dependencies[f]`: which previous factors influence factor `f` transitions

This is the key to building structured models without forcing every modality to
depend on every factor.

## Minimal indexing example

```python
# Two modalities
# m=0 -> location observation
# m=1 -> reward observation
A = [A_location, A_reward]
obs = [obs_location, obs_reward]

# Two hidden-state factors
# f=0 -> location state
# f=1 -> context state
D = [D_location, D_context]
qs = [qs_location, qs_context]
B = [B_location, B_context]
```

Read this as:

- `A[1]` defines how reward observations are generated.
- `qs[0]` is your current belief over location states.
- `B[1]` defines context dynamics.

## Common shape/indexing pitfalls

1. `len(observations)` does not match `len(A)`.
2. Mixing modality index `m` and factor index `f`.
3. `A_dependencies` or `B_dependencies` indices not matching your list layout.
4. Passing raw integer observations when your setup expects categorical vectors
   (or vice versa).

## How this maps to the agent loop

At each step:

1. infer states with `infer_states(...)` to update `qs[f]`,
2. infer policies with `infer_policies(qs)`,
3. sample actions and propagate beliefs through `B[f]`.

For end-to-end loops, combine this mental model with the
[`rollout()` guide](rollout-active-inference-loop.md).
