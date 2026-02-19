# Thinking in Generative Models in `pymdp`

This guide explains how to map Active Inference concepts onto the list-based
model representation used in `pymdp`.

For a model with `F` hidden-state factors and `M` observation modalities over
`T` timesteps, a common joint factorization is:

$$
{\small p\left(o_{1:T}^{1:M}, s_{1:T}^{1:F}, \pi;\,A,B,D\right)
= p(\pi)\underbrace{p\left(s_1^{1:F};\,D\right)}_{\text{initial-state model / }D}
\prod_{t=1}^{T-1}\underbrace{p\left(s_{t+1}^{1:F} \mid s_t^{1:F}, u_t^\pi;\,B\right)}_{\text{transition model / }B}
\prod_{t=1}^{T}\underbrace{p\left(o_t^{1:M} \mid s_t^{1:F};\,A\right)}_{\text{observation model / }A}.}
$$

Here, $\pi$ is a latent policy variable that indexes a full sequence of
actions, and $u_t^\pi$ denotes the action entailed by policy $\pi$ at time $t$.

In `pymdp`, those terms are typically factorized as:

$$
p\left(s_1^{1:F};\,D\right)
= \prod_{f=1}^{F} p\left(s_1^f;\,D_f\right).
$$

$$
p\left(s_{t+1}^{1:F} \mid s_t^{1:F}, u_t^\pi;\,B\right)
= \prod_{f=1}^{F}
p\left(s_{t+1}^f \mid s_t^{B_{\mathrm{dependencies}}[f]}, u_t^\pi;\,B_f\right).
$$

$$
p\left(o_t^{1:M} \mid s_t^{1:F};\,A\right)
= \prod_{m=1}^{M}
p\left(o_t^m \mid s_t^{A_{\mathrm{dependencies}}[m]};\,A_m\right).
$$

## Mental model

Use two independent indexing systems:

1. **Observation modalities** (`m`) index independent sensory channels.
2. **Hidden-state factors** (`f`) index latent causes.

In code, this means:

- modality-indexed lists: `A[m]`, `observations[m]`, `C[m]`, `pA[m]`
- factor-indexed lists: `D[f]`, `qs[f]`, `B[f]`, `pB[f]`

## Observation modalities: `A[m]` and `observations[m]`

Each modality gets its own likelihood tensor:

- `A[m] = P(obs_m | state_i, state_j, state_k)`, if `i, j, k` are in `A_dependencies[m]`
- `observations[m]` is the actual observation for modality `m`

So if you have two modalities (for example, location and reward), you should
expect `len(A) == 2` and your observation input to have two modality entries.

## Hidden-state factors: `D[f]`, `qs[f]`, `B[f]`

Each hidden-state factor gets its own prior, posterior, and transition model:

- `D[f]`: prior over factor-`f` states
- `qs[f]`: posterior over factor-`f` states
- `B[f] = P(s_{[f,t+1]} | state_i, state_j, state_k, action_a, action_b)`, if `i, j, k` are in `B_dependencies[f]` and `a, b` are in `B_action_dependencies[f]`

This means you can reason about each factor semantically (for example, context,
location, or object identity) and keep dimensions aligned by factor index.

## Dependencies connect modalities and factors

`pymdp` lets you be explicit about sparse structure:

- `A_dependencies[m]`: which factors modality `m` depends on
- `B_dependencies[f]`: which previous factors influence factor `f` transitions
- `B_action_dependencies[f]`: which action/control factors influence factor `f` transitions

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
3. `A_dependencies`, `B_dependencies`, or `B_action_dependencies` indices not matching your list layout.
4. Passing raw integer observations when your setup expects categorical vectors
   (or vice versa).

## How this maps to the agent loop

At each step:

1. infer states with `infer_states(...)` to update `qs[f]`,
2. infer policies with `infer_policies(qs)`,
3. sample actions and propagate beliefs through `B[f]`.

For end-to-end loops, combine this mental model with the
[`rollout()` guide](rollout-active-inference-loop.md).
