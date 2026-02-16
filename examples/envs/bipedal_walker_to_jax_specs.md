# BipedalWalker JAX Env Spec for `pymdp` `rollout()`

## Goal
Write a JAX-compatible BipedalWalker environment wrapper that can be passed directly into `pymdp.envs.rollout.rollout()` and run under `jit(rollout, static_argnums=(1, 2))` in block-wise episode loops.

## Hard API Contract 
Implement a class inheriting `pymdp.envs.env.Env` with:

1. `reset(self, key, state=None, env_params=None) -> (observation, env_state)`
2. `step(self, key, state, action, env_params=None) -> (observation, env_state)`
3. `generate_env_params(self, key=None, batch_size=None)` (optional but strongly recommended)

Constraints:
- `reset` and `step` must be JIT-able (no Python side effects, no host callbacks, no mutable global state).
- `reset` and `step` must work under `vmap`, since `rollout()` calls `vmap(env.reset)` and `vmap(env.step)`.
- Return trees and array shapes must be static across timesteps.

## Rollout Compatibility Requirements
`rollout()` assumes:
- `env.reset` and `env.step` are callable as `vmap(env.reset)(keys, env_params=...)` and `vmap(env.step)(keys, state, action, env_params=...)`.
- Observation is a pytree (typically list by modality) with fixed structure.
- `env_state` is a pytree that can be carried through `lax.scan`.

For offline learning compatibility, observation leaves should be emitted as one of:
- categorical: `(1, O_m)` per sample (preferred), which becomes `(B, T, 1, O_m)` in rollout history
- or categorical `(O_m)` / discrete index variants accepted by `rollout()` formatting

Preferred for this project:
- categorical one-hot observations with shape `(1, O_m)` per modality.

## Episode/Done Semantics
`rollout()` itself does not early-stop on `done`; done handling is external (block trimming via `env_state.done`).

Environment state must therefore include:
- `done: bool` (JAX scalar per sample)

`step()` must implement non-auto-reset behavior:
- Once `state.done == True`, keep state frozen and keep returning the same observation/state via `lax.select`.
- Update `done_out = state.done OR done_next`.

This matches the CartPole wrapper pattern and is required for correct block trimming and carry reuse.

## Observation Contract for Bipedal
Match the legacy notebook modality layout (18 modalities):

- `obs[0..7]`: discretized continuous values from BipedalWalker observation indices `0..7`
- `obs[8]`: binary contact flag (from raw observation index `8`)
- `obs[9..12]`: discretized continuous values from raw indices `9..12`
- `obs[13]`: binary contact flag (from raw index `13`)
- `obs[14..17]`: discretized executed motor actions (4 joints)

Detailed semantics of each modality group:

- `obs[0..7]` (hull + first leg kinematics), from Gymnasium Bipedal state entries `0..7`:
  - `obs[0]`: hull angle
  - `obs[1]`: normalized hull angular velocity (`2 * hull.angularVelocity / FPS`)
  - `obs[2]`: normalized horizontal velocity (`0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS`)
  - `obs[3]`: normalized vertical velocity (`0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS`)
  - `obs[4]`: first hip joint angle (`joints[0].angle`)
  - `obs[5]`: first hip joint speed (`joints[0].speed / SPEED_HIP`)
  - `obs[6]`: first knee joint angle offset (`joints[1].angle + 1.0`)
  - `obs[7]`: first knee joint speed (`joints[1].speed / SPEED_KNEE`)

- `obs[8]` (binary contact):
  - first lower leg ground-contact flag (`1.0 if legs[1].ground_contact else 0.0`)
  - discretized as a 2-class modality in the notebook pipeline

- `obs[9..12]` (second leg kinematics), from Gymnasium Bipedal state entries `9..12`:
  - `obs[9]`: second hip joint angle (`joints[2].angle`)
  - `obs[10]`: second hip joint speed (`joints[2].speed / SPEED_HIP`)
  - `obs[11]`: second knee joint angle offset (`joints[3].angle + 1.0`)
  - `obs[12]`: second knee joint speed (`joints[3].speed / SPEED_KNEE`)

- `obs[13]` (binary contact):
  - second lower leg ground-contact flag (`1.0 if legs[3].ground_contact else 0.0`)
  - discretized as a 2-class modality in the notebook pipeline

- `obs[14..17]` (action-echo channels; notebook-specific override):
  - `obs[14]`: discretized executed motor command for joint/action channel 0
  - `obs[15]`: discretized executed motor command for joint/action channel 1
  - `obs[16]`: discretized executed motor command for joint/action channel 2
  - `obs[17]`: discretized executed motor command for joint/action channel 3
  - each maps continuous action `[-1, 1]` to discrete bins `[0, action_division-1]`

Important note on Gym default vs notebook:
- Default Gymnasium Bipedal observation has 24 dims: indices `14..23` are 10 lidar fractions.
- The notebook does **not** use those lidar entries in `num_obs`; instead it injects action-echo modalities at `14..17`.
- This spec intentionally follows the notebook semantics (18 modalities) for parity with existing pymdp experiments.

Recommended dimensions:
- `num_obs = [obs_division]*8 + [2] + [obs_division]*4 + [2] + [action_division]*4`
- total modalities = 18

Discretization for continuous modalities:
- Use notebook-compatible bounds by default:
  - `low = observation_space.low / 2`
  - `high = observation_space.high / 2`
  - `dx = (high - low) / (obs_division - 1)`
- Per feature:
  - `idx = round((x - low[i]) / dx[i])`
  - `idx = clip(idx, 0, obs_division - 1).astype(int32)`

Binary modalities (`8`, `13`):
- Convert to int in `{0,1}` and one-hot with 2 classes.

All modalities returned as one-hot categorical vectors `(1, O_m)`.

## Action Contract
Underlying physics action must be 4 continuous motor commands in `[-1, 1]`.

The wrapper must accept discrete actions from `pymdp` and map them to continuous:
- discrete index -> continuous:
  - `a_cont = -1.0 + a_idx * (2.0 / (action_division - 1))`
- clip to `[-1, 1]` after mapping for safety.

Support at least one of these agent-action conventions (choose and document):

1. Multi-factor action input (preferred for clarity):
- `action` contains 4 discrete indices, one per motor.

2. Flattened single-index action input (if using `B_action_dependencies=[[0,1,2,3]]`):
- `action` contains one flattened index.
- Decode to 4 indices using the same ordering as `pymdp` (`index_to_combination` semantics).

The environment must define and document exactly which convention it supports so model config and env wrapper are aligned.

## Reset Behavior
- On `reset`, return initial observation from environment initial state.
- For action-observation modalities (`14..17`) at reset, use a documented default (e.g., zero action index).
- Initialize `done=False`.

## Env Params
Implement `generate_env_params(batch_size=...)` that:
- returns default env parameters unbatched when `batch_size=None`
- broadcasts params to leading batch dimension when `batch_size` is provided

This keeps behavior aligned with existing CartPole wrappers/harnesses.

## Backend Choice Possibilities
- `gymnax` unfortunately does not currently provide a BipedalWalker env, so that's a no-go.
- `brax` could be a suitable alternative, it has MuJuCo-like continuous control tasks like `walker2d` which could be a nice JAX-native alternative
- could re-implement a `BipedalWalker` env from scratch using `Jax2D`.
- `gymnasium` For parity with Kidera's notebook, could use the gymnasium environment and then just jit the infer_and_plan() calls, but keep the environment-action loop an explicit Python for loop. NOTE: To use old gymnasium BipedalWalker env, gymnasium and the Box2D extras need to be installed (`BipedalWalker` import requires `gymnasium[box2d]`).

Implication:
- A pure Gymnax drop-in is not available out of the box.
- If using Brax, add dependency and create an adapter that still satisfies the observation/action contract above.
- If targeting Gymnasium parity as baseline, ensure Box2D dependency is available for reference/testing.

## Acceptance Tests
Implementation is complete only if all pass:

1. JIT/vmap smoke:
- `jit(vmap(env.reset))` works.
- `jit(vmap(env.step))` works with representative states/actions.

2. Rollout integration:
- `run_rollout = jit(rollout, static_argnums=(1, 2))` compiles and runs.
- Block loop with carry reuse works (`initial_carry=last_block` pattern).

3. Done/block correctness:
- `env_state.done` is emitted and used to compute valid steps.
- Post-done freeze works (no state drift after terminal).
