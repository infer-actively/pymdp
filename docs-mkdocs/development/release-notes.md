# Release Notes

## 1.0.0

`pymdp` 1.0.0 is the first major release since the `0.0.7` line. The small
`v0.0.7.1` patch was primarily packaging-oriented; the substantive story of
this release is that `pymdp` is now JAX-first. The primary docs, examples,
testing surface, and supported workflows now center on accelerator-friendly,
batched, differentiable active inference, while the previous NumPy/object-array
implementation is preserved under `pymdp.legacy`.

Alongside the backend transition, this release adds a much richer environment
and rollout workflow, upgraded inference and planning APIs, variational free
energy diagnostics, probabilistic model-fitting workflows built around
`pybefit` and NumPyro, a JAX-first documentation site, and release-grade
testing and notebook execution gates.

### Breaking Changes And Migration

If you are upgrading from the legacy NumPy backend, start with the
[NumPy/legacy to JAX migration guide](../migration/numpy-to-jax.md).

- `pymdp.*` now refers to the modern JAX implementation. Legacy NumPy code is
  still available, but it lives under `pymdp.legacy.*`.
- The `Agent` API is now explicitly JAX- and batch-oriented. In particular,
  `batch_size` replaces the older `apply_batch` naming, and many public
  surfaces now assume a leading batch axis.
- The `Agent`'s policy set is now stored as a dedicated Equinox module rather
  than a raw ndarray. In practice, `self.policies` is now a structured
  container, and the underlying policy array lives under
  `self.policies.policy_arr`.
- Policy scoring is now consistently exposed as `neg_efe = -EFE`. In practice,
  `agent.infer_policies(qs)` returns `(q_pi, neg_efe)`, `rollout()` extras use
  `info["neg_efe"]` instead of `info["G"]`, and sophisticated-inference
  planning thresholds use `neg_efe_stop_threshold`.
- Observation handling is stricter and more explicit. The older
  `onehot_obs` naming is now standardized as `categorical_obs`, and environment
  observations should match the `Agent`'s expected observation format.
- Randomness is now explicit in the JAX workflow. Sampling and simulation paths
  such as `sample_action()`, `rollout()`, and environment `reset`/`step`
  methods use explicit PRNG keys.
- JAX workflows are more functional than mutable. Existing code that assumed
  in-place stateful updates on `Agent` objects may need to be rewritten around
  returned values and updated carries.
- Validation is stricter in several places. For example, `learn_A=True` now
  requires `pA`, and model-shape / normalization errors are surfaced earlier.

### Major New Capabilities

#### JAX-First Core Library And Utilities

- The primary `Agent`, inference, control, learning, and planning APIs are now
  built around JAX arrays and pytrees rather than legacy object-array
  conventions.
- The core JAX surface is designed to be JIT-friendly and autodiff-compatible,
  with tighter validation around `Distribution` shapes, normalization, and
  model structure.
- `Agent` methods and related JAX-first library paths are now designed to be
  differentiable, which enables workflows such as fitting or training learned
  front ends around `pymdp`-defined POMDPs, including amortized neural
  encoders.
- Generative models now support explicit sparse dependency structure through
  `A_dependencies` and `B_dependencies`, making it easier to express structured
  observation models and sparse state-transition graphs without falling back to
  dense all-to-all coupling.
- JAX-native utility helpers now cover common model-construction workflows such
  as `random_A_array`, `random_B_array`, `make_A_full`,
  `construct_controllable_B`, and factorized categorical helpers, reducing the
  need to rely on `legacy.utils`.

#### Inference, Control, And Planning

- State inference was substantially upgraded. The release includes optimized
  `infer_states` paths, exact HMM associative-scan inference, corrected
  message-passing details for MMP/VMP, and better support for interacting
  `B_dependencies`.
- Policy scoring and planning semantics were cleaned up and made more explicit.
  The release standardizes `neg_efe` naming and improves reliability in
  parameter-information-gain and inductive policy-scoring paths.
- Sophisticated-inference planning is now JIT-friendly, and the planning stack
  supports richer deliberation and tree-search workflows.
- Monte Carlo Tree Search (MCTS) is now part of the sophisticated-inference
  planning surface, giving users a dedicated planning workflow for more complex
  sequential search problems.
- Canonical variational free energy is now exposed directly as part of the
  public API through `calc_vfe(...)` and `return_info=True` diagnostic paths in
  inference.

#### Environments And Compiled Rollouts

- `pymdp` now has a much clearer JAX-native environment and rollout surface.
  The `Env` abstraction is closer to gymnax-style workflows and is designed to
  work directly with compiled active inference loops.
- `rollout()` is a new first-class API for compiled active inference loops over
  environments, making it much easier to express full perception-action
  rollouts in JAX-native workflows.
- `infer_and_plan()` is also a new first-class helper for stepping agents
  through active inference loops with reusable inference-and-action logic.
- These rollout APIs support explicit key flow, updated-agent carries, and
  learning-aware behavior for online and offline experimental setups.
- Environments can now emit categorical observations directly, which aligns
  better with the rest of the JAX-first API surface.
- The release also broadens the environment and demo corpus with updated
  GridWorld and T-maze workflows, plus the cue-chaining environment and
  tutorial as a new example of epistemic foraging and sequential information
  seeking.

#### Model Fitting And Probabilistic Inference

- `pymdp` now supports probabilistic model-fitting workflows through
  [`pybefit`](https://github.com/dimarkov/pybefit) and NumPyro-backed
  inference, making it possible to fit active-inference POMDP parameters to
  behavioral data such as observed outcomes and action sequences.
- The shipped workflow covers both sampling-based and variational approaches.
  The `fitting_with_pybefit` notebook demonstrates Hamiltonian Monte Carlo via
  NUTS as well as stochastic variational inference with NumPyro-based guides.
- These model-fitting workflows are designed for batched multi-agent settings,
  which makes it easier to express fitting problems over multiple synthetic or
  observed subjects in one JAX-native pipeline.
- Installation and contributor setup now expose this workflow directly through
  the `modelfit` extra.

### Docs, Tutorials, And Examples

- The documentation stack was rebuilt around MkDocs and Material, replacing the
  older Sphinx-centered setup for the primary `1.0.0` docs experience.
- The docs are now intentionally JAX-first. The landing page, quickstart,
  migration guide, rollout guide, environment guide, generated API reference,
  and legacy archive are organized around the modern backend.
- The notebook gallery is now a first-class part of the docs site, with
  refreshed tutorials spanning environments, planning, inference, learning,
  sparse methods, advanced model construction, neural encoders, and model
  fitting.
- Legacy material is still preserved in the docs under a dedicated archive path
  for users who need the older NumPy-oriented reference material during
  migration.

### Packaging, Testing, And Contributor Workflow

- The packaging surface now centers on `pyproject.toml` and setuptools-backed
  builds, with explicit optional-dependency extras for `gpu`, `docs`, `nb`,
  and `modelfit` workflows.
- Notebook execution is now tiered and enforced through manifests: PR CI runs
  the `test/notebooks/ci_notebooks.txt` tier, while nightly coverage runs the
  `test/notebooks/nightly_notebooks.txt` tier.
- Contributor workflow also improved through notebook sanitation hooks, strict
  docs builds, unit-test coverage reporting, and `pytest-xdist` support.

### Traceability Appendix

This page is intentionally curated rather than chronological. The following
rows identify the strongest evidence clusters behind each major section.

| Section | Representative evidence | Key files |
| --- | --- | --- |
| JAX-first backend and migration | `57d730d`, `16985dd`, `#360`, `#361`, `#325` | `pymdp/agent.py`, `pymdp/legacy/agent.py`, `pymdp/utils.py`, `docs-mkdocs/migration/numpy-to-jax.md` |
| Inference, control, and planning | `#315`, `#341`, `#348`, `#349`, `#365`, `#368`, `#370`, `#372` | `pymdp/inference.py`, `pymdp/control.py`, `pymdp/planning/si.py`, `pymdp/planning/mcts.py`, `pymdp/maths.py` |
| Environments and rollout | `#334`, `#354`, `#369` plus the rollout refactor series | `pymdp/envs/env.py`, `pymdp/envs/rollout.py`, `pymdp/envs/cue_chaining.py`, `docs-mkdocs/guides/rollout-active-inference-loop.md` |
| Model fitting and probabilistic inference | `873116a`, `ac34a1b`, `623662b` | `examples/model_fitting/fitting_with_pybefit.ipynb`, `test/test_pybefit_model_fitting.py`, `pyproject.toml`, `docs-mkdocs/index.md` |
| Docs, tutorials, and examples | `9fa629f`, `#354`, `de6ead7` | `mkdocs.yml`, `docs-mkdocs/index.md`, `docs-mkdocs/tutorials/notebooks/index.md`, `docs-mkdocs/getting-started/quickstart-jax.md` |
| Packaging and contributor workflow | `#364`, `#367`, `#373`, `#375` | `pyproject.toml`, `.github/workflows/test.yaml`, `.github/workflows/nightly-tests.yaml`, `test/notebooks/` |
