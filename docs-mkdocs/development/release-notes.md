# Release Notes

## 1.0.0

- JAX is now the default backend for the primary docs, examples, and release
  surface.
- `Agent.infer_states(..., return_info=True)` and the lower-level inference API
  expose canonical variational free energy diagnostics.
- `rollout()` extras now use `info["neg_efe"]` instead of `info["G"]`, and SI
  planning threshold keywords now use `neg_efe_stop_threshold`.
- The notebook test split is now enforced in CI: PR coverage runs the
  `test/notebooks/ci_notebooks.txt` tier and nightly coverage runs
  `test/notebooks/nightly_notebooks.txt`.
- Installation docs now include explicit `pygraphviz` troubleshooting for macOS
  and Ubuntu contributors.
