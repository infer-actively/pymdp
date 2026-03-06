# Notebook Test Manifests

This directory contains the notebook manifests for the `v1.0.0_alpha` release-closure work.
They are the source of truth for how `examples/` notebooks are split between the lighter PR CI tier and the heavier nightly tier.

## Manifest Files

- `ci_notebooks.txt`: notebooks intended for pull-request CI.
- `nightly_notebooks.txt`: notebooks reserved for nightly coverage because they are heavier or have extra environment requirements.

Both manifests are:

1. repo-relative
2. explicitly enumerated
3. sorted for stable diffs

## Current Scope

The manifests cover all non-legacy notebooks under `examples/`.

`examples/legacy/` remains intentionally excluded from these manifests because the current pytest configuration ignores that directory and those notebooks are not part of the release gating scope.

The nightly tier is currently locked to:

- `examples/advanced/pymdp_with_neural_encoder.ipynb`
- `examples/learning/learning_gridworld.ipynb`
- `examples/model_fitting/fitting_with_pybefit.ipynb`

All other non-legacy notebooks in `examples/` belong to the CI tier.

## Running Notebook Tests Locally

`nbval` does not play well with `pytest-xdist` in parallel notebook execution. Keep notebook runs to a single worker.

```bash
# Execute the PR-CI notebook tier without strict output matching
uv run pytest --nbval-lax $(cat test/notebooks/ci_notebooks.txt)

# Execute the nightly notebook tier without strict output matching
uv run pytest --nbval-lax $(cat test/notebooks/nightly_notebooks.txt)

# Strictly validate saved outputs for the PR-CI tier
uv run pytest --nbval $(cat test/notebooks/ci_notebooks.txt)
```

## Authoring Notes

1. Keep notebooks focused and reasonably small.
2. Avoid massive printed outputs.
3. If a notebook is meant to stay in the CI tier, rerun and save it in a state compatible with `nbval`.
4. Workflow wiring is intentionally separate from these manifests and will be handled downstream.
