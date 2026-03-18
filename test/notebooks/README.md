# Notebook Test Manifests

This directory contains the notebook manifests used for `pymdp` release gating.
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
- `examples/sparse/sparse_benchmark.ipynb`

All other non-legacy notebooks in `examples/` belong to the CI tier.

## Running Notebook Tests Locally

`nbval` does not play well with `pytest-xdist` in parallel notebook execution. `scripts/run_notebook_manifest.py` therefore forces `-n0` unless you explicitly override it with your own `pytest` arguments.

```bash
# Execute the PR-CI notebook tier without strict output matching
uv run python scripts/run_notebook_manifest.py test/notebooks/ci_notebooks.txt

# Execute the nightly notebook tier without strict output matching
uv run python scripts/run_notebook_manifest.py test/notebooks/nightly_notebooks.txt

# Strictly validate saved outputs for the PR-CI tier
uv run python scripts/run_notebook_manifest.py test/notebooks/ci_notebooks.txt --strict-output
```

## Pre-commit Hooks

Install the notebook hooks with the lightweight dev tooling group:

```bash
uv sync --group dev
uv run --group dev pre-commit install
uv run --group dev pre-commit run --all-files
```

The hook behavior is tier-aware and reads the manifests above:

1. CI-tier notebooks keep saved outputs, strip noisy top-level `kernelspec` / `language_info` metadata, and canonicalize execution counts for output-bearing code cells.
2. Nightly-tier notebooks still run through `nbstripout --keep-output` to remove notebook noise, and then canonicalize execution counts for output-bearing code cells so `nbval` can execute them.
3. Manifest-tested notebooks fail the validation hook if any code cell has saved outputs with `execution_count: null`, or if an `execute_result` output count disagrees with its parent cell.
4. Non-legacy notebooks under `examples/` must appear in one of the two manifests or the hook fails with an instruction to update them.
5. `examples/legacy/` remains excluded from both pytest notebook gating and these hooks.

## Authoring Notes

1. Keep notebooks focused and reasonably small.
2. Avoid massive printed outputs.
3. If a notebook keeps saved outputs, the hook will canonicalize its execution counts so local kernel history does not create meaningless diffs.
4. CI wiring consumes these manifests directly through `scripts/run_notebook_manifest.py`.
