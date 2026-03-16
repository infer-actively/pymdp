# `pymdp` 1.0.0 Release Checklist

Use this checklist for the final release-candidate pass before tagging `v1.0.0`.

## Release gates

- [ ] `pygraphviz` installation guidance is merged in [`README.md`](README.md) and [`docs-mkdocs/getting-started/installation.md`](docs-mkdocs/getting-started/installation.md).
- [ ] Inductive-inference regression coverage is merged in `test/test_inductive_inference_jax.py`.
- [ ] PR CI passes on `.github/workflows/test.yaml`, including `test/notebooks/ci_notebooks.txt`.
- [ ] Nightly CI passes on `.github/workflows/nightly-tests.yaml`, including `test/notebooks/nightly_notebooks.txt`.
- [ ] Docs build cleanly in strict mode.
- [ ] Source distribution and wheel build successfully.

## Local verification

```bash
# unit tests (non-nightly)
uv run pytest test -n 2 -m "not nightly"

# CI-tier notebooks
uv run python scripts/run_notebook_manifest.py test/notebooks/ci_notebooks.txt

# nightly-only Python tests
uv run pytest test -n 2 -m nightly

# nightly-tier notebooks
uv run python scripts/run_notebook_manifest.py test/notebooks/nightly_notebooks.txt

# docs build
uv run --no-default-groups --extra docs mkdocs build --strict

# packaging
uv run --with build python -m build --sdist --wheel
```

## Tagging pass

- [ ] Confirm release notes in [`docs-mkdocs/development/release-notes.md`](docs-mkdocs/development/release-notes.md) describe the shipped `1.0.0` surface.
- [ ] Confirm public links in [`README.md`](README.md) point to stable or `latest` docs rather than pre-release URLs.
- [ ] Create and push the annotated tag `v1.0.0`.
- [ ] Publish the built artifacts to PyPI and create the GitHub release entry.
