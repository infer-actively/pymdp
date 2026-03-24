# Viewing Docs Locally

## Quick commands

From repo root:

```bash
uv sync --no-default-groups --extra docs
./scripts/docs_build.sh
./scripts/docs_serve.sh
```

Local URL:
- <http://127.0.0.1:8000>

## One-shot sync + serve

```bash
./scripts/docs_sync_and_serve.sh
```

## RTD preview for PRs

1. Open your PR in GitHub.
2. Open the Read the Docs build linked to that PR.
3. Verify nav, notebooks, API pages, and redirects.

## Notes
- `examples/` notebooks are the source of truth for the notebook gallery.
- `docs-mkdocs/tutorials/notebooks/examples/` is generated at docs build time by `./scripts/sync_docs_notebooks.sh` and is not intended to be edited by hand.
- `./scripts/docs_build.sh` and `./scripts/docs_serve.sh` sync curated notebooks automatically before invoking MkDocs.
- To add notebook docs, update `docs-mkdocs/tutorials/notebooks.manifest`. The sync step will copy the listed notebooks into the generated docs tree.
- MkDocs source-of-truth content lives in `docs-mkdocs/`.
- `docs/` is retained as legacy Sphinx-era content for compatibility/history.
