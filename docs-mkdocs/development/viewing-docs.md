# Viewing Docs Locally

## Quick commands

From repo root:

```bash
uv sync --no-default-groups --extra docs
./scripts/sync_docs_notebooks.sh
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
- Notebook docs are rendered from pre-executed `.ipynb` files committed in-repo.
- To add notebook docs, update `docs-mkdocs/tutorials/notebooks.manifest` then run `./scripts/sync_docs_notebooks.sh`.
