# Installation

## Recommended workflow: `uv`

### 1) Create and activate a virtual environment
```bash
# from repo root
uv venv .venv
source .venv/bin/activate
```

If `.venv` already exists, just activate it:

```bash
source .venv/bin/activate
```

### 2) Sync dependencies
```bash
uv sync --group test
```

### Dependency groups and extras

- `--group test`: test and notebook tooling (`pytest`, `pytest-xdist`, `nbval`, `jupyter`, `ipykernel`) plus common visualization deps (`mediapy`, `pygraphviz`) and model-fitting dependency (`pybefit`).
- `--group docs`: documentation toolchain (`mkdocs`, `mkdocs-material`, `mkdocstrings`, `mkdocs-jupyter`, `mkdocs-redirects`).
- `--extra nb`: optional notebook/media extras (`mediapy`, `pygraphviz`) when notebook visualization is needed outside test tooling.
- `--extra modelfit`: installs `pybefit` for model-fitting workflows.
- `--extra gpu`: installs CUDA-enabled JAX packages.
- `--extra docs`: documentation extras (same package set as docs tooling).

Common `uv` sync patterns:

```bash
# standard dev/test work
uv sync --group test

# notebook-heavy workflows
uv sync --group test --extra nb

# model fitting workflows
uv sync --group test --extra modelfit

# docs workflow
uv sync --group test --extra docs

# GPU workflow
uv sync --group test --extra gpu
```

## Verify docs build
```bash
./scripts/docs_build.sh
```

## Alternative: `pip`

If you only want package installation (without the `uv` workflow):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install inferactively-pymdp
```
