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
- `--extra nb`: optional notebook/media extras (`mediapy`, `pygraphviz`) when notebook visualization is needed outside test tooling.
- `--extra modelfit`: installs `pybefit` for model-fitting workflows.
- `--extra gpu`: installs CUDA-enabled JAX packages.
- `--extra docs`: documentation toolchain (`mkdocs`, `mkdocs-material`, `mkdocstrings`, `mkdocs-jupyter`, `mkdocs-redirects`).

`pygraphviz` is only needed for Graphviz-backed notebook visualizations such as the MCTS graph-world demo, but it is included in the current `test` and `nb` dependency sets. On many machines `uv sync --group test` works without any extra steps; if it fails while building `pygraphviz`, use the troubleshooting notes below and then rerun the sync command.

Notebook cells that call `mediapy.show_video(...)` also require a system `ffmpeg`
binary. If those cells fail with `Program 'ffmpeg' is not found`, install
`ffmpeg` using the platform-specific notes below before rerunning the notebook.

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

## Notebook video rendering (`mediapy`)

Notebook examples that render videos through `mediapy.show_video(...)` need a
system `ffmpeg` binary in addition to the Python package.

### macOS (Homebrew)

```bash
brew install ffmpeg
```

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

On other platforms, install `ffmpeg` with your system package manager and make
sure the `ffmpeg` executable is available on `PATH`.

## `pygraphviz` troubleshooting

If installation fails while building `pygraphviz`, the missing dependency is usually one of:

- Graphviz itself, including the C headers (`graphviz/cgraph.h`)
- Python development headers for the exact interpreter version in your virtual environment (`Python.h`)

### macOS (Homebrew)

Install Graphviz with Homebrew before syncing dependencies:

```bash
brew install graphviz
```

If `pygraphviz` still fails to build, install it explicitly with Homebrew's include and library paths:

```bash
uv pip install \
  --config-settings=--global-option=build_ext \
  --config-settings=--global-option="-I$(brew --prefix graphviz)/include" \
  --config-settings=--global-option="-L$(brew --prefix graphviz)/lib" \
  pygraphviz==1.14
```

Then rerun:

```bash
uv sync --group test
```

or, if you only need the notebook/media extras:

```bash
uv sync --extra nb
```

### Ubuntu / Debian

Install the Graphviz runtime, Graphviz development headers, `pkg-config`, and the compiler toolchain before syncing dependencies:

```bash
sudo apt-get update
sudo apt-get install -y graphviz libgraphviz-dev pkg-config build-essential
```

Install Python development headers that match the interpreter in your environment:

```bash
sudo apt-get install -y python3-dev
```

If you are using a version-specific interpreter that is newer than the system default, install the matching package instead, for example:

```bash
sudo apt-get install -y python3.12-dev
```

Then rerun:

```bash
uv sync --group test
```

or install the dependency directly inside the active environment:

```bash
uv pip install pygraphviz==1.14
```

If the build output mentions `graphviz/cgraph.h`, install or verify the Graphviz development packages above. If it mentions `Python.h`, install or verify the matching Python development headers.

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
