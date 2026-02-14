# Installation

## Pip
```bash
pip install inferactively-pymdp
```

## Local development with uv
```bash
# from repo root
uv sync --group test
```

## Docs dependencies
```bash
uv sync --no-default-groups --extra docs
```

## Verify docs build
```bash
./scripts/docs_build.sh
```
