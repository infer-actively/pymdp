# Developer Guide: Writing Testable Notebooks

This guide explains how to write Jupyter notebooks that work well with the automated testing infrastructure.

## Overview

All notebooks in the `examples/` directory are automatically tested using `nbval`, a pytest plugin that executes notebooks and validates their outputs.

## Running Notebook Tests Locally

### Basic Commands

```bash
# Test all notebooks (execution only, no output validation)
uv run pytest --nbval-lax examples/

# Test specific directory
uv run pytest --nbval-lax examples/api/

# Test only "fast" notebooks
uv run pytest test/notebooks/test_notebooks.py -m "fast" --nbval-lax -v

# Test with full output validation
uv run pytest --nbval examples/api/

# Test specific notebook
uv run pytest --nbval examples/api/model_construction_tutorial.ipynb -v
```

:warning: For now, notebook tests are opt-in, i.e. running `uv run pytest test` will only check that notebooks exist.
This tool is currently intended to help test notebooks locally, and eventually we will move them into CI if we can keep execution times low.

The legacy notebooks are not tested.
