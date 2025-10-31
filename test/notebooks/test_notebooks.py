"""
Automated tests for Jupyter notebooks in the examples/ directory.

This module organizes notebooks by category and execution speed,
allowing for flexible test execution strategies.
"""
import pytest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"

# API and tutorial notebooks (fast-running, suitable for every PR)
FAST_NOTEBOOKS = [
    EXAMPLES_DIR / "api" / "model_construction_tutorial.ipynb",
]

# Environment demonstration notebooks
ENV_NOTEBOOKS = [
    EXAMPLES_DIR / "envs" / "tmaze_demo.ipynb",
    EXAMPLES_DIR / "envs" / "generalized_tmaze_demo.ipynb",
    EXAMPLES_DIR / "envs" / "graph_worlds_demo.ipynb",
    EXAMPLES_DIR / "envs" / "knapsack_demo.ipynb",
]

# Learning notebooks
LEARNING_NOTEBOOKS = [
    EXAMPLES_DIR / "learning" / "learning_gridworld.ipynb",
]

# Inference notebooks
INFERENCE_NOTEBOOKS = [
    EXAMPLES_DIR / "inference_and_learning" / "inference_methods_comparison.ipynb",
]

# Inductive inference notebooks
INDUCTIVE_NOTEBOOKS = [
    EXAMPLES_DIR / "inductive_inference" / "inductive_inference_example.ipynb",
    EXAMPLES_DIR / "inductive_inference" / "inductive_inference_gridworld.ipynb",
]

# Advanced feature notebooks
ADVANCED_NOTEBOOKS = [
    EXAMPLES_DIR / "advanced" / "complex_action_dependency.ipynb",
]

# Model fitting notebooks
MODEL_FITTING_NOTEBOOKS = [
    EXAMPLES_DIR / "model_fitting" / "model_inversion.ipynb",
]

# Sparse computation notebooks
SPARSE_NOTEBOOKS = [
    EXAMPLES_DIR / "sparse" / "sparse_benchmark.ipynb",
]

# Experimental notebooks (may be unstable)
EXPERIMENTAL_NOTEBOOKS = [
    EXAMPLES_DIR / "experimental" / "sophisticated_inference" / "si_graph_world.ipynb",
    EXAMPLES_DIR / "experimental" / "sophisticated_inference" / "si_generalized_tmaze.ipynb",
    EXAMPLES_DIR / "experimental" / "sophisticated_inference" / "mcts_graph_world.ipynb",
    EXAMPLES_DIR / "experimental" / "sophisticated_inference" / "mcts_generalized_tmaze.ipynb",
]


@pytest.mark.notebook
@pytest.mark.fast
@pytest.mark.parametrize("notebook", FAST_NOTEBOOKS)
def test_fast_notebooks(notebook):
    """Test fast-running notebooks suitable for every commit."""
    assert notebook.exists(), f"Notebook not found: {notebook}"


@pytest.mark.notebook
@pytest.mark.slow
@pytest.mark.parametrize("notebook", ENV_NOTEBOOKS)
def test_env_notebooks(notebook):
    """Test environment demonstration notebooks."""
    assert notebook.exists(), f"Notebook not found: {notebook}"


@pytest.mark.notebook
@pytest.mark.slow
@pytest.mark.parametrize("notebook", LEARNING_NOTEBOOKS)
def test_learning_notebooks(notebook):
    """Test learning-related notebooks."""
    assert notebook.exists(), f"Notebook not found: {notebook}"


@pytest.mark.notebook
@pytest.mark.slow
@pytest.mark.parametrize("notebook", INFERENCE_NOTEBOOKS)
def test_inference_notebooks(notebook):
    """Test inference method notebooks."""
    assert notebook.exists(), f"Notebook not found: {notebook}"


@pytest.mark.notebook
@pytest.mark.slow
@pytest.mark.parametrize("notebook", INDUCTIVE_NOTEBOOKS)
def test_inductive_notebooks(notebook):
    """Test inductive inference notebooks."""
    assert notebook.exists(), f"Notebook not found: {notebook}"


@pytest.mark.notebook
@pytest.mark.slow
@pytest.mark.parametrize("notebook", ADVANCED_NOTEBOOKS)
def test_advanced_notebooks(notebook):
    """Test advanced feature notebooks."""
    assert notebook.exists(), f"Notebook not found: {notebook}"


@pytest.mark.notebook
@pytest.mark.slow
@pytest.mark.skip("model fitting notebook is currently being migrated, see https://github.com/infer-actively/pymdp/issues/287")
@pytest.mark.parametrize("notebook", MODEL_FITTING_NOTEBOOKS)
def test_model_fitting_notebooks(notebook):
    """Test model fitting and inversion notebooks."""
    assert notebook.exists(), f"Notebook not found: {notebook}"


@pytest.mark.notebook
@pytest.mark.slow
@pytest.mark.parametrize("notebook", SPARSE_NOTEBOOKS)
def test_sparse_notebooks(notebook):
    """Test sparse computation benchmarking notebooks."""
    assert notebook.exists(), f"Notebook not found: {notebook}"


@pytest.mark.notebook
@pytest.mark.experimental
@pytest.mark.parametrize("notebook", EXPERIMENTAL_NOTEBOOKS)
def test_experimental_notebooks(notebook):
    """Test experimental feature notebooks (may be unstable)."""
    assert notebook.exists(), f"Notebook not found: {notebook}"
