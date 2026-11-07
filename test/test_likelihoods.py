"""Optional import coverage for the NumPyro likelihood module."""

import pytest

numpyro = pytest.importorskip(
    "numpyro",
    reason="numpyro likelihood integration is optional; install the modelfit extra",
)

from pymdp import likelihoods


def test_likelihoods_imports_with_optional_numpyro_dependency():
    assert numpyro.__name__ == "numpyro"
    assert callable(likelihoods.evolve_trials)
    assert callable(likelihoods.aif_likelihood)
