"""Import coverage for the optional NumPyro likelihood module."""

import pytest


def test_likelihoods_imports_with_optional_numpyro_dependency():
    pytest.importorskip("numpyro", reason="Install the modelfit extra for NumPyro integration")
    from pymdp import likelihoods

    assert callable(likelihoods.evolve_trials)
    assert callable(likelihoods.aif_likelihood)
