"""
Shared fixtures and configuration for notebook tests.
"""
import pytest


def pytest_configure(config):
    """Register custom markers for notebook tests."""
    config.addinivalue_line(
        "markers", "notebook: mark test as a notebook test"
    )
    config.addinivalue_line(
        "markers", "fast: mark test as fast-running (suitable for every PR)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running (run on schedule)"
    )
    config.addinivalue_line(
        "markers", "experimental: mark test as experimental/potentially unstable"
    )
