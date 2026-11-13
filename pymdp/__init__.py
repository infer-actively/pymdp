"""pymdp: active inference for discrete POMDPs in JAX."""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    # Single source of truth: the version in pyproject.toml, as installed.
    __version__ = _version("inferactively-pymdp")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"

del PackageNotFoundError, _version
