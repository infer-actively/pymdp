"""Package metadata stays single-sourced from pyproject.toml."""

import importlib.metadata
import unittest

import pymdp


class TestPackageMetadata(unittest.TestCase):
    def test_version_matches_installed_metadata(self):
        self.assertEqual(
            pymdp.__version__, importlib.metadata.version("inferactively-pymdp")
        )


if __name__ == "__main__":
    unittest.main()
