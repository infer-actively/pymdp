"""`pymdp.planning.visualize` without the optional IPython dependency."""

import os
import subprocess
import sys
import textwrap
import unittest

import pymdp


class TestPlanningVisualizeWithoutIPython(unittest.TestCase):
    def test_import_and_clear_error_without_ipython(self):
        # Run in a fresh interpreter so blocking IPython can't affect other tests.
        code = textwrap.dedent(
            """
            import sys
            sys.modules["IPython"] = None  # makes `import IPython` raise ImportError

            import pymdp.planning.visualize as visualize

            assert callable(visualize.visualize_plan_tree)
            try:
                visualize.visualize_env({})
            except ImportError as e:
                assert "IPython" in str(e), e
            else:
                raise AssertionError("visualize_env should need IPython")
            """
        )
        # import the same pymdp as this test process
        src_root = os.path.dirname(os.path.dirname(os.path.abspath(pymdp.__file__)))
        env = dict(os.environ, MPLBACKEND="Agg")
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (src_root, env.get("PYTHONPATH")) if p
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
