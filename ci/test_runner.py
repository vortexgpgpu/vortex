"""The Vortex test case.

Every catalog entry becomes one `test_case` item (parametrization, markers and
the sim_build fixture live in ci/conftest.py). The body shells out to the
existing executor (blackbox.sh / make) and asserts a clean exit, so those stay
the execution primitives. See docs/designs/continuous_integration.md.

Every failure (and every warning the build escalates to an error) is a real, red
failure — the one exception is a case carrying a `known_issue:` reason in the
catalog, which conftest turns into a tracked xfail: it still builds and runs, but
its failure is expected and does not fail CI (and an unexpected pass shows as
XPASS). Use it only for triaged, documented breakage.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testcase as tc  # noqa: E402


def test_case(case, sim_build):
    xlen = int(os.environ.get("VX_XLEN", "32"))
    argv, env = case.run_command(xlen)
    rc = tc.execute(argv, env)
    assert rc == 0, "{} failed (exit {}): {}".format(case.id, rc, " ".join(argv))
