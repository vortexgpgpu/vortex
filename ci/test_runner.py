"""The Vortex test case.

Every catalog entry becomes one `test_case` item (parametrization, markers and
the sim_build fixture live in ci/conftest.py). The body shells out to the
existing executor (blackbox.sh / make) and asserts a clean exit, so those stay
the execution primitives. See docs/designs/continuous_integration.md.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testcase as tc  # noqa: E402

# Environment variable that proves a `needs:` dependency is provisioned.
_NEEDS_ENV = {"sst": "SST_ELEMENTS_HOME", "gem5": "GEM5_HOME", "mpi": "MPIHOME"}


def test_case(case, sim_build):
    for need in case.needs:
        env = _NEEDS_ENV.get(need)
        if env and not os.environ.get(env):
            pytest.skip("needs '{}' ({} unset)".format(need, env))
    xlen = int(os.environ.get("VX_XLEN", "32"))
    argv, env = case.run_command(xlen)
    rc = tc.execute(argv, env)
    assert rc == 0, "{} failed (exit {}): {}".format(case.id, rc, " ".join(argv))
