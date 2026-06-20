"""pytest hooks + fixtures for the Vortex test cases.

Registers markers dynamically from the data, turns each test case
(ci/testcase.py) into a parametrized pytest item, and builds each sim once per
build-key. The test itself is ci/test_runner.py. No pytest config file is
needed — markers and collection are handled here. Run from a build tree:

  VX_XLEN=32 pytest ci -m "amo and simx" --strict-markers --dist=loadgroup -n auto

See docs/designs/continuous_integration.md.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testcase as tc  # noqa: E402


def ambient_xlen():
    """The XLEN of the build tree pytest runs in (build32/ or build64/)."""
    return int(os.environ.get("VX_XLEN", "32"))


def pytest_configure(config):
    # Register every marker the test cases use, derived from the data — so adding
    # a category/driver needs no edit here. With --strict-markers this also makes
    # a typo'd `-m` expression an error instead of a silent empty selection.
    for marker in sorted({m for c in tc.load_all() for m in c.markers()}):
        config.addinivalue_line("markers", "{}: test-case selector".format(marker))


def pytest_generate_tests(metafunc):
    if "case" not in metafunc.fixturenames:
        return
    xlen = ambient_xlen()
    params = []
    for case in tc.load_all():
        if not case.applies_to_xlen(xlen):
            continue
        marks = [getattr(pytest.mark, name) for name in case.markers()]
        params.append(pytest.param(case, marks=marks, id=case.id))
    metafunc.parametrize("case", params)


# Sim builds are deduped per (driver, configs) build-key. Cases run serially
# within a cell (parallelism is across GitHub matrix cells, each its own build
# tree), so this cache builds each sim once and successive CONFIGS never clobber
# a build that is still in use.
_BUILT = set()


@pytest.fixture
def sim_build(case):
    if not case.needs_sim:        # driverless via:script cases self-build
        return None
    key = case.build_key()
    if key not in _BUILT:
        # Clean first: a new CONFIGS must not reuse the previous config's obj_dir
        # (stale Verilator state -> spurious lint errors). Legacy does the same:
        # `make -C sim/<d> clean && CONFIGS=… make -C sim/<d>`.
        tc.execute(["make", "-C", case.sim_dir, "clean"])
        argv, env = case.build_command(ambient_xlen())
        rc = tc.execute(argv, env)
        if rc != 0:
            pytest.fail("sim build failed (exit {}): {}".format(rc, " ".join(argv)))
        _BUILT.add(key)
    return key
