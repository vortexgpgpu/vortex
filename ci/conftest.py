"""pytest hooks + fixtures for the Vortex test cases.

Registers markers dynamically from the data, turns each test case
(ci/testcase.py) into a parametrized pytest item, and builds each sim once per
build-key. The test itself is ci/test_runner.py. No pytest config file is
needed — markers and collection are handled here. Run from a build tree:

  VX_XLEN=32 pytest ci -m "amo and simx" --strict-markers --dist=loadgroup -n auto

See docs/designs/continuous_integration.md.
"""

import filecmp
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testcase as tc  # noqa: E402


def ambient_xlen():
    """The XLEN of the build tree pytest runs in (build32/ or build64/)."""
    return int(os.environ.get("VX_XLEN", "32"))


def pytest_addoption(parser):
    # perf_gate record-mode: measure and rewrite the golden cycle
    # baselines instead of asserting against them. Human-run + reviewed; CI
    # never passes this flag. See ci/perf_baseline.py.
    parser.addoption("--update-baselines", action="store_true", default=False,
                     help="perf_gate: record measured cycles into "
                          "ci/baselines/perf/ instead of comparing")
    parser.addoption("--shard", default=None, metavar="K/N",
                     help="run only shard K of N of the selected cases, split "
                          "on their recorded runtimes (ci/runtimes.json)")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Keep this --shard's slice of the selection, then run cases that compile
    the same sim back to back. trylast: both act on what -m selected, which is
    the set the planner sized the cell from."""
    _select_shard(config, items)
    _group_by_sim(items)


def _group_by_sim(items):
    """Order cases so those with one sim build signature run consecutively.

    A sim is rebuilt whenever a case's flags differ from the previous case's, so
    a selection that alternates configs A, B, A compiles A twice. Each group
    keeps the position of its first case, and cases keep their order within it.
    """
    xlen = ambient_xlen()
    first, keys = {}, []
    for pos, item in enumerate(items):
        case = item.callspec.params.get("case") if hasattr(item, "callspec") else None
        sig = case.sim_signature(xlen) if case is not None else None
        key = sig if sig is not None else ("#", pos)   # no sim: stays in place
        first.setdefault(key, pos)
        keys.append((first[key], pos))
    order = sorted(range(len(items)), key=lambda i: keys[i])
    items[:] = [items[i] for i in order]


def _select_shard(config, items):
    spec = config.getoption("--shard")
    if not spec:
        return
    try:
        k, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise pytest.UsageError("--shard expects K/N, got {!r}".format(spec))
    if not 1 <= k <= n:
        raise pytest.UsageError("--shard {}: K must be in 1..N".format(spec))
    ids = [item.callspec.id for item in items if hasattr(item, "callspec")]
    _, plan = tc.shard_plan(ids, ambient_xlen(), n)
    keep, drop = [], []
    for item in items:
        cid = item.callspec.id if hasattr(item, "callspec") else None
        (keep if plan.get(cid, 0) == k - 1 else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def pytest_sessionfinish(session):
    if session.config.getoption("--update-baselines"):
        import perf_baseline
        perf_baseline.flush()


def _build_vars():
    """(source root, XLEN) for the build tree pytest is running in.

    config.mk is what configure recorded, so it is authoritative: an ambient
    XLEN in the environment may differ from the one this tree was generated
    with, and regenerating against the wrong width would report every header as
    stale.
    """
    root, xlen = None, None
    try:
        with open("config.mk") as fh:
            for line in fh:
                m = re.match(r"\s*VORTEX_HOME\s*[?:]?=\s*(\S+)", line)
                if m:
                    root = m.group(1)
                m = re.match(r"\s*XLEN\s*[?:]?=\s*(\S+)", line)
                if m:
                    xlen = m.group(1)
    except OSError:
        pass
    return root, xlen


def _assert_config_current():
    """Fail the session if the build tree's generated config is stale.

    configure regenerates <build>/{hw,sw}/*.{vh,h} from the source tomls behind
    an mtime guard, so a tree that was never re-configured after a toml edit or
    a branch switch keeps compiling against the previous configuration. Nothing
    downstream notices, because the drivers, the RTL and the stored perf
    baselines all agree with each other and disagree with the toml.
    Regeneration is byte-stable, so comparing against a fresh generation is an
    exact test.
    """
    source_root, xlen = _build_vars()
    if not source_root or not xlen:
        return
    gen = os.path.join(source_root, "ci", "gen_config.py")
    tomls = sorted(glob.glob(os.path.join(source_root, "*.toml")))
    if not tomls or not os.path.exists(gen):
        return
    env = dict(os.environ)
    env["XLEN"] = xlen
    stale = []
    tmpdir = tempfile.mkdtemp(prefix="vx-cfgchk-", dir=os.getcwd())
    try:
        for toml in tomls:
            name = os.path.splitext(os.path.basename(toml))[0]
            extra = ["--resolved"] if name == "VX_types" else []
            for subdir, ext, fmt in (("hw", ".vh", "verilog"), ("sw", ".h", "cpp")):
                built = os.path.join(subdir, name + ext)
                if not os.path.exists(built):
                    continue
                ref = os.path.join(tmpdir, subdir, name + ext)
                os.makedirs(os.path.dirname(ref), exist_ok=True)
                rc = subprocess.call(
                    [sys.executable, gen, "--config", toml, "--output", ref,
                     "--format", fmt] + extra, env=env)
                if rc != 0:
                    stale.append("{} (could not regenerate)".format(built))
                elif not filecmp.cmp(built, ref, shallow=False):
                    stale.append(built)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if stale:
        raise pytest.UsageError(
            "build tree is stale against the source configuration: {}. "
            "Re-run configure from this build directory before testing — "
            "results and perf baselines would otherwise describe a "
            "configuration that is no longer in the tree.".format(", ".join(stale)))


def pytest_configure(config):
    _assert_config_current()
    # Register every marker the test cases use, derived from the data — so adding
    # a category/driver needs no edit here. With --strict-markers this also makes
    # a typo'd `-m` expression an error instead of a silent empty selection.
    for marker in sorted({m for c in tc.load_all() for m in c.markers()}):
        config.addinivalue_line("markers", "{}: test-case selector".format(marker))


def pytest_generate_tests(metafunc):
    if "case" not in metafunc.fixturenames:
        return
    xlen = ambient_xlen()
    # An opt-in tier needs hardware or a licensed tool a general run does not
    # have (fpga: a Vivado sweep, hours of a whole machine). A bare `pytest ci`
    # -- or regression.sh --all -- must not launch it, so it is skipped unless
    # the selection asks for it by name. The planner's OPT_IN_TIERS check only
    # covers the GitHub matrix; collection is the other way in.
    selected = metafunc.config.getoption("markexpr") or ""
    params = []
    for case in tc.load_all():
        if not case.applies_to_xlen(xlen):
            continue
        marks = [getattr(pytest.mark, name) for name in case.markers()]
        if case.tier in tc.OPT_IN_TIERS and not (
                case.tier in selected or case.category in selected):
            marks.append(pytest.mark.skip(
                reason="opt-in tier '{}': select it by name to run "
                       "(-m {})".format(case.tier, case.tier)))
        if case.known_issue:
            # Tracked expected-failure: the case still builds and runs (so its
            # logs and an unexpected pass are visible as XPASS), but its failure
            # does not fail CI. strict=False tolerates an XPASS rather than
            # converting it to a hard failure.
            marks.append(pytest.mark.xfail(reason="known issue: " + case.known_issue,
                                           strict=False))
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
    # blackbox cases build their own sim at run time with the FULL flag set
    # (configs + the shape-derived -DVX_CFG_NUM_* defines that blackbox.sh
    # appends). A fixture pre-build here uses configs only, so it would compile
    # a shape-stripped config the test never runs — e.g. NUM_TEX_CORES without
    # the matching core count, which elaborates an unreachable arbiter geometry
    # and trips spurious Verilator width lints — and blackbox would rebuild it
    # anyway. Let the run own the build; the sim Makefile's flag stamp clears a
    # stale obj_dir on any flag change, so cross-config builds stay clean.
    if case.via == "blackbox":
        return None
    key = case.build_key()
    if key not in _BUILT:
        # Clean first: a new CONFIGS must not reuse the previous config's obj_dir
        # (stale Verilator state -> spurious lint errors). The regression.sh flow
        # does the same: `make -C sim/<d> clean && CONFIGS=… make -C sim/<d>`.
        tc.execute(["make", "-C", case.sim_dir, "clean"])
        argv, env = case.build_command(ambient_xlen())
        rc = tc.execute(argv, env)
        if rc != 0:
            pytest.fail("sim build failed (exit {}): {}".format(rc, " ".join(argv)))
        _BUILT.add(key)
    return key
