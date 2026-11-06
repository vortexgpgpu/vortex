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

A `check: model_parity` case runs the same app/args/configs on simx and rtlsim
and compares the runtime's final `PERF: instrs=…, cycles=…` summary: retired
instructions must match exactly (both are deterministic ISA-level executions,
so any delta is functional divergence) and cycles must agree within the case's
`tolerance` (SimX is the timing model of the RTL, not just a functional oracle).
A case with `parity_counters` also compares that MPM class's counters.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testcase as tc  # noqa: E402
import perf_baseline as pb  # noqa: E402

# The device-level summary vx_device_dump_perf always prints (per-core lines
# have a `coreN:` prefix and don't match). cycles is max across cores.
_PERF_RE = re.compile(r"^PERF: instrs=(\d+), cycles=(\d+), IPC=", re.M)


def _run_one(case, xlen, driver):
    """One parity leg: run under `driver`, return (instrs, cycles, output)."""
    argv, env = case.run_command(xlen, driver=driver)
    rc, out = tc.execute_capture(argv, env)
    assert rc == 0, "{} [{}] failed (exit {}): {}".format(
        case.id, driver, rc, " ".join(argv))
    perf = _PERF_RE.findall(out)
    assert perf, "{} [{}]: no 'PERF: instrs=…, cycles=…' summary in output " \
                 "(app must call vx_device_dump_perf)".format(case.id, driver)
    instrs, cycles = perf[-1]
    return int(instrs), int(cycles), out


def parse_counters(out, prefix):
    """The raw `name=value` counters of an MPM class's device-total line
    (`PERF: <prefix>: ...`; per-unit lines carry a `coreN:` prefix and do not
    match). Derived metrics in parentheses are dropped."""
    lines = re.findall(r"^PERF: {}: (.*)$".format(re.escape(prefix)), out, re.M)
    if not lines:
        return None
    raw = re.sub(r"\([^)]*\)", "", lines[-1].split(";")[0])
    return {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", raw)}


def rtu_invariants(c, cycles):
    """Violations of the RTU counters' definitions (empty when consistent)."""
    errs = []
    if c["hits"] > c["rays"]:
        errs.append("hits {} > rays {}".format(c["hits"], c["rays"]))
    if c["busy"] > cycles:
        errs.append("busy {} > cycles {}".format(c["busy"], cycles))
    return errs


def _compare_counters(case, simx_out, rtl_out, simx_cycles, rtl_cycles):
    pc = case.parity_counters
    prefix = pc["prefix"]
    sc = parse_counters(simx_out, prefix)
    rc = parse_counters(rtl_out, prefix)
    assert sc and rc, "{}: no 'PERF: {}: ...' counter line (simx={}, rtlsim={})".format(
        case.id, prefix, sc is not None, rc is not None)
    if prefix == "rtu":
        for drv, c, cyc in (("simx", sc, simx_cycles), ("rtlsim", rc, rtl_cycles)):
            errs = rtu_invariants(c, cyc)
            assert not errs, "{} [{}]: RTU counter invariants violated: {}".format(
                case.id, drv, "; ".join(errs))
    fails = []
    for name in pc.get("exact", []):
        print("PARITY: {}: {} simx={} rtlsim={}".format(case.id, name, sc.get(name), rc.get(name)))
        if name not in sc or sc[name] != rc.get(name):
            fails.append("{} simx={} rtlsim={}".format(name, sc.get(name), rc.get(name)))
    for name in pc.get("tolerance", []):
        s, r = sc.get(name), rc.get(name)
        gap = abs(r - s) / float(max(r, 1)) if (s is not None and r is not None) else 1.0
        print("PARITY: {}: {} simx={} rtlsim={} gap={:.2%}".format(case.id, name, s, r, gap))
        if gap > case.tolerance:
            fails.append("{} simx={} rtlsim={} gap={:.2%}".format(name, s, r, gap))
    assert not fails, "{}: {} counter mismatch: {}".format(case.id, prefix, "; ".join(fails))


def _model_parity(case, xlen):
    simx_instrs, simx_cycles, simx_out = _run_one(case, xlen, "simx")
    rtl_instrs, rtl_cycles, rtl_out = _run_one(case, xlen, "rtlsim")
    gap = abs(rtl_cycles - simx_cycles) / float(rtl_cycles)
    print("PARITY: {}: instrs simx={} rtlsim={}, cycles simx={} rtlsim={}, "
          "gap={:.2%} (tolerance {:.0%})".format(
              case.id, simx_instrs, rtl_instrs, simx_cycles, rtl_cycles,
              gap, case.tolerance))
    assert simx_instrs == rtl_instrs, \
        "{}: retired-instruction mismatch (simx={}, rtlsim={}) — functional " \
        "divergence, not a timing gap".format(case.id, simx_instrs, rtl_instrs)
    assert gap <= case.tolerance, \
        "{}: cycle gap {:.2%} exceeds tolerance {:.0%} (simx={}, rtlsim={})".format(
            case.id, gap, case.tolerance, simx_cycles, rtl_cycles)
    if case.parity_counters:
        _compare_counters(case, simx_out, rtl_out, simx_cycles, rtl_cycles)


def _perf_gate(case, xlen, update):
    """Run the benchmark on rtlsim; record or compare its cycles vs baseline."""
    instrs, cycles, _ = _run_one(case, xlen, "rtlsim")
    if update:
        pb.record(case, xlen, cycles, instrs)
        print("PERF-BASELINE: {} xlen={} record cycles={} instrs={}".format(
            case.id, xlen, cycles, instrs))
        return
    cat = pb.load(case.category)
    base = cat.get(case.id)
    assert base, "{}: no perf baseline — run pytest --update-baselines".format(case.id)
    assert base.get("config_hash") == pb.config_hash(case), \
        "{}: baseline stale (run config changed) — regenerate with " \
        "--update-baselines".format(case.id)
    ref = base.get(str(xlen))
    assert ref, "{}: no perf baseline at xlen {} — regenerate".format(case.id, xlen)
    assert instrs == ref["instrs"], \
        "{}: workload changed (instrs {} != baseline {}) — regenerate " \
        "baseline".format(case.id, instrs, ref["instrs"])
    ratio = cycles / float(ref["cycles"])
    print("PERF: {}: cycles={} baseline={} ratio={:.3f} (tolerance +/-{:.0%})".format(
        case.id, cycles, ref["cycles"], ratio, pb.TOLERANCE))
    assert ratio <= 1 + pb.TOLERANCE, \
        "{}: perf REGRESSION — cycles {} exceed baseline {} by {:.1%} " \
        "(> {:.0%})".format(case.id, cycles, ref["cycles"], ratio - 1, pb.TOLERANCE)
    assert ratio >= 1 - pb.TOLERANCE, \
        "{}: perf IMPROVEMENT of {:.1%} beyond ratchet — rerun " \
        "--update-baselines to lock the gain in (baseline={}, now={})".format(
            case.id, 1 - ratio, ref["cycles"], cycles)


def test_case(case, sim_build, request):
    xlen = int(os.environ.get("VX_XLEN", "32"))
    if case.check == "model_parity":
        _model_parity(case, xlen)
        return
    if case.check == "perf_gate":
        _perf_gate(case, xlen, request.config.getoption("--update-baselines"))
        return
    argv, env = case.run_command(xlen)
    rc = tc.execute(argv, env)
    assert rc == 0, "{} failed (exit {}): {}".format(case.id, rc, " ".join(argv))
