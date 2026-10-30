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

A `check: model_parity` case runs the same app/args/configs on its two `legs`
(simx and rtlsim by default) and compares the runtime's final `PERF: instrs=…,
cycles=…` summary: retired instructions must match exactly (both are
deterministic ISA-level executions, so any delta is functional divergence) and
cycles must agree within the case's `tolerance` (SimX is the timing model of the
RTL, not just a functional oracle; Foldsim is cycle-exact, so its tolerance is 0).
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


def _compare_counters(case, model_leg, ref_leg):
    """Compare an MPM class's counters between the two legs, each given as
    (driver, output, cycles)."""
    pc = case.parity_counters
    prefix = pc["prefix"]
    (m, m_out, m_cycles), (r, r_out, r_cycles) = model_leg, ref_leg
    mc = parse_counters(m_out, prefix)
    rc = parse_counters(r_out, prefix)
    assert mc and rc, "{}: no 'PERF: {}: ...' counter line ({}={}, {}={})".format(
        case.id, prefix, m, mc is not None, r, rc is not None)
    if prefix == "rtu":
        for drv, c, cyc in ((m, mc, m_cycles), (r, rc, r_cycles)):
            errs = rtu_invariants(c, cyc)
            assert not errs, "{} [{}]: RTU counter invariants violated: {}".format(
                case.id, drv, "; ".join(errs))
    fails = []
    for name in pc.get("exact", []):
        print("PARITY: {}: {} {}={} {}={}".format(case.id, name, m, mc.get(name), r, rc.get(name)))
        if name not in mc or mc[name] != rc.get(name):
            fails.append("{} {}={} {}={}".format(name, m, mc.get(name), r, rc.get(name)))
    for name in pc.get("tolerance", []):
        s, t = mc.get(name), rc.get(name)
        gap = abs(t - s) / float(max(t, 1)) if (s is not None and t is not None) else 1.0
        print("PARITY: {}: {} {}={} {}={} gap={:.2%}".format(case.id, name, m, s, r, t, gap))
        if gap > case.tolerance:
            fails.append("{} {}={} {}={} gap={:.2%}".format(name, m, s, r, t, gap))
    assert not fails, "{}: {} counter mismatch: {}".format(case.id, prefix, "; ".join(fails))


def _model_parity(case, xlen):
    model, ref = case.legs
    model_instrs, model_cycles, model_out = _run_one(case, xlen, model)
    ref_instrs, ref_cycles, ref_out = _run_one(case, xlen, ref)
    gap = abs(ref_cycles - model_cycles) / float(ref_cycles)
    print("PARITY: {}: instrs {m}={} {r}={}, cycles {m}={} {r}={}, "
          "gap={:.2%} (tolerance {:.0%})".format(
              case.id, model_instrs, ref_instrs, model_cycles, ref_cycles,
              gap, case.tolerance, m=model, r=ref))
    assert model_instrs == ref_instrs, \
        "{}: retired-instruction mismatch ({m}={}, {r}={}) — functional " \
        "divergence, not a timing gap".format(case.id, model_instrs, ref_instrs,
                                              m=model, r=ref)
    assert gap <= case.tolerance, \
        "{}: cycle gap {:.2%} exceeds tolerance {:.0%} ({m}={}, {r}={})".format(
            case.id, gap, case.tolerance, model_cycles, ref_cycles, m=model, r=ref)
    if case.parity_counters:
        _compare_counters(case, (model, model_out, model_cycles),
                          (ref, ref_out, ref_cycles))


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
