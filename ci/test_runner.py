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

A `check: cycle_parity` case runs the same app/args/configs on simx and rtlsim
and compares the runtime's final `PERF: instrs=…, cycles=…` summary: retired
instructions must match exactly (both are deterministic ISA-level executions,
so any delta is functional divergence) and cycles must agree within the case's
`tolerance` (SimX is the timing model of the RTL, not just a functional oracle).
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testcase as tc  # noqa: E402

# The device-level summary vx_device_dump_perf always prints (per-core lines
# have a `coreN:` prefix and don't match). cycles is max across cores.
_PERF_RE = re.compile(r"^PERF: instrs=(\d+), cycles=(\d+), IPC=", re.M)


def _run_one(case, xlen, driver):
    """One parity leg: run under `driver`, return (instrs, cycles)."""
    argv, env = case.run_command(xlen, driver=driver)
    rc, out = tc.execute_capture(argv, env)
    assert rc == 0, "{} [{}] failed (exit {}): {}".format(
        case.id, driver, rc, " ".join(argv))
    perf = _PERF_RE.findall(out)
    assert perf, "{} [{}]: no 'PERF: instrs=…, cycles=…' summary in output " \
                 "(app must call vx_device_dump_perf)".format(case.id, driver)
    instrs, cycles = perf[-1]
    return int(instrs), int(cycles)


def _cycle_parity(case, xlen):
    simx_instrs, simx_cycles = _run_one(case, xlen, "simx")
    rtl_instrs, rtl_cycles = _run_one(case, xlen, "rtlsim")
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


def test_case(case, sim_build):
    xlen = int(os.environ.get("VX_XLEN", "32"))
    if case.check == "cycle_parity":
        _cycle_parity(case, xlen)
        return
    argv, env = case.run_command(xlen)
    rc = tc.execute(argv, env)
    assert rc == 0, "{} failed (exit {}): {}".format(case.id, rc, " ".join(argv))
