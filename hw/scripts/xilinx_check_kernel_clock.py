#!/usr/bin/env python3
# Post-build kernel-clock truth gate for the xrt hw flow.
#
# Design-wide WNS on an Alveo build is dominated by shell/HBM domains and says
# nothing about the kernel, and the vpl timing check trusts whatever constraint
# is in the design. This gate re-derives the truth from the routed timing
# summary alone: the kernel clock must be constrained at the requested period,
# must meet setup and hold at that period, and must actually contain the
# kernel's endpoints.
#
# Usage: xilinx_check_kernel_clock.py --freq <MHz> --report <timing_summary_routed.rpt>

import argparse
import re
import sys

KERNEL_CLOCK = "clk_kernel_00_unbuffered_net"
PERIOD_TOL_NS = 0.01
MIN_ENDPOINTS = 1000


def parse_clock_summary(lines):
    """Return the constrained period (ns) of the kernel clock, or None."""
    in_section = False
    for line in lines:
        if re.match(r"^\|?\s*Clock Summary\s*$", line):
            in_section = True
            continue
        if in_section:
            if re.search(r"Intra Clock Table", line):
                break
            m = re.match(
                r"^\s*" + re.escape(KERNEL_CLOCK) + r"\s+\{[^}]*\}\s+([\d.]+)\s+([\d.]+)",
                line,
            )
            if m:
                return float(m.group(1))
    return None


def parse_intra_clock(lines):
    """Return (setup_wns, setup_endpoints, hold_whs) for the kernel clock, or None."""
    in_section = False
    for line in lines:
        if re.match(r"^\|?\s*Intra Clock Table\s*$", line):
            in_section = True
            continue
        if in_section:
            if re.search(r"Inter Clock Table", line):
                break
            m = re.match(r"^\s*" + re.escape(KERNEL_CLOCK) + r"\s+(.*)$", line)
            if m:
                fields = m.group(1).split()
                # WNS TNS FailingEP TotalEP  WHS THS FailingEP TotalEP  WPWS ...
                if len(fields) < 8:
                    return None
                return float(fields[0]), int(fields[3]), float(fields[4])
    return None


def parse_new_clk_freq(path):
    """Return the kernel clock 0 frequency (MHz) recorded for the xclbin, or None."""
    try:
        with open(path, errors="replace") as f:
            for line in f:
                fields = line.strip().split(":")
                if len(fields) >= 4 and fields[0] == "kernel" and fields[1] == "0":
                    return float(fields[3])
    except OSError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, required=True, help="target kernel frequency in MHz")
    ap.add_argument("--report", required=True, help="routed timing summary report")
    ap.add_argument("--new-clk-freq", help="vpl _new_clk_freq file (the frequency the xclbin will program)")
    ap.add_argument("--no-scaling", dest="no_scaling", action="store_true",
                    help="fail on a timing miss instead of accepting an honest downscale")
    args = ap.parse_args()

    target_period = 1000.0 / args.freq
    with open(args.report, errors="replace") as f:
        lines = f.read().splitlines()

    failures = []

    period = parse_clock_summary(lines)
    if period is None:
        failures.append(f"clock '{KERNEL_CLOCK}' not found in the Clock Summary")
    elif abs(period - target_period) > PERIOD_TOL_NS:
        failures.append(
            f"kernel clock constrained at {period} ns ({1000.0/period:.1f} MHz), "
            f"target is {target_period:.3f} ns ({args.freq:g} MHz) — "
            f"the implementation was NOT timed at the requested frequency"
        )

    missed = False
    intra = parse_intra_clock(lines)
    if intra is None:
        failures.append(f"clock '{KERNEL_CLOCK}' not found in the Intra Clock Table")
    else:
        wns, endpoints, whs = intra
        if whs < 0:
            failures.append(f"kernel clock hold WHS {whs} < 0: unusable at any frequency")
        if endpoints < MIN_ENDPOINTS:
            failures.append(
                f"kernel clock has only {endpoints} setup endpoints (< {MIN_ENDPOINTS}): "
                f"the kernel does not appear to be in this clock domain"
            )
        if wns < 0 and period is not None:
            missed = True
            achievable = 1000.0 / (period - wns)
            if args.no_scaling:
                failures.append(
                    f"kernel clock setup WNS {wns} at {args.freq:g} MHz "
                    f"(achievable ~{achievable:.1f} MHz) and FREQ_SCALING=0"
                )
            else:
                programmed = parse_new_clk_freq(args.new_clk_freq) if args.new_clk_freq else None
                if programmed is None:
                    failures.append(
                        f"timing missed (WNS {wns}) but the scaled clock frequency "
                        f"could not be read from {args.new_clk_freq}: cannot prove the "
                        f"xclbin reports its real clock"
                    )
                elif programmed > achievable + 0.5:
                    failures.append(
                        f"timing missed (WNS {wns}, achievable ~{achievable:.1f} MHz) but the "
                        f"xclbin will program {programmed:g} MHz — dishonest metadata"
                    )

    if failures:
        print(f"ERROR: kernel clock gate FAILED for {args.report}:")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)

    wns, endpoints, whs = intra
    if missed:
        print(
            f"kernel clock gate PASSED (downscaled): requested {args.freq:g} MHz not met "
            f"(WNS {wns}); the xclbin honestly reports the achieved clock "
            f"(~{1000.0/(period - wns):.1f} MHz). Set FREQ_SCALING=0 to fail such builds."
        )
    else:
        print(
            f"kernel clock gate PASSED: {KERNEL_CLOCK} @ {period} ns ({args.freq:g} MHz), "
            f"setup WNS {wns}, hold WHS {whs}, {endpoints} endpoints"
        )


if __name__ == "__main__":
    main()
