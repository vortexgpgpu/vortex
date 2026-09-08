#!/usr/bin/env python3
"""Plot frontend issue shares and line-equivalent memory request rates."""
import argparse
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ISSUE = re.compile(r"TRACE\s+(\d+):.*?pipeline issue:.*?ex=([A-Za-z0-9_]+)")
L2 = re.compile(r"TRACE\s+(\d+):\s+cluster\d+-l2cache(?:-bank\d+)?\s+core-req:.*?op=(LD|ST),")
DRAM = re.compile(r"TRACE\s+(\d+):\s+dram mem-req\d+:\s+op=(LD|ST),")
FUS = ("ALU", "FPU", "LSU", "SFU", "TCU")

def smooth(values, radius):
    if radius <= 1:
        return values
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - radius // 2), min(len(values), i + radius // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out

def parse(path, binsize, line_bytes):
    fu = defaultdict(lambda: defaultdict(int)); l2 = defaultdict(int); dram = defaultdict(int); cycles = []
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            m = ISSUE.search(line)
            if m:
                cycle, unit = int(m.group(1)), m.group(2).upper(); fu[unit][cycle // binsize] += 1; cycles.append(cycle); continue
            m = L2.search(line)
            if m: l2[int(m.group(1)) // binsize] += line_bytes; continue
            m = DRAM.search(line)
            if m: dram[int(m.group(1)) // binsize] += line_bytes
    if not cycles:
        raise ValueError(f"no pipeline issue records in {path}")
    first, last = min(cycles) // binsize, max(cycles) // binsize
    bins = list(range(first, last + 1))
    return bins, fu, l2, dram

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-cycles", type=int, default=512)
    ap.add_argument("--smooth-bins", type=int, default=11)
    ap.add_argument("--line-bytes", type=int, default=64)
    ap.add_argument("--issue-width", type=int, default=2)
    ap.add_argument("--cores", type=int, default=1, help="total cores included in this trace")
    ap.add_argument("--out", required=True)
    ap.add_argument("trace")
    args = ap.parse_args()
    if min(args.bin_cycles, args.line_bytes, args.issue_width, args.cores) <= 0:
        ap.error("bin size, line bytes, issue width, and core count must be positive")
    if args.smooth_bins < 1 or args.smooth_bins % 2 == 0:
        ap.error("smooth-bins must be a positive odd number")
    bins, fu, l2, dram = parse(args.trace, args.bin_cycles, args.line_bytes)
    x = [b * args.bin_cycles for b in bins]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {"ALU":"#4c78a8", "FPU":"#f58518", "LSU":"#54a24b", "SFU":"#e45756", "TCU":"#72b7b2"}
    for unit in FUS:
        raw = [100.0 * fu[unit][b] / (args.bin_cycles * args.issue_width * args.cores) for b in bins]
        axes[0].plot(x, raw, color=colors[unit], alpha=.12, linewidth=.7)
        axes[0].plot(x, smooth(raw, args.smooth_bins), color=colors[unit], label=unit, linewidth=1.8)
    l2raw = [l2[b] / args.bin_cycles for b in bins]; dramraw = [dram[b] / args.bin_cycles for b in bins]
    axes[1].plot(x, l2raw, color="#1f77b4", alpha=.12, linewidth=.7)
    axes[1].plot(x, smooth(l2raw, args.smooth_bins), color="#1f77b4", label="L2 line-equivalent B/cycle", linewidth=1.8)
    axes[1].plot(x, dramraw, "--", color="#ff7f0e", alpha=.12, linewidth=.7)
    axes[1].plot(x, smooth(dramraw, args.smooth_bins), "--", color="#ff7f0e", label="DRAM line-equivalent B/cycle", linewidth=1.8)
    axes[0].set_ylabel("Frontend issue share (%)"); axes[1].set_ylabel("line-equivalent B/cycle"); axes[1].set_xlabel("simulated cycle")
    axes[0].set_title(f"Frontend issues, not FU busy cycles ({args.cores} cores x {args.issue_width} issue slots/cycle)\nMoving average: {args.smooth_bins} x {args.bin_cycles} cycles; raw trace faint", fontsize=11)
    axes[1].set_title(f"Accepted memory requests x {args.line_bytes} B (not measured data-bus occupancy)", fontsize=11)
    for axis in axes: axis.grid(alpha=.25); axis.legend(ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(args.out, dpi=150)

if __name__ == "__main__":
    main()
