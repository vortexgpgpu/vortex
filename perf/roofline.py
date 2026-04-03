#!/usr/bin/env python3
# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vortex SGEMM Roofline Plotter
# Runs the regression/sgemmx kernel via ci/blackbox.sh and plots the roofline.
#
# Usage (from build dir):
#   python3 tests/regression/sgemmx/roofline.py [--driver=rtlsim] [--cores=1]
#           [--warps=4] [--threads=4] [--n=128]
#           [--freq=<auto>] [--bw=51.2] [--perf=1]
#           [--output=roofline.png]
#
# Usage (from source tree, targeting a specific build dir):
#   python3 tests/regression/sgemmx/roofline.py --build-dir=build_test32 ...

import argparse
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VORTEX_HOME = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))


def read_platform_clock(build_dir):
    """Read PLATFORM_CLOCK_RATE from generated VX_config.h in the build dir."""
    for candidate in (build_dir, os.path.join(build_dir, "hw")):
        cfg = os.path.join(candidate, "VX_config.h") if candidate else None
        if cfg and os.path.isfile(cfg):
            with open(cfg) as f:
                for line in f:
                    m = re.match(r"#define\s+PLATFORM_CLOCK_RATE\s+(\d+)", line)
                    if m:
                        return int(m.group(1))
    return None


# ────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Vortex SGEMM roofline analyser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--driver",    default="rtlsim",
                   choices=["rtlsim", "simx", "opae", "xrt"],
                   help="Vortex driver")
    p.add_argument("--cores",     type=int, default=1,
                   help="Number of cores (NUM_CORES)")
    p.add_argument("--warps",     type=int, default=4,
                   help="Warps per core (NUM_WARPS)")
    p.add_argument("--threads",   type=int, default=4,
                   help="Threads per warp (NUM_THREADS)")
    p.add_argument("--issue-width", type=int, default=None,
                   help="Issue width (ISSUE_WIDTH)")
    p.add_argument("--n",         type=int, default=32,
                   help="Square matrix dimension N (SGEMM computes N×N × N×N)")
    p.add_argument("--freq",      type=float, default=0,
                   help="Pipeline clock frequency in MHz "
                        "(0 = per-cycle mode, implies --by-cycle; "
                        "omit or set >0 to use time domain)")
    p.add_argument("--bw",        type=float, default=None,
                   help="Peak memory bandwidth in GB/s "
                        "(default: PLATFORM_MEMORY_DATA_SIZE × "
                        "PLATFORM_MEMORY_NUM_BANKS × freq)")
    p.add_argument("--mem-data-size", type=int, default=64,
                   help="PLATFORM_MEMORY_DATA_SIZE in bytes (one port width)")
    p.add_argument("--mem-banks",     type=int, default=2,
                   help="PLATFORM_MEMORY_NUM_BANKS")
    p.add_argument("--perf",      type=int, default=1, choices=[0, 1, 2],
                   help="VORTEX_PROFILING class (0=off, 1=pipeline, 2=memsys)")
    p.add_argument("--by-cycle",  action="store_true",
                   help="Plot in cycle domain (FLOP/cycle vs FLOP/B) "
                        "instead of time domain (GFLOP/s vs FLOP/B). "
                        "Frequency-independent; reveals micro-architectural limits directly.")
    p.add_argument("--output",    default="roofline.png",
                   help="Output image file (extension selects format)")
    p.add_argument("--build-dir", default=None,
                   help="Build directory containing ci/blackbox.sh "
                        "(auto-detected from VORTEX_HOME if omitted)")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Run sgemmx via blackbox.sh
# ────────────────────────────────────────────────────────────────────────────

def find_blackbox(args):
    """Return (blackbox_path, cwd) for the correct build directory."""
    if args.build_dir:
        bdir = os.path.abspath(os.path.expanduser(args.build_dir))
    else:
        # Try the build_test32 / build_test64 sibling of VORTEX_HOME, or VORTEX_HOME itself
        for candidate in (
            os.path.join(VORTEX_HOME, "build"),
            os.path.join(VORTEX_HOME, "build32"),
            os.path.join(VORTEX_HOME, "build64"),
            VORTEX_HOME,
        ):
            bb = os.path.join(candidate, "ci", "blackbox.sh")
            if os.path.isfile(bb):
                return bb, candidate
        # Fallback: source-tree blackbox (may lack config.mk)
        return os.path.join(VORTEX_HOME, "ci", "blackbox.sh"), VORTEX_HOME

    bb = os.path.join(bdir, "ci", "blackbox.sh")
    if not os.path.isfile(bb):
        print(f"ERROR: {bb} not found")
        sys.exit(1)
    return bb, bdir


def run_sgemmx_capture(args):
    """Run sgemmx via blackbox.sh and return captured stdout+stderr."""
    blackbox, cwd = find_blackbox(args)

    cmd = [
        blackbox,
        f"--driver={args.driver}",
        "--app=sgemmx",
        f"--args=-n{args.n}",
    ]
    if args.perf:
        cmd.append(f"--perf={args.perf}")

    # All GPU microarchitecture parameters are injected via CONFIGS so they
    # flow through blackbox.sh's "CONFIGS=$CONFIGS" environment handoff and
    # reach both the RTL build and the simulator.
    configs = []
    configs.append(f"-DNUM_CORES={args.cores}")
    configs.append(f"-DNUM_WARPS={args.warps}")
    configs.append(f"-DNUM_THREADS={args.threads}")
    if args.issue_width is not None:
        configs.append(f"-DISSUE_WIDTH={args.issue_width}")
    if args.mem_banks is not None:
        configs.append(f"-DPLATFORM_MEMORY_NUM_BANKS={args.mem_banks}")
    if args.mem_data_size is not None:
        configs.append(f"-DPLATFORM_MEMORY_DATA_SIZE={args.mem_data_size}")

    env = os.environ.copy()
    existing = env.get("CONFIGS", "")
    env["CONFIGS"] = (existing + " " + " ".join(configs)).strip()

    print(f"CWD    : {cwd}")
    print(f"CMD    : {' '.join(cmd)}")
    print(f"CONFIGS: {env['CONFIGS']}")
    print()

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        env=env,
    )
    output = result.stdout
    sys.stdout.write(output)
    if result.returncode != 0:
        print(f"ERROR: sgemmx exited with status {result.returncode}")
        sys.exit(result.returncode)
    return output


# ────────────────────────────────────────────────────────────────────────────
# Parse PERF output
# ────────────────────────────────────────────────────────────────────────────

def parse_perf(output):
    """
    Parse PERF lines from kernel output.

    Summary line (always present when VORTEX_PROFILING > 0):
      PERF: instrs=X, cycles=Y, IPC=Z

    Per-core lines (class 1, pipeline profiling):
      PERF: core0: instrs=X, cycles=Y, IPC=Z
      PERF: core0: memory: ifetches=..., loads=L, ..., stores=S

    Returns a dict with aggregated metrics.
    """
    perf = {}

    # --- Summary PERF line (instrs/cycles/IPC without "core" prefix) ---
    summary = re.findall(
        r"^PERF:\s+instrs=(\d+),\s*cycles=(\d+),\s*IPC=([0-9.]+)",
        output, re.MULTILINE
    )
    if summary:
        # Last matching summary line is the aggregated one printed by the driver
        instrs, cycles, ipc = summary[-1]
        perf["instrs"] = int(instrs)
        perf["cycles"] = int(cycles)
        perf["ipc"]    = float(ipc)
    else:
        # Fallback: sum per-core lines
        core_lines = re.findall(
            r"PERF:\s+core\d+:\s+instrs=(\d+),\s*cycles=(\d+),\s*IPC=([0-9.]+)",
            output
        )
        if not core_lines:
            print("ERROR: No 'PERF: instrs=...' line found. "
                  "Enable profiling with --perf=1.")
            sys.exit(1)
        perf["instrs"] = sum(int(m[0]) for m in core_lines)
        perf["cycles"] = max(int(m[1]) for m in core_lines)
        perf["ipc"]    = sum(float(m[2]) for m in core_lines)

    perf["cores_seen"] = max(1, len(re.findall(r"PERF: core\d+: instrs=", output)))

    # --- Memory transactions: actual bytes transferred ---
    # read_bytes/write_bytes are printed on the summary memory line.
    mem = re.search(
        r"PERF:\s+memory:.*?read_bytes=(\d+).*?write_bytes=(\d+)",
        output
    )
    if mem:
        perf["actual_bytes"] = int(mem.group(1)) + int(mem.group(2))
        perf["actual_ai"]    = None  # computed later from flops / actual_bytes
    else:
        perf["actual_bytes"] = None
        perf["actual_ai"]    = None

    return perf


# ────────────────────────────────────────────────────────────────────────────
# Roofline maths
# ────────────────────────────────────────────────────────────────────────────

def compute_metrics(args, perf):
    n        = args.n
    freq_hz  = args.freq * 1e6                          # MHz → Hz
    num_hw_threads    = args.cores * args.warps * args.threads
    # Compute throughput threads: one warp issues per cycle per core, executing
    # N_threads SIMT lanes.  N_warps provides occupancy for latency hiding, not
    # additional per-cycle throughput → do NOT multiply by N_warps here.
    num_compute_threads = args.cores * args.threads

    # --- Workload FLOPs ---
    # SGEMM: C = A × B  → 2·N³ FLOPs (N³ muls + N³ adds, or N³ FMAs)
    flops = 2.0 * n ** 3

    # --- Arithmetic intensity ---
    # Ideal (Roofline "capacity" model): one-pass, perfect reuse
    #   load A (N² f32) + load B (N² f32) + store C (N² f32)
    bytes_ideal = 3 * n * n * 4                         # bytes, float32
    ai_ideal    = flops / bytes_ideal                   # FLOP/byte  (= N/6)

    # Actual (from profiler cache-line accounting), if available
    if perf.get("actual_bytes") is not None:
        bytes_actual = perf["actual_bytes"]
        ai_actual    = flops / bytes_actual
    else:
        bytes_actual = None
        ai_actual    = None

    # --- Measured performance ---
    time_sec       = perf["cycles"] / freq_hz
    gflops_actual  = flops / time_sec / 1e9             # GFLOP/s

    # --- Peak compute ---
    # One warp issues per cycle per core → N_threads SIMT lanes active.
    # fmadd.s = 2 FLOPs per lane.
    # P_peak = 2 × N_cores × N_threads  [FLOP/cycle × freq → GFLOP/s]
    gflops_peak    = 2.0 * num_compute_threads * freq_hz / 1e9

    # --- Peak memory bandwidth ---
    if args.bw is not None:
        peak_bw_GBs = args.bw
    else:
        # PLATFORM_MEMORY_DATA_SIZE bytes/port × NUM_BANKS × freq
        peak_bw_GBs = args.mem_data_size * args.mem_banks * freq_hz / 1e9

    ridge_ai = gflops_peak / peak_bw_GBs

    # ── Cycle domain ─────────────────────────────────────────────────────────
    # All quantities below are frequency-independent; they expose micro-
    # architectural limits directly (no clock assumption needed for plotting).
    #
    #   P_peak  [FLOP/cycle] = 2 × N_cores × N_threads
    #                          (one warp/cycle per core × N_threads SIMT lanes × 2 FLOPs/FMA)
    #
    #   BW_peak [B/cycle]    = (B_port × N_banks) / R_clock
    #                        = peak_bw_GBs × 10⁹ / f_core
    #                          (memory bytes delivered per core cycle)
    #
    #   P_actual [FLOP/cycle] = 2·N³ / cycles
    #
    #   I_ridge [FLOP/B]     = P_peak_cycle / BW_peak_cycle  (same as time domain)
    peak_flops_per_cycle  = 2.0 * num_compute_threads          # FLOP/cycle
    peak_bw_per_cycle     = peak_bw_GBs * 1e9 / freq_hz        # B/cycle
    flops_per_cycle       = flops / perf["cycles"]              # FLOP/cycle (measured)
    ridge_ai_cycle        = peak_flops_per_cycle / peak_bw_per_cycle  # same FLOP/B

    return {
        "flops":               flops,
        "bytes_ideal":         bytes_ideal,
        "bytes_actual":        bytes_actual,
        "ai_ideal":            ai_ideal,
        "ai_actual":           ai_actual,
        "time_sec":            time_sec,
        "gflops_actual":       gflops_actual,
        "gflops_peak":         gflops_peak,
        "peak_bw_GBs":         peak_bw_GBs,
        "ridge_ai":            ridge_ai,
        "util_compute":        gflops_actual / gflops_peak * 100,
        "util_memory":         (gflops_actual / (peak_bw_GBs * ai_actual) * 100
                                if ai_actual else None),
        "freq_hz":             freq_hz,
        "num_hw_threads":      num_hw_threads,
        "num_compute_threads": num_compute_threads,
        # cycle-domain equivalents
        "peak_flops_per_cycle": peak_flops_per_cycle,
        "peak_bw_per_cycle":    peak_bw_per_cycle,
        "flops_per_cycle":      flops_per_cycle,
        "ridge_ai_cycle":       ridge_ai_cycle,
        "util_compute_cycle":   flops_per_cycle / peak_flops_per_cycle * 100,
    }


# ────────────────────────────────────────────────────────────────────────────
# Plot
# ────────────────────────────────────────────────────────────────────────────

def plot_roofline(args, perf, m, outfile):
    fig, ax = plt.subplots(figsize=(12, 7))

    ai_range = np.logspace(-3, 4, 3000)

    by_cycle = args.by_cycle

    if by_cycle:
        # ── Cycle domain ─────────────────────────────────────────────────────
        # Y-axis: FLOP/cycle   X-axis: FLOP/B   BW slope: B/cycle
        #
        #   P_peak   = 2 × N_cores × N_warps × N_threads      [FLOP/cycle]
        #   BW_peak  = (B_port × N_banks) / R_clock            [B/cycle]
        #            = peak_bw_GBs × 10⁹ / f_core
        #   P(I)     = min(P_peak, BW_peak × I)               [FLOP/cycle]
        peak_perf  = m["peak_flops_per_cycle"]
        peak_bw    = m["peak_bw_per_cycle"]
        act_perf   = m["flops_per_cycle"]
        ridge      = m["ridge_ai_cycle"]
        y_label    = "Performance  (FLOP / cycle)"
        bw_label   = f"{peak_bw:.2f} B/cycle"
        peak_label = f"Peak compute  {peak_perf:.0f} FLOP/cycle  " \
                     f"(2 × {args.cores}C × {args.threads}T)"
        act_label  = f"{act_perf:.3f} FLOP/cycle"
    else:
        # ── Time domain ──────────────────────────────────────────────────────
        # Y-axis: GFLOP/s   X-axis: FLOP/B   BW slope: GB/s
        #
        #   P_peak   = 2 × N_cores × N_threads × f_core  [GFLOP/s]
        #   BW_peak  = B_port × N_banks × f_core          [GB/s]
        #   P(I)     = min(P_peak, BW_peak × I)           [GFLOP/s]
        peak_perf  = m["gflops_peak"]
        peak_bw    = m["peak_bw_GBs"]
        act_perf   = m["gflops_actual"]
        ridge      = m["ridge_ai"]
        y_label    = "Performance  (GFLOP/s)"
        bw_label   = f"{peak_bw:.1f} GB/s"
        peak_label = f"Peak compute  {peak_perf:.2f} GFLOP/s"
        act_label  = f"{act_perf:.4f} GFLOP/s"

    bw_roof   = peak_bw * ai_range
    comp_roof = np.full_like(ai_range, peak_perf)
    roofline  = np.minimum(bw_roof, comp_roof)

    # --- Roofline envelope ---
    ax.loglog(ai_range, roofline, "b-", linewidth=2.5,
              label="Roofline (peak BW + peak compute)")
    ax.axvline(ridge, color="green", linestyle=":", linewidth=1.0, alpha=0.6,
               label=f"Ridge point  {ridge:.2f} FLOP/B")
    ax.axhline(peak_perf, color="blue", linestyle="--", linewidth=1.0,
               alpha=0.5, label=peak_label)

    # Bandwidth slope label
    bw_mid_ai = ridge * 0.05
    bw_mid_y  = peak_bw * bw_mid_ai
    ax.text(bw_mid_ai * 1.4, bw_mid_y * 1.6,
            bw_label, fontsize=8, color="blue", rotation=35, va="bottom")

    # Efficiency iso-lines
    for eff in (0.25, 0.5, 0.75):
        ax.axhline(peak_perf * eff, color="gray",
                   linestyle=":", linewidth=0.5, alpha=0.4)
        ax.text(ai_range[-1] * 0.98, peak_perf * eff * 1.05,
                f"{int(eff * 100)}%", fontsize=7, ha="right",
                color="gray", va="bottom")

    # --- Operating points ---
    # 1. Actual AI (from profiler cache-line accounting) — primary point
    if m["ai_actual"] is not None:
        ax.plot(m["ai_actual"], act_perf, "ro", markersize=12, zorder=6,
                label=f"Actual  AI={m['ai_actual']:.2f} FLOP/B  ({act_label})")
        _annotate(ax, m["ai_actual"], act_perf, "darkred",
                  f"  actual AI\n  {act_label}")

    # 2. Ideal AI (one-pass, perfect reuse) — shows potential
    ax.plot(m["ai_ideal"], act_perf, "r^", markersize=10, zorder=5,
            markerfacecolor="none", markeredgecolor="darkred",
            label=f"Ideal  AI={m['ai_ideal']:.1f} FLOP/B  "
                  f"(2N / (3×sizeof(T)) = N/6, perfect reuse)")
    _annotate(ax, m["ai_ideal"], act_perf * 0.6, "darkred",
              f"  ideal AI\n  (N/6={m['ai_ideal']:.1f})")

    # Dashed arrow: actual AI → ideal AI (data-reuse gap)
    if m["ai_actual"] is not None:
        ax.annotate(
            "", xy=(m["ai_ideal"], act_perf * 0.65),
            xytext=(m["ai_actual"] * 1.05, act_perf * 0.95),
            arrowprops=dict(arrowstyle="->", color="salmon",
                            lw=1.0, linestyle="dashed"),
        )
        ax.text((m["ai_actual"] + m["ai_ideal"]) * 0.7, act_perf * 0.77,
                "cache\nreuse gap", fontsize=7, color="salmon",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor="salmon", alpha=0.7))

    # --- Labels / title ---
    domain_tag = "cycle domain" if by_cycle else f"{args.freq:.0f} MHz"
    ax.set_xlabel("Arithmetic Intensity  (FLOP / byte)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(
        f"Vortex SGEMM Roofline — "
        f"{args.cores}C / {args.warps}W / {args.threads}T, "
        f"N={args.n}, {args.driver}, {domain_tag}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.25)

    # --- Summary text box ---
    util_c = m["util_compute_cycle"] if by_cycle else m["util_compute"]
    lines = [
        f"cycles        = {perf['cycles']:>13,}",
        f"instrs        = {perf['instrs']:>13,}",
        f"IPC           = {perf['ipc']:>13.3f}",
        f"FLOPs         = {m['flops'] / 1e6:>12.2f} M  (2·N³)",
    ]
    if by_cycle:
        lines += [
            f"actual perf   = {m['flops_per_cycle']:>10.3f} FLOP/cycle",
            f"peak compute  = {m['peak_flops_per_cycle']:>10.0f} FLOP/cycle",
            f"peak BW       = {m['peak_bw_per_cycle']:>10.2f} B/cycle",
            f"ridge AI      = {m['ridge_ai_cycle']:>11.2f} FLOP/B",
        ]
    else:
        lines += [
            f"exec time     = {m['time_sec'] * 1e3:>12.2f} ms",
            f"actual perf   = {m['gflops_actual']:>11.4f} GFLOP/s",
            f"peak compute  = {m['gflops_peak']:>11.2f} GFLOP/s",
            f"peak BW       = {m['peak_bw_GBs']:>12.1f} GB/s",
            f"ridge AI      = {m['ridge_ai']:>12.2f} FLOP/B",
        ]
    lines += [
        f"compute util  = {util_c:>11.3f} %",
        f"ideal AI      = {m['ai_ideal']:>12.2f} FLOP/B  (N/6)",
    ]
    if m["util_memory"] is not None:
        lines.append(f"bw util (act) = {m['util_memory']:>11.3f} %")
    if m["ai_actual"] is not None:
        lines.append(f"actual AI     = {m['ai_actual']:>12.2f} FLOP/B")
        lines.append(f"reuse factor  = {m['ai_actual'] / m['ai_ideal']:>12.4f}×")

    ax.text(
        0.99, 0.03, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=7.5, family="monospace",
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    return os.path.abspath(outfile)


def _annotate(ax, ai, gfl, color, label):
    ax.annotate(
        label,
        xy=(ai, gfl),
        xytext=(ai * 3.0, gfl * 1.3),
        fontsize=8, color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
    )


# ────────────────────────────────────────────────────────────────────────────
# Console summary
# ────────────────────────────────────────────────────────────────────────────

def print_metrics(args, perf, m):
    sep = "═" * 56
    print(f"\n{sep}")
    print("  Vortex SGEMM Roofline Metrics")
    print(sep)
    print(f"  Config          : {args.cores}C / {args.warps}W / {args.threads}T")
    print(f"  Matrix size     : {args.n} × {args.n}")
    print(f"  Driver          : {args.driver}")
    if args.by_cycle:
        print(f"  Domain          : per-cycle")
    else:
        print(f"  Clock           : {args.freq:.0f} MHz")
    print()
    print(f"  FLOPs           : {m['flops'] / 1e6:.3f} M  (2·N³)")
    print(f"  Bytes  (ideal)  : {m['bytes_ideal'] / 1024:.1f} KiB"
          f"  (3·N²·4, perfect reuse)")
    if m["bytes_actual"] is not None:
        print(f"  Bytes  (actual) : {m['bytes_actual'] / 1e6:.2f} MB"
              f"  ({m['bytes_actual'] / m['bytes_ideal']:.0f}× ideal)")
    print()
    print(f"  Cycles          : {perf['cycles']:,}")
    print(f"  Instrs          : {perf['instrs']:,}")
    print(f"  IPC             : {perf['ipc']:.3f}")
    print()
    if args.by_cycle:
        print(f"  Actual FLOP/c   : {m['flops_per_cycle']:.4f}")
        print(f"  Peak FLOP/c     : {m['peak_flops_per_cycle']:.0f}"
              f"  (2 × {args.cores}C × {args.threads}T)")
        print(f"  Compute util.   : {m['util_compute_cycle']:.3f} %")
        print()
        print(f"  Peak BW         : {m['peak_bw_per_cycle']:.2f} B/cycle")
        print(f"  Ridge AI        : {m['ridge_ai_cycle']:.2f} FLOP/byte")
    else:
        print(f"  Exec time       : {m['time_sec'] * 1e3:.3f} ms")
        print(f"  Actual GFLOP/s  : {m['gflops_actual']:.4f}")
        print(f"  Peak GFLOP/s    : {m['gflops_peak']:.2f}"
              f"  (2 × {args.cores}C × {args.threads}T × {args.freq:.0f} MHz)")
        print(f"  Compute util.   : {m['util_compute']:.3f} %")
        print()
        print(f"  Peak BW         : {m['peak_bw_GBs']:.1f} GB/s")
        print(f"  Ridge AI        : {m['ridge_ai']:.2f} FLOP/byte")
    print(f"  AI  (ideal)     : {m['ai_ideal']:.2f} FLOP/byte  (N/6)")
    if m["ai_actual"] is not None:
        print(f"  AI  (actual)    : {m['ai_actual']:.3f} FLOP/byte")
        print(f"  Cache reuse     : {m['ai_actual'] / m['ai_ideal']:.4f}×"
              f"  of ideal  ← data-reuse bottleneck")
    if m["util_memory"] is not None:
        print(f"  BW util (act.)  : {m['util_memory']:.3f} %")
    print(sep)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # --freq=0 implies --by-cycle (frequency-independent mode)
    if args.freq == 0:
        args.by_cycle = True
        _, bdir = find_blackbox(args)
        freq = read_platform_clock(bdir)
        args.freq = float(freq) if freq else 1.0

    output = run_sgemmx_capture(args)
    perf   = parse_perf(output)

    print(f"\nParsed ({perf['cores_seen']} core(s)): "
          f"instrs={perf['instrs']}, cycles={perf['cycles']}, "
          f"IPC={perf['ipc']:.3f}")

    m = compute_metrics(args, perf)
    print_metrics(args, perf, m)

    saved = plot_roofline(args, perf, m, args.output)
    print(f"\nRoofline saved → {saved}")


if __name__ == "__main__":
    main()
