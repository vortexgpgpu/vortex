#!/usr/bin/env python3
# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Hardware profile check
#
# Compares a hardware profile (profiles/<name>.json) against its `reference`
# block: the part's published or measured numbers. Peaks are computed from the
# profile's resolved configuration; latencies and kernel cycles are measured
# on a driver (SimX by default) through ci/blackbox.sh.
#
# Reference keys (each optional; a profile checks the ones it has):
#   clock_mhz                    core clock the cycle counts are taken at
#   fp32_lanes                   FP32 FMA lanes of the whole chip (clock-free peak)
#   dram_bandwidth_gbs, dram_peak_bytes_per_cycle
#                                peak DRAM bandwidth
#   l1_hit_cycles, l2_hit_cycles, dram_cycles
#                                pointer-chase load-to-use latency (one core)
#   stream_cycles {"<n>MiB": c}  vecadd with n MiB per buffer (whole chip)
#   sgemm_cycles  {"<n>": c}     sgemm -n<n> (whole chip)
#
# Run from a configured build directory whose XLEN matches the profile:
#   python3 ci/profile_check.py --profile=a100
#   python3 ci/profile_check.py --profile=rtx5080 --metrics=latency --tolerance=0.05
#
# Exit status: 0 all checked metrics within tolerance, 1 a metric outside it,
# 2 a run failed.

import argparse
import json
import os
import re
import shlex
import subprocess
import sys

VORTEX_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Latency iterations: the per-load latency is the cycle difference of a run
# with 2N dependent loads and one with N, over N.
CHASE_ITERS = 256
LINE_STRIDE = 128

GROUPS = ("peak", "latency", "stream", "sgemm")


def parse_args():
    p = argparse.ArgumentParser(description="Check a hardware profile against its reference numbers.")
    p.add_argument("--profile", required=True, help="profile name (profiles/<name>.json)")
    p.add_argument("--driver", default="simx", help="driver for measured metrics (default: simx)")
    p.add_argument("--metrics", default=",".join(GROUPS),
                   help="comma list of metric groups: " + ", ".join(GROUPS))
    p.add_argument("--tolerance", type=float, default=0.10,
                   help="relative error allowed per metric (default: 0.10)")
    p.add_argument("--build-dir", default=os.getcwd(), help="configured build directory (default: cwd)")
    p.add_argument("--timeout", type=int, default=7200, help="seconds per run (default: 7200)")
    p.add_argument("--dry-run", action="store_true", help="print the runs without executing them")
    return p.parse_args()


def build_xlen(build_dir):
    with open(os.path.join(build_dir, "config.mk")) as f:
        for line in f:
            m = re.match(r"\s*XLEN\s*\?=\s*(\d+)", line)
            if m:
                return int(m.group(1))
    sys.exit(f"ERROR: no XLEN in {build_dir}/config.mk")


def load_profile(name):
    path = os.path.join(VORTEX_HOME, "profiles", f"{name}.json")
    with open(path) as f:
        data = json.load(f)
    reference = {}
    if "extends" in data:
        reference.update(load_profile(data["extends"])[1])
    reference.update(data.get("reference", {}))
    return data, reference


def resolved_config(name, xlen):
    cmd = [sys.executable, os.path.join(VORTEX_HOME, "ci", "gen_config.py"),
           "--config", os.path.join(VORTEX_HOME, "VX_config.toml"),
           "--profile", name, "--xlen", str(xlen), "-f", "cflags",
           "--cflags", f"-DVX_CFG_XLEN={xlen} -DVX_CFG_XLEN_{xlen}"]
    out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True).stdout
    cfg = {}
    for flag in shlex.split(out):
        m = re.match(r"-D(VX_CFG_\w+)(?:=(.*))?$", flag)
        if m:
            val = m.group(2)
            try:
                cfg[m.group(1)] = int(val, 0) if val is not None else 1
            except ValueError:
                cfg[m.group(1)] = val
    return cfg


class Runner:
    def __init__(self, args):
        self.args = args
        self.blackbox = os.path.join(args.build_dir, "ci", "blackbox.sh")
        if not os.path.isfile(self.blackbox):
            sys.exit(f"ERROR: {self.blackbox} not found; run from a configured build directory")
        self.failed = False
        self.built = set()

    def cycles(self, app, app_args, extra=()):
        cmd = [self.blackbox, f"--driver={self.args.driver}", f"--profile={self.args.profile}",
               f"--app={app}", f"--args={app_args}", *extra]
        print("  run: " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        if self.args.dry_run:
            return None
        # blackbox rebuilds the driver, not the kernel: a kernel left from
        # another configuration would run with its geometry.
        if (app, extra) not in self.built:
            subprocess.run(["make", "-C", os.path.join("tests", "regression", app), "clean"],
                           cwd=self.args.build_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.built.add((app, extra))
        try:
            proc = subprocess.run(cmd, cwd=self.args.build_dir, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, timeout=self.args.timeout)
        except subprocess.TimeoutExpired:
            print(f"  ERROR: {app} timed out after {self.args.timeout}s")
            self.failed = True
            return None
        found = re.findall(r"cycles=(\d+)", proc.stdout)
        if proc.returncode != 0 or "PASSED" not in proc.stdout or not found:
            tail = "\n".join(proc.stdout.splitlines()[-15:])
            print(f"  ERROR: {app} {app_args} failed (exit {proc.returncode}):\n{tail}")
            self.failed = True
            return None
        return int(found[-1])

    def chase_latency(self, footprint, warm, cold=False):
        single_core = ("--clusters=1", "--cores=1")
        base = f"-b{footprint} -w{warm}" + (" -c" if cold else "")
        c1 = self.cycles("chase", f"{base} -i{CHASE_ITERS}", single_core)
        c2 = self.cycles("chase", f"{base} -i{2 * CHASE_ITERS}", single_core)
        if c1 is None or c2 is None:
            return None
        return (c2 - c1) / CHASE_ITERS


def pow2_floor(x):
    return 1 << (int(x).bit_length() - 1)


def peak_metrics(cfg, reference):
    out = {}
    cores = cfg["VX_CFG_NUM_CLUSTERS"] * cfg["VX_CFG_NUM_CORES"]
    clock_mhz = cfg["VX_CFG_PLATFORM_CLOCK_RATE"]
    if "fp32_lanes" in reference:
        out["fp32_lanes"] = cores * cfg["VX_CFG_ISSUE_WIDTH"] * cfg["VX_CFG_NUM_FPU_LANES"]
    if "dram_bandwidth_gbs" in reference:
        out["dram_bandwidth_gbs"] = cfg["VX_CFG_PLATFORM_MEMORY_PEAK_BW"] / 1000
    if "dram_peak_bytes_per_cycle" in reference:
        out["dram_peak_bytes_per_cycle"] = cfg["VX_CFG_PLATFORM_MEMORY_PEAK_BW"] / clock_mhz
    if "clock_mhz" in reference:
        out["clock_mhz"] = clock_mhz
    return out


def latency_metrics(runner, cfg, reference):
    l1 = cfg["VX_CFG_DCACHE_SIZE"]
    l2 = cfg["VX_CFG_L2_SIZE"] if cfg.get("VX_CFG_L2_ENABLED") else 0
    # A random chase over F bytes hits a C-byte cache about C/F of the time:
    # half the L1 always hits, 32 L1s miss it 97% of the time.
    plan = {
        "l1_hit_cycles": (pow2_floor(l1 // 2), 1),
        "l2_hit_cycles": (min(pow2_floor(32 * l1), pow2_floor(l2 // 2)), 1) if l2 else None,
        # a cold chain over a buffer never uploaded (uploads land in the
        # last-level cache): every load misses to DRAM
        "dram_cycles":   (2 * 2 * CHASE_ITERS * LINE_STRIDE, 0, True),
    }
    out = {}
    for key, spec in plan.items():
        if key not in reference:
            continue
        if spec is None:
            print(f"  {key}: the profile has no L2")
            continue
        out[key] = runner.chase_latency(*spec)
    return out


def table_metrics(runner, reference, key, app, to_args):
    out = {}
    for size, _ in reference.get(key, {}).items():
        out[f"{key}[{size}]"] = runner.cycles(app, to_args(size))
    return out


def stream_args(size):
    m = re.fullmatch(r"(\d+)MiB", size)
    if not m:
        sys.exit(f"ERROR: stream_cycles key '{size}' is not '<n>MiB'")
    return f"-n{int(m.group(1)) * (1 << 20) // 4}"


def flatten_reference(reference):
    flat = {}
    for key, val in reference.items():
        if isinstance(val, dict):
            for size, v in val.items():
                flat[f"{key}[{size}]"] = v
        else:
            flat[key] = val
    return flat


def main():
    args = parse_args()
    groups = [g for g in args.metrics.split(",") if g]
    for g in groups:
        if g not in GROUPS:
            sys.exit(f"ERROR: unknown metric group '{g}'")

    xlen = build_xlen(args.build_dir)
    data, reference = load_profile(args.profile)
    if data.get("xlen") not in (None, xlen):
        sys.exit(f"ERROR: profile {args.profile} needs XLEN={data['xlen']}, the build is XLEN={xlen}")
    if not reference:
        sys.exit(f"ERROR: profile {args.profile} has no reference block")
    cfg = resolved_config(args.profile, xlen)
    runner = Runner(args)

    measured = {}
    if "peak" in groups:
        measured.update(peak_metrics(cfg, reference))
    if "latency" in groups:
        measured.update(latency_metrics(runner, cfg, reference))
    if "stream" in groups:
        measured.update(table_metrics(runner, reference, "stream_cycles", "vecadd", stream_args))
    if "sgemm" in groups:
        measured.update(table_metrics(runner, reference, "sgemm_cycles", "sgemm", lambda n: f"-n{n}"))

    if args.dry_run:
        return 0

    flat = flatten_reference(reference)
    outside = 0
    print(f"\nprofile {args.profile} ({args.driver}), tolerance {args.tolerance:.0%}")
    print(f"{'metric':<28}{'reference':>14}{'model':>14}{'error':>9}  verdict")
    for key, model in measured.items():
        ref = flat[key]
        if model is None:
            print(f"{key:<28}{ref:>14g}{'-':>14}{'-':>9}  ERROR")
            continue
        err = (model - ref) / ref
        ok = abs(err) <= args.tolerance
        outside += not ok
        print(f"{key:<28}{ref:>14g}{model:>14.6g}{err:>+9.1%}  {'ok' if ok else 'OUTSIDE'}")
    unchecked = sorted(k for k in flat if k not in measured)
    if unchecked:
        print("not checked: " + ", ".join(unchecked))

    if runner.failed:
        return 2
    return 1 if outside else 0


if __name__ == "__main__":
    sys.exit(main())
