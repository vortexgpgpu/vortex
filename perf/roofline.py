#!/usr/bin/env python3
# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Vortex Roofline + Configuration Search
#
# Runs a kernel via ci/blackbox.sh, sweeps microarchitecture knobs to maximize
# IPC, and (optionally) plots a conventional FLOP/cycle roofline for the
# winning configuration.
#
# Knobs accept a single value or a comma-list:
#   --threads=8          single
#   --threads=4,8,16     search space
#
# Examples — sgemm (N=128, FLOPs = 2·N³ = 4194304):
#
#   # Single run with roofline plot:
#   python3 perf/roofline.py --driver=simx --app=sgemm --args="-n128" \
#     --flops=4194304 --plot --output=sgemm128.png
#
#   # Search best config (coordinate descent) with plot of the winner:
#   python3 perf/roofline.py --driver=simx --app=sgemm --args="-n128" \
#     --flops=4194304 --plot --output=sgemm128_best.png \
#     --config=fast \
#     --threads=4,8,16 --warps=4,8,16 --issue-width=1,2,4 \
#     --fpu-blocks=1,2,4 --dcache-mshr=8,16,32 --l2-enable=0,1
#
#   # Random sampling (50 trials) across a wide space:
#   python3 perf/roofline.py --driver=simx --app=sgemm --args="-n128" \
#     --flops=4194304 --config=random --max-trials=50 \
#     --threads=4,8,16 --warps=4,8,16,32 --fpu-blocks=1,2,4 \
#     --dcache-size=16384,32768,65536 --dcache-ways=2,4,8

import argparse
import datetime
import itertools
import json
import os
import random as rand
import re
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
VORTEX_HOME = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))


# ────────────────────────────────────────────────────────────────────────────
# Knob table: (arg_name, macro_name, cli_flag, pow2?, is_bool?)
# Order below is also the priority order used by the 'fast' search.
# ────────────────────────────────────────────────────────────────────────────
KNOBS = [
    ("threads",          "NUM_THREADS",                "--threads",          True,  False),
    ("warps",            "NUM_WARPS",                  "--warps",            True,  False),
    ("issue_width",      "ISSUE_WIDTH",                "--issue-width",      True,  False),
    ("fpu_blocks",       "NUM_FPU_BLOCKS",             "--fpu-blocks",       False, False),
    ("alu_blocks",       "NUM_ALU_BLOCKS",             "--alu-blocks",       False, False),
    ("lsu_blocks",       "NUM_LSU_BLOCKS",             "--lsu-blocks",       False, False),
    ("cores",            "NUM_CORES",                  "--cores",            True,  False),
    ("ibuf_size",        "IBUF_SIZE",                  "--ibuf-size",        True,  False),
    ("lsu_inq_size",     "LSUQ_IN_SIZE",               "--lsu-inq-size",     False, False),
    ("lsu_outq_size",    "LSUQ_OUT_SIZE",              "--lsu-outq-size",    False, False),
    ("dcache_mshr",      "DCACHE_MSHR_SIZE",           "--dcache-mshr",      False, False),
    ("dcache_size",      "DCACHE_SIZE",                "--dcache-size",      True,  False),
    ("dcache_ways",      "DCACHE_NUM_WAYS",            "--dcache-ways",      True,  False),
    ("dcache_banks",     "DCACHE_NUM_BANKS",           "--dcache-banks",     True,  False),
    ("dcache_repl",      "DCACHE_REPL_POLICY",         "--dcache-repl",      False, False),
    ("dcache_writeback", "DCACHE_WRITEBACK",           "--dcache-writeback", False, False),
    ("icache_size",      "ICACHE_SIZE",                "--icache-size",      True,  False),
    ("icache_ways",      "ICACHE_NUM_WAYS",            "--icache-ways",      True,  False),
    ("icache_banks",     "ICACHE_NUM_BANKS",           "--icache-banks",     True,  False),
    ("icache_mshr",      "ICACHE_MSHR_SIZE",           "--icache-mshr",      False, False),
    ("icache_repl",      "ICACHE_REPL_POLICY",         "--icache-repl",      False, False),
    ("icache_writeback", "ICACHE_WRITEBACK",           "--icache-writeback", False, False),
    ("l2_enable",        "L2_ENABLE",                  "--l2-enable",        False, True),
    ("l2_size",          "L2_CACHE_SIZE",              "--l2-size",          True,  False),
    ("l2_ways",          "L2_NUM_WAYS",                "--l2-ways",          True,  False),
    ("l2_banks",         "L2_NUM_BANKS",               "--l2-banks",         True,  False),
    ("l2_mshr",          "L2_MSHR_SIZE",               "--l2-mshr",          False, False),
    ("l2_repl",          "L2_REPL_POLICY",             "--l2-repl",          False, False),
    ("l2_writeback",     "L2_WRITEBACK",               "--l2-writeback",     False, False),
    ("mem_banks",        "PLATFORM_MEMORY_NUM_BANKS",  "--mem-banks",        True,  False),
    ("mem_data_size",    "PLATFORM_MEMORY_DATA_SIZE",  "--mem-data-size",    True,  False),
]

KNOB_BY_NAME = {k[0]: k for k in KNOBS}


# ────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────

class _HelpFmt(argparse.ArgumentDefaultsHelpFormatter,
               argparse.RawDescriptionHelpFormatter):
    """Defaults formatter that skips '(default: None)' and keeps raw epilog."""
    def _get_help_string(self, action):
        if action.default is None or action.default == "":
            return action.help
        return super()._get_help_string(action)


EXAMPLES = """\
Examples — sgemm (N=128, FLOPs = 2·N³ = 4194304):

  # Single run with roofline plot:
  python3 perf/roofline.py --driver=simx --app=sgemm --args="-n128" \\
    --flops=4194304 --plot --output=sgemm128.png

  # Search best config (coordinate descent) with plot of the winner:
  python3 perf/roofline.py --driver=simx --app=sgemm --args="-n128" \\
    --flops=4194304 --plot --output=sgemm128_best.png \\
    --config=fast \\
    --threads=4,8,16 --warps=4,8,16 --issue-width=1,2,4 \\
    --fpu-blocks=1,2,4 --dcache-mshr=8,16,32 --l2-enable=0,1

  # Random sampling (50 trials) across a wide space:
  python3 perf/roofline.py --driver=simx --app=sgemm --args="-n128" \\
    --flops=4194304 --config=random --max-trials=50 \\
    --threads=4,8,16 --warps=4,8,16,32 --fpu-blocks=1,2,4 \\
    --dcache-size=16384,32768,65536 --dcache-ways=2,4,8
"""


def parse_args():
    p = argparse.ArgumentParser(
        description="Vortex roofline analyzer",
        epilog=EXAMPLES,
        formatter_class=_HelpFmt,
    )
    p.add_argument("--driver", default="rtlsim", metavar="name",
                   choices=["rtlsim", "simx", "opae", "xrt"],
                   help="Vortex driver: rtlsim|simx|opae|xrt")
    p.add_argument("--app", required=True, metavar="name",
                   help="Benchmark application")
    p.add_argument("--args", dest="app_args", default="", metavar="str",
                   help="App arguments (passed to blackbox.sh --args)")
    p.add_argument("--perf", type=int, default=1, choices=[0, 1, 2], metavar="n")
    p.add_argument("--configs", default="", metavar="str",
                   help="Extra CONFIGS macros")
    p.add_argument("--build-dir", default=None, metavar="dir")

    # search controls
    p.add_argument("--config", choices=["single", "random", "fast", "best"],
                   default=None, metavar="mode",
                   help="Search strategy: single|random|fast|best (default: single)")
    p.add_argument("--max-trials", type=int, default=50, metavar="n",
                   help="Number of random trials for --config=random")
    p.add_argument("--seed", type=int, default=0, metavar="n")

    # roofline / plot
    p.add_argument("--plot", action="store_true",
                   help="Generate roofline PNG")
    p.add_argument("--flops", type=float, default=None, metavar="f",
                   help="Total FLOPs performed by the kernel (required with --plot)")
    p.add_argument("--freq", type=float, default=0, metavar="f",
                   help="Clock frequency in MHz (0 = cycle-domain plot)")
    p.add_argument("--bw", type=float, default=None, metavar="f",
                   help="Peak memory bandwidth in GB/s (default derived)")
    p.add_argument("--output", default="roofline.png", metavar="file")
    p.add_argument("--cache", default="smoke.cache", metavar="file",
                   help="Persistent trial cache (JSONL, append-only). "
                        "Delete the file to start over.")
    p.add_argument("--jobs", type=int, default=1, metavar="n",
                   help="Parallel blackbox jobs (uses --nohup for isolation)")
    p.add_argument("--timeout", type=int, default=3600, metavar="n",
                   help="Per-trial timeout in seconds")

    for name, _, flag, _, is_bool in KNOBS:
        help_text = f"{'bool' if is_bool else 'int'}, single value or comma-list"
        p.add_argument(flag, default=None, metavar="n", help=help_text)

    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Range parsing & validation
# ────────────────────────────────────────────────────────────────────────────

def parse_knob_list(name, raw, is_bool):
    """Parse '4,8,16' → [4,8,16]. Empty / None → None (knob unused)."""
    if raw is None or raw == "":
        return None
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    try:
        if is_bool:
            return [bool(int(x)) for x in parts]
        return [int(x) for x in parts]
    except ValueError:
        sys.exit(f"ERROR: could not parse {name}={raw!r} as int list")


def is_pow2(x):
    return isinstance(x, int) and x > 0 and (x & (x - 1)) == 0


def validate(cfg):
    """Return (ok, reason) for a concrete config dict."""
    for name, _, _, pow2, is_bool in KNOBS:
        v = cfg.get(name)
        if v is None or is_bool:
            continue
        if pow2 and not is_pow2(v):
            return False, f"{name}={v} not pow2"

    iw, w = cfg.get("issue_width"), cfg.get("warps")
    if iw is not None and w is not None and iw > w:
        return False, f"issue_width({iw}) > warps({w})"

    # *_blocks issue at most ISSUE_WIDTH instructions per cycle
    if iw is not None:
        for blk in ("alu_blocks", "fpu_blocks", "lsu_blocks"):
            b = cfg.get(blk)
            if b is not None and b > iw:
                return False, f"{blk}({b}) > issue_width({iw})"

    for pref, line in [("icache", 64), ("dcache", 64), ("l2", 64)]:
        sz = cfg.get(f"{pref}_size")
        wy = cfg.get(f"{pref}_ways")
        if sz and wy:
            if sz % (wy * line) != 0:
                return False, f"{pref}_size/{pref}_ways not aligned"
            sets = sz // (wy * line)
            if not is_pow2(sets):
                return False, f"{pref} sets({sets}) not pow2"
    return True, ""


# ────────────────────────────────────────────────────────────────────────────
# Blackbox runner
# ────────────────────────────────────────────────────────────────────────────

def find_blackbox(args):
    if args.build_dir:
        bdir = os.path.abspath(os.path.expanduser(args.build_dir))
        bb = os.path.join(bdir, "ci", "blackbox.sh")
        if not os.path.isfile(bb):
            sys.exit(f"ERROR: {bb} not found")
        return bb, bdir
    # Prefer the current working directory — user typically `cd build_dir`.
    cwd = os.getcwd()
    bb = os.path.join(cwd, "ci", "blackbox.sh")
    if os.path.isfile(bb):
        return bb, cwd
    for cand in (os.path.join(VORTEX_HOME, "build2_test32"),
                 os.path.join(VORTEX_HOME, "build_test32"),
                 os.path.join(VORTEX_HOME, "build_test64"),
                 os.path.join(VORTEX_HOME, "build"),
                 VORTEX_HOME):
        bb = os.path.join(cand, "ci", "blackbox.sh")
        if os.path.isfile(bb):
            return bb, cand
    return os.path.join(VORTEX_HOME, "ci", "blackbox.sh"), VORTEX_HOME


def build_configs_env(cfg, extra_configs):
    tokens = []
    for name, macro, _, _, is_bool in KNOBS:
        v = cfg.get(name)
        if v is None:
            continue
        if is_bool:
            if v:
                tokens.append(f"-D{macro}")
        else:
            tokens.append(f"-D{macro}={v}")
    if extra_configs:
        tokens.extend(shlex.split(extra_configs))
    return " ".join(tokens)


def run_trial(args, cfg, blackbox, cwd, use_nohup):
    """Run one simulation; return (ipc, instrs, cycles, bytes_or_None, captured_output).
    Output is fully captured (not streamed) so callers can print it atomically."""
    cmd = [blackbox, f"--driver={args.driver}", f"--app={args.app}"]
    if args.app_args:
        cmd.append(f"--args={args.app_args}")
    if args.perf:
        cmd.append(f"--perf={args.perf}")
    if use_nohup:
        cmd.append("--nohup")

    env = os.environ.copy()
    env["CONFIGS"] = (env.get("CONFIGS", "") + " "
                      + build_configs_env(cfg, args.configs)).strip()

    header = (
        "\n── trial ───────────────────────────────────────────────────\n"
        f"CONFIG : { {k:v for k,v in cfg.items() if v is not None} }\n"
        f"CMD    : {' '.join(cmd)}\n"
        f"CONFIGS: {env['CONFIGS']}\n"
    )

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, cwd=cwd, env=env, timeout=args.timeout)
    except subprocess.TimeoutExpired as e:
        partial = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) \
                  else (e.stdout or "")
        return None, None, None, None, (
            header + partial +
            f"\nERROR: TIMEOUT after {args.timeout}s — killed\n")
    output = header + proc.stdout
    if proc.returncode != 0:
        return None, None, None, None, output + f"\nERROR: exit status {proc.returncode}"

    m = re.findall(r"PERF:\s+instrs=(\d+),\s*cycles=(\d+),\s*IPC=([0-9.]+)", proc.stdout)
    if not m:
        return None, None, None, None, output + "\nERROR: no PERF summary line"
    instrs, cycles, ipc = int(m[-1][0]), int(m[-1][1]), float(m[-1][2])

    mem = re.search(r"PERF:\s+memory:.*?read_bytes=(\d+).*?write_bytes=(\d+)", proc.stdout)
    total_bytes = int(mem.group(1)) + int(mem.group(2)) if mem else None

    output += f">>> IPC = {ipc:.4f}  (instrs={instrs}, cycles={cycles})\n"
    return ipc, instrs, cycles, total_bytes, output


# ────────────────────────────────────────────────────────────────────────────
# Runner: cache + lock + optional thread pool
# ────────────────────────────────────────────────────────────────────────────

class Runner:
    def __init__(self, args, blackbox, cwd, cache_by_key, cache_path):
        self.args = args
        self.blackbox = blackbox
        self.cwd = cwd
        self.cache = cache_by_key
        self.cache_path = cache_path
        self.jobs = max(1, args.jobs)
        self.use_nohup = self.jobs > 1
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=self.jobs) if self.jobs > 1 else None
        self.stop_requested = False

    def _one(self, cfg):
        """Runs a single trial OR returns cached. Thread-safe. Returns (ipc, instrs, cycles, bytes, None)."""
        k = cache_key(cfg)
        with self.lock:
            if k in self.cache:
                ipc, rest = self.cache[k]
                print(f"── cache hit ─ IPC={ipc:.4f}  cfg="
                      f"{ {kk:vv for kk,vv in cfg.items() if vv is not None} }")
                sys.stdout.flush()
                return (ipc, *rest)
            if self.stop_requested:
                return (None, None, None, None, None)

        ipc, instrs, cycles, total_bytes, output = run_trial(
            self.args, cfg, self.blackbox, self.cwd, self.use_nohup)

        with self.lock:
            sys.stdout.write(output)
            sys.stdout.flush()
            if ipc is None:
                # fail-fast: mark stop and record error; other in-flight jobs will
                # still finish but no new ones will run.
                self.stop_requested = True
                raise SystemExit(f"ERROR: trial failed — stopping "
                                 f"(cfg={ {k:v for k,v in cfg.items() if v is not None} })")
            rest = [instrs, cycles, total_bytes, None]
            self.cache[k] = (ipc, rest)
            append_cache(self.cache_path, cfg, ipc, instrs, cycles, total_bytes)
        return (ipc, instrs, cycles, total_bytes, None)

    def run_one(self, cfg):
        return self._one(cfg)

    def run_batch(self, cfgs):
        """Run a list of configs; returns list of (cfg, ipc, rest) in input order."""
        if self.pool is None:
            return [self._pack(c, self._one(c)) for c in cfgs]
        futures = {self.pool.submit(self._one, c): i for i, c in enumerate(cfgs)}
        results = [None] * len(cfgs)
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = self._pack(cfgs[i], fut.result())
        return results

    @staticmethod
    def _pack(cfg, full):
        ipc, instrs, cycles, total_bytes, _ = full
        return (cfg, ipc, [instrs, cycles, total_bytes, None])

    def shutdown(self):
        if self.pool:
            self.pool.shutdown(wait=True)


# ────────────────────────────────────────────────────────────────────────────
# Search strategies
# ────────────────────────────────────────────────────────────────────────────

def cache_key(cfg):
    return tuple(sorted((k, v) for k, v in cfg.items() if v is not None))


# ────────────────────────────────────────────────────────────────────────────
# Persistent trial cache (JSONL, append-only)
# ────────────────────────────────────────────────────────────────────────────

def load_cache(path):
    """Load JSONL cache → (map[key]→result, ordered list of entries)."""
    entries = []
    by_key = {}
    if not path or not os.path.isfile(path):
        return by_key, entries
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARN: {path}:{line_no} malformed, skipping")
                continue
            cfg = {k: v for k, v in e.get("config", {}).items() if v is not None}
            entries.append(e)
            by_key[cache_key(cfg)] = (
                e["ipc"], [e["instrs"], e["cycles"], e.get("bytes"), None])
    return by_key, entries


def append_cache(path, cfg, ipc, instrs, cycles, total_bytes):
    if not path:
        return
    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config":    {k: v for k, v in cfg.items() if v is not None},
        "ipc":       ipc,
        "instrs":    instrs,
        "cycles":    cycles,
        "bytes":     total_bytes,
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def enumerate_valid(space):
    """Cartesian product of space (dict name→list), filtered by validate()."""
    names = list(space)
    for combo in itertools.product(*(space[n] for n in names)):
        cfg = dict(zip(names, combo))
        ok, _ = validate(cfg)
        if ok:
            yield cfg


def search_single(args, space, runner):
    cfg = {k: v[0] for k, v in space.items()}
    ok, reason = validate(cfg)
    if not ok:
        sys.exit(f"ERROR: invalid config: {reason}")
    results = runner.run_batch([cfg])
    return results[0]


def search_exhaustive(args, space, runner):
    cfgs = list(enumerate_valid(space))
    if not cfgs:
        sys.exit("ERROR: no valid configs in search space")
    print(f"[best] {len(cfgs)} valid configs")
    results = runner.run_batch(cfgs)
    best = max(results, key=lambda r: r[1])
    print(f"[best] winning IPC = {best[1]:.4f}")
    return best


def search_random(args, space, runner, max_trials):
    rng = rand.Random(args.seed)
    seen = set()
    cfgs = []
    hard_cap = max_trials * 4
    attempts = 0
    while len(cfgs) < max_trials and attempts < hard_cap:
        attempts += 1
        cfg = {n: rng.choice(vals) for n, vals in space.items()}
        if not validate(cfg)[0]:
            continue
        k = cache_key(cfg)
        if k in seen:
            continue
        seen.add(k)
        cfgs.append(cfg)
    if not cfgs:
        sys.exit("ERROR: no valid configs sampled")
    print(f"[random] sampled {len(cfgs)} configs")
    results = runner.run_batch(cfgs)
    best = max(results, key=lambda r: r[1])
    print(f"[random] winning IPC = {best[1]:.4f}")
    return best


def search_fast(args, space, runner):
    """Coordinate descent in KNOBS priority order. Within each knob sweep
    candidates are dispatched in parallel (respecting --jobs). Stops when a
    full pass yields no IPC improvement."""
    # baseline: first value of each knob, else first valid combo
    cur = {n: vals[0] for n, vals in space.items()}
    if not validate(cur)[0]:
        for cfg in enumerate_valid(space):
            cur = cfg
            break
    base_res = runner.run_batch([cur])[0]
    best_ipc = base_res[1]
    best_rest = base_res[2]
    print(f"\n[fast] baseline IPC = {best_ipc:.4f}")

    pass_num = 0
    while True:
        pass_num += 1
        improved = False
        print(f"\n[fast] pass {pass_num}")
        for name, _, _, _, _ in KNOBS:
            if name not in space or len(space[name]) <= 1:
                continue
            base_val = cur[name]
            candidates = []
            for v in space[name]:
                if v == base_val:
                    continue
                trial = dict(cur); trial[name] = v
                if not validate(trial)[0]:
                    continue
                candidates.append((v, trial))
            if not candidates:
                continue
            print(f"[fast] sweep {name}: {len(candidates)} candidate(s)")
            results = runner.run_batch([t for _, t in candidates])
            for (v, _), (cfg_r, ipc_r, rest_r) in zip(candidates, results):
                if ipc_r > best_ipc:
                    best_ipc = ipc_r
                    best_rest = rest_r
                    cur = cfg_r
                    improved = True
                    print(f"    ★ new best IPC = {best_ipc:.4f}  (locked {name}={v})")
        if not improved:
            print(f"\n[fast] converged at pass {pass_num}, best IPC = {best_ipc:.4f}")
            break
    return dict(cur), best_ipc, best_rest


# ────────────────────────────────────────────────────────────────────────────
# Plot
# ────────────────────────────────────────────────────────────────────────────

def plot_roofline(args, cfg, ipc, instrs, cycles, total_bytes, outfile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_cycle = (args.freq == 0)
    freq_hz  = args.freq * 1e6 if args.freq > 0 else 1.0

    cores      = cfg.get("cores", 1)        or 1
    threads    = cfg.get("threads", 4)      or 4
    fpu_blocks = cfg.get("fpu_blocks", 1)   or 1
    mem_bytes  = cfg.get("mem_data_size", 64) or 64
    mem_banks  = cfg.get("mem_banks", 2)    or 2

    peak_flops_per_cycle = 2.0 * cores * threads * fpu_blocks
    peak_bw_GBs = args.bw if args.bw is not None else mem_bytes * mem_banks * freq_hz / 1e9
    peak_bw_per_cycle = peak_bw_GBs * 1e9 / freq_hz if args.freq > 0 else mem_bytes * mem_banks

    flops_total = args.flops
    flops_per_cycle = flops_total / cycles
    ai = (flops_total / total_bytes) if total_bytes else None

    if by_cycle:
        peak_perf, peak_bw, act_perf = peak_flops_per_cycle, peak_bw_per_cycle, flops_per_cycle
        y_label = "Performance (FLOP/cycle)"
        act_str = f"{act_perf:.3f} FLOP/cycle"
        bw_str  = f"{peak_bw:.2f} B/cycle"
    else:
        gflops_peak  = peak_flops_per_cycle * freq_hz / 1e9
        gflops_act   = flops_total / (cycles / freq_hz) / 1e9
        peak_perf, peak_bw, act_perf = gflops_peak, peak_bw_GBs, gflops_act
        y_label = "Performance (GFLOP/s)"
        act_str = f"{act_perf:.3f} GFLOP/s"
        bw_str  = f"{peak_bw:.1f} GB/s"

    ridge = peak_perf / peak_bw

    fig, ax = plt.subplots(figsize=(12, 7))
    ai_range = np.logspace(-3, 4, 3000)
    roof = np.minimum(peak_bw * ai_range, np.full_like(ai_range, peak_perf))
    ax.loglog(ai_range, roof, "b-", linewidth=2.5, label="Roofline")
    ax.axhline(peak_perf, color="blue", linestyle="--", linewidth=1, alpha=0.5,
               label=f"Peak compute {peak_perf:.1f}")
    ax.axvline(ridge, color="green", linestyle=":", alpha=0.6,
               label=f"Ridge {ridge:.2f} FLOP/B")

    if ai is not None:
        ax.plot(ai, act_perf, "ro", markersize=12, zorder=6,
                label=f"Actual AI={ai:.2f}  ({act_str})")

    ax.text(ridge * 0.05 * 1.4, peak_bw * ridge * 0.05 * 1.6,
            bw_str, fontsize=8, color="blue", rotation=35)

    domain = "cycle domain" if by_cycle else f"{args.freq:.0f} MHz"
    cfg_str = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()) if v is not None)
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Vortex Roofline — {args.app} [{args.app_args}] — "
                 f"{args.driver}, {domain}\nIPC={ipc:.3f}  {cfg_str}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    return os.path.abspath(outfile)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.plot and args.flops is None:
        sys.exit("ERROR: --plot requires --flops")

    # Build search space from CLI
    space = {}
    for name, _, _, _, is_bool in KNOBS:
        raw = getattr(args, name.replace("-", "_"))
        vals = parse_knob_list(name, raw, is_bool)
        if vals is not None:
            space[name] = vals

    # Auto-select strategy if not specified
    if args.config is None:
        args.config = "random" if any(len(v) > 1 for v in space.values()) else "single"
    print(f"Strategy: {args.config}")
    print(f"Search space: { {k:v for k,v in space.items()} }")

    blackbox, cwd = find_blackbox(args)
    print(f"Blackbox: {blackbox}\nCWD     : {cwd}")

    cache_by_key, cache_entries = load_cache(args.cache)
    if cache_entries:
        print(f"Cache   : {args.cache} ({len(cache_entries)} prior trials)")

        # --config=single + cache present → replay last entry without running
        if args.config == "single":
            last = cache_entries[-1]
            cfg = {k: v for k, v in last["config"].items() if v is not None}
            print(f"\n[resume] replaying last cache entry "
                  f"({last.get('timestamp','?')}) — IPC={last['ipc']:.4f}")
            ipc = last["ipc"]
            rest = [last["instrs"], last["cycles"], last.get("bytes"), None]
            _print_winner_and_plot(args, cfg, ipc, rest)
            return
    else:
        print(f"Cache   : {args.cache} (new)")
    print()

    runner = Runner(args, blackbox, cwd, cache_by_key, args.cache)
    if runner.jobs > 1:
        print(f"Jobs    : {runner.jobs} (parallel, --nohup for isolation)")

    try:
        if args.config == "single":
            cfg, ipc, rest = search_single(args, space, runner)
        elif args.config == "random":
            cfg, ipc, rest = search_random(args, space, runner, args.max_trials)
        elif args.config == "fast":
            cfg, ipc, rest = search_fast(args, space, runner)
        else:
            cfg, ipc, rest = search_exhaustive(args, space, runner)
    finally:
        runner.shutdown()

    _print_winner_and_plot(args, cfg, ipc, rest)


def _print_winner_and_plot(args, cfg, ipc, rest):
    instrs, cycles, total_bytes, _ = rest
    sep = "═" * 64
    print(f"\n{sep}\n  WINNING CONFIGURATION\n{sep}")
    print(f"  IPC     : {ipc:.4f}")
    print(f"  instrs  : {instrs:,}")
    print(f"  cycles  : {cycles:,}")
    if total_bytes:
        print(f"  bytes   : {total_bytes:,}")
    print(f"  config  :")
    for k, v in sorted(cfg.items()):
        if v is not None:
            print(f"    {k:<20} = {v}")
    repro = [f"{KNOB_BY_NAME[k][2]}={1 if v is True else (0 if v is False else v)}"
             for k, v in cfg.items() if v is not None]
    print(f"\n  Reproduce:")
    print(f"    python3 perf/roofline.py --driver={args.driver} --app={args.app} "
          f"--args='{args.app_args}' --config=single " + " ".join(repro))
    print(sep)

    if args.plot:
        saved = plot_roofline(args, cfg, ipc, instrs, cycles, total_bytes, args.output)
        print(f"\nRoofline saved → {saved}")


if __name__ == "__main__":
    main()
