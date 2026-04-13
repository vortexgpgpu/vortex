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
import itertools
import os
import random as rand
import re
import shlex
import subprocess
import sys

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
    for cand in (os.path.join(VORTEX_HOME, "build_test32"),
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


def run_trial(args, cfg, blackbox, cwd):
    """Run one simulation; return (ipc, instrs, cycles, bytes_or_None, raw_output)."""
    cmd = [blackbox, f"--driver={args.driver}", f"--app={args.app}"]
    if args.app_args:
        cmd.append(f"--args={args.app_args}")
    if args.perf:
        cmd.append(f"--perf={args.perf}")

    env = os.environ.copy()
    env["CONFIGS"] = (env.get("CONFIGS", "") + " "
                      + build_configs_env(cfg, args.configs)).strip()

    print(f"\n── trial ───────────────────────────────────────────────────")
    print(f"CONFIG : { {k:v for k,v in cfg.items() if v is not None} }")
    print(f"CMD    : {' '.join(cmd)}")
    print(f"CONFIGS: {env['CONFIGS']}")
    sys.stdout.flush()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            cwd=cwd, env=env, bufsize=1)
    chunks = []
    for line in proc.stdout:
        sys.stdout.write(line)
        chunks.append(line)
    rc = proc.wait()
    output = "".join(chunks)
    if rc != 0:
        sys.exit(f"ERROR: {args.app} exited with status {rc} — stopping search")

    m = re.findall(r"PERF:\s+instrs=(\d+),\s*cycles=(\d+),\s*IPC=([0-9.]+)", output)
    if not m:
        sys.exit("ERROR: no PERF summary line found (enable --perf=1)")
    instrs, cycles, ipc = m[-1]
    instrs, cycles, ipc = int(instrs), int(cycles), float(ipc)

    mem = re.search(r"PERF:\s+memory:.*?read_bytes=(\d+).*?write_bytes=(\d+)", output)
    total_bytes = int(mem.group(1)) + int(mem.group(2)) if mem else None

    print(f">>> IPC = {ipc:.4f}  (instrs={instrs}, cycles={cycles})")
    return ipc, instrs, cycles, total_bytes, output


# ────────────────────────────────────────────────────────────────────────────
# Search strategies
# ────────────────────────────────────────────────────────────────────────────

def cache_key(cfg):
    return tuple(sorted(cfg.items()))


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
    ipc, *rest = runner(cfg)
    return cfg, ipc, rest


def search_exhaustive(args, space, runner):
    best = None
    for cfg in enumerate_valid(space):
        ipc, *rest = runner(cfg)
        if best is None or ipc > best[1]:
            best = (dict(cfg), ipc, rest)
            print(f"    ★ new best IPC = {ipc:.4f}")
    if best is None:
        sys.exit("ERROR: no valid configs in search space")
    return best


def search_random(args, space, runner, max_trials):
    rng = rand.Random(args.seed)
    seen = set()
    best = None
    trials = 0
    # cap at total valid space
    hard_cap = max_trials * 4
    attempts = 0
    while trials < max_trials and attempts < hard_cap:
        attempts += 1
        cfg = {n: rng.choice(vals) for n, vals in space.items()}
        ok, _ = validate(cfg)
        if not ok:
            continue
        k = cache_key(cfg)
        if k in seen:
            continue
        seen.add(k)
        trials += 1
        print(f"\n[random {trials}/{max_trials}]")
        ipc, *rest = runner(cfg)
        if best is None or ipc > best[1]:
            best = (dict(cfg), ipc, rest)
            print(f"    ★ new best IPC = {ipc:.4f}")
    if best is None:
        sys.exit("ERROR: no valid configs sampled")
    return best


def search_fast(args, space, runner):
    """Coordinate descent in KNOBS priority order. Stops when a full pass
    yields no IPC improvement."""
    seen = {}
    def eval_cfg(cfg):
        k = cache_key(cfg)
        if k in seen:
            print(f"    (cached IPC={seen[k][0]:.4f})")
            return seen[k]
        ok, reason = validate(cfg)
        if not ok:
            print(f"    (skip invalid: {reason})")
            return None
        ipc, *rest = runner(cfg)
        seen[k] = (ipc, rest)
        return seen[k]

    # baseline: first value of each knob
    cur = {n: vals[0] for n, vals in space.items()}
    if not validate(cur)[0]:
        # pick first valid combo
        for cfg in enumerate_valid(space):
            cur = cfg
            break
    res = eval_cfg(cur)
    if res is None:
        sys.exit("ERROR: baseline invalid")
    best_ipc = res[0]
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
            for v in space[name]:
                if v == base_val:
                    continue
                trial = dict(cur); trial[name] = v
                print(f"\n[fast] sweep {name}: {base_val} → {v}")
                r = eval_cfg(trial)
                if r is None:
                    continue
                if r[0] > best_ipc:
                    best_ipc = r[0]
                    cur = trial
                    improved = True
                    print(f"    ★ new best IPC = {best_ipc:.4f}  (locked {name}={v})")
        if not improved:
            print(f"\n[fast] converged at pass {pass_num}, best IPC = {best_ipc:.4f}")
            break
    return dict(cur), best_ipc, seen[cache_key(cur)][1]


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
    print(f"Blackbox: {blackbox}\nCWD     : {cwd}\n")

    def runner(cfg):
        return run_trial(args, cfg, blackbox, cwd)

    if args.config == "single":
        cfg, ipc, rest = search_single(args, space, runner)
    elif args.config == "random":
        cfg, ipc, rest = search_random(args, space, runner, args.max_trials)
    elif args.config == "fast":
        cfg, ipc, rest = search_fast(args, space, runner)
    else:
        cfg, ipc, rest = search_exhaustive(args, space, runner)

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
