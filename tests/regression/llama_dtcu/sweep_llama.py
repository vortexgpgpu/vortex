#!/usr/bin/env python3
"""Run llama_dtcu prefill forwards over modes and batch sizes, in parallel, into a CSV.

No building happens here: the binaries in the build tree are used as they are (build them
first with `make all` in build/tests/regression/llama_dtcu, and make sure libsimx.so was
built with the same CONFIGS, i.e. by `make run-simx` of this test or of cgo27_motivation).

  usage: sweep_llama.py --modes 1,7,8 --T 128 [--jobs 4] [--verify] [--fallback 1] [--out file.csv]
         sweep_llama.py --decode --modes 7,8 --T 32,128 ...     (T = B, sequences per decode step;
                                                                 rows go to llama_dtcu_decode_results.csv)

Each (T, mode) is one process. Its [LLAMA] lines become CSV rows (one per site, one total).
A mode that cannot run a site aborts before launching anything and is recorded with
status=illegal unless --fallback is given.
"""

import argparse
import concurrent.futures
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Paths are derived from this file's location so the sweep runs from any checkout; each one
# still takes an environment override for the out-of-tree and _dev layouts.
VORTEX_HOME = Path(__file__).resolve().parents[3]
BUILD = Path(os.environ.get("VX_BUILD", str(VORTEX_HOME / "build"))).resolve()
TEST_DIR = Path(os.environ.get("VX_TEST_DIR", str(BUILD / "tests/regression/llama_dtcu"))).resolve()   # e.g. the _dev copy
RT_DIR = BUILD / "sw/runtime"
MODEL_DIR = Path(os.environ.get("VX_MODEL_DIR",
                                str(Path(__file__).resolve().parents[1] / "llama"))).resolve()

ALL_MODES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15)

def kv(line):
    """Parse a '[LLAMA] k=v k=v ...' line into a dict."""
    return dict(tok.split("=", 1) for tok in line.split()[1:] if "=" in tok)


FIELDS = ("date", "T", "mode", "run", "site", "site_mode", "launches", "cycles", "gemm_cycles", "pass_cycles",
          "cycles_per_token", "tok_per_s_400MHz", "verify_fail_sites", "next_dev", "next_ref", "next_fp32",
          "program_swaps", "wall_s", "status")


def run_point(T, mode, prompt, verify, fallback, timeout, decode=False):
    cmd = [str(TEST_DIR / "llama_dtcu"), "--decode" if decode else "--run", str(MODEL_DIR / "stories15M.bin"),
           "-z", str(MODEL_DIR / "tokenizer.bin"), "-i", prompt, "-B" if decode else "-T", str(T), "-m", str(mode), "--notext"]
    if not verify:
        cmd.append("--noverify")
    if fallback is not None:
        cmd += ["--fallback", str(fallback)]
    env = dict(os.environ, VORTEX_DRIVER="simx", LD_LIBRARY_PATH=f"{RT_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=TEST_DIR, env=env, text=True, capture_output=True, timeout=timeout)
        out = p.stdout + p.stderr
        status = "ok" if p.returncode == 0 else f"exit{p.returncode}"
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "") if isinstance(exc.stdout, str) else ""
        status = "timeout"
    wall = time.time() - t0
    rows = []
    date = time.strftime("%Y-%m-%d_%H:%M:%S")
    tm = None
    for line in out.splitlines():
        if not line.startswith("[LLAMA] "):
            continue
        d = kv(line)
        if "site" in d:
            rows.append(dict(date=date, T=T, mode=mode, run=d.get("run"), site=d["site"], site_mode=d.get("mode"),
                             launches=d.get("launches"), cycles=d.get("cycles"), status=status))
        elif "total_cycles" in d:
            tm = d
            rows.append(dict(date=date, T=T, mode=mode, run=d.get("run"), site="TOTAL", site_mode=mode,
                             launches=d.get("total_launches"), cycles=d.get("total_cycles"),
                             gemm_cycles=d.get("gemm_cycles"), pass_cycles=d.get("pass_cycles"),
                             cycles_per_token=d.get("cycles_per_token"), tok_per_s_400MHz=d.get("tok_per_s_400MHz"),
                             verify_fail_sites=d.get("verify_fail_sites"), next_dev=d.get("next_dev"),
                             next_ref=d.get("next_ref"), next_fp32=d.get("next_fp32"),
                             program_swaps=d.get("program_swaps"), wall_s=d.get("wall_s"), status=status))
    if not rows:
        reason = "illegal" if "is not a multiple" in out or "exceeds" in out or "not runnable" in out else status
        rows.append(dict(date=date, T=T, mode=mode, run=f"uniform:{mode}", site="TOTAL", site_mode=mode,
                         wall_s=f"{wall:.1f}", status=reason))
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        print(f"  T={T} mode={mode}: {reason} {tail}", flush=True)
    else:
        print(f"  T={T} mode={mode}: {status} total_cycles={tm.get('total_cycles') if tm else '?'} "
              f"gemm_cycles={tm.get('gemm_cycles') if tm else '?'} tok/s={tm.get('tok_per_s_400MHz') if tm else '?'} "
              f"wall={wall:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(map(str, ALL_MODES)))
    ap.add_argument("--T", default="128", help="comma-separated batch sizes")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--verify", action="store_true", help="per-site host verification (slower host side)")
    ap.add_argument("--fallback", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--decode", action="store_true", help="batched decode step instead of prefill (T is B)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = str(TEST_DIR / ("llama_dtcu_decode_results.csv" if args.decode else "llama_dtcu_results.csv"))

    modes = [int(m) for m in args.modes.split(",") if m]
    Ts = [int(t) for t in args.T.split(",") if t]
    for m in modes:
        if m not in ALL_MODES:
            sys.exit(f"unsupported mode {m}")
    points = [(T, m) for T in Ts for m in modes]
    print(f"[sweep] {len(points)} runs, jobs={args.jobs}, verify={args.verify}, out={args.out}", flush=True)

    out_path = Path(args.out)
    new_file = not out_path.exists()
    with out_path.open("a", newline="") as f, concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        futures = {pool.submit(run_point, T, m, args.prompt, args.verify, args.fallback, args.timeout, args.decode): (T, m)
                   for T, m in points}
        for fut in concurrent.futures.as_completed(futures):
            for row in fut.result():
                w.writerow(row)
            f.flush()
    print("[sweep] done", flush=True)


if __name__ == "__main__":
    main()
