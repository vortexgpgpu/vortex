#!/usr/bin/env python3
# Experiment 2 (cgo27_motivation): knob sweep at a large fixed size.
# Sweeps DTCU sim latency-model knobs (each value = a sim rebuild) and records
# per-mode cycles. Coarse one-factor-at-a-time (OFAT) first; go fine (grid) only
# where a knob shows high sensitivity.
#
# Run explicitly:
#   python3 sweep_exp2.py --size 8 --apps 1,3      # --size is a ladder rung
#
# Knobs are plain -D macros consumed by sim/simx/dtcu/dtcu_params.h (#ifndef).
# They are injected through MOTI_EXTRA_CONFIGS in this test's Makefile. Exporting
# CONFIGS directly would override the Makefile's complete machine configuration.
import argparse, csv, fcntl, os, re, shutil, subprocess, sys
from pathlib import Path

BUILD_DIR = os.environ.get(
    "VX_BUILD", "/nethome/sjeong306/vortex_scheduler/vortex/build")
SOURCE_DIR = Path(__file__).resolve().parent
TEST_DIR = Path(BUILD_DIR) / "tests/regression/cgo27_motivation"

# Must match main.cpp's kShortNames[] EXACTLY; run_case() cross-checks and exits on a
# mismatch. Keep in sync with sweep_exp1.py.
MODES = {0:"SIMT",1:"TCU",2:"TCU+DXA",
         3:"TCU_wg+DXA",4:"TCU_wg",5:"TCU_wg+Acol",6:"TCU_wg+Acol_SB",
         7:"DTCU_socket",8:"DTCU_cluster",
         9:"TCU+DTCU_socket",10:"TCU+DTCU_cluster",11:"TCU+DTCU_both",
         14:"DTCU_socket_pipe",15:"DTCU_cluster_pipe"}
# 12 and 13 are reserved. 9-11 are planned but not built.

# See sweep_exp1.py for why the ladder lives in the scripts: `-s` is gone from the
# harness because two DTCU engines with different tiles leave no single native tile to
# multiply. Keep these two in sync.
REF_TILE_M, REF_TILE_N, REF_TILE_K = 64, 32, 16   # fp16 in / fp32 out

def shape_for(rung):
    """Ladder rung -> (M, N, K). Rung r reproduces what `-s r` used to produce."""
    return REF_TILE_M * rung, REF_TILE_N * rung, REF_TILE_K * rung
APPS  = {1:"baseline",2:"relu",5:"scale",4:"residual",3:"gelu",
         6:"softmax",9:"column_mean_broadcast"}

# Coarse OFAT grid. Default (baseline) value listed first; sweep holds all other
# knobs at their sim default while varying one.
KNOBS = {
    "DTCU_SWIZZLE":        [0, 1],
    "DTCU_MACS_PER_CYCLE": [8, 16, 32, 64],
    "DTCU_SMEM_BANKS":     [2, 4, 8],
    "DTCU_MAX_OUTSTANDING":[1, 4, 16],
}
# Modes whose cycles a DTCU knob can move (record these; others for context). The knobs
# below are engine-internal, so they move 7 and 8 and leave the in-core modes alone.
# NOTE: DTCU_MACS_PER_CYCLE alone does nothing -- the accumulator floor
# (2*tile_m*tile_n / DTCU_ACC_BANKS) ties the compute model at the default width, so it
# has to be swept together with DTCU_ACC_BANKS to move anything.
DTCU_MODES = [7, 8, 14, 15]

# No size= group: the harness no longer has -s (the script knows its own rung). `name=`
# is required so a stale MODES table hard-errors instead of mislabelling a CSV column.
MOTI_RE = re.compile(
    r'\[MOTI\]\s+app=(?P<app>\d+)\s+M=(?P<M>\d+)\s+N=(?P<N>\d+)\s+K=(?P<K>\d+)\s+'
    r'mode=(?P<mode>\d+)\s+name=(?P<name>\S+)\s+cycles=(?P<cycles>\d+)\s+'
    r'errors=(?P<errors>-?\d+)(?:\s+skipped=(?P<skipped>\d+))?')

def run_case(app, size, extra_configs="", timeout=10800):
    env = dict(os.environ)
    env["MOTI_APP"] = str(app)
    env["MOTI_EXTRA_CONFIGS"] = extra_configs
    M, N, K = shape_for(size)
    cmd = ["./ci/blackbox.sh", "--driver=simx", "--app=cgo27_motivation",
           "--perf=1", f"--args=-a {app} -M {M} -N {N} -K {K}"]
    p = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True,
                       timeout=timeout, env=env)
    out = p.stdout + p.stderr
    if p.returncode != 0:
        tail = "\n".join(out.splitlines()[-80:])
        raise RuntimeError(f"build/run failed ({p.returncode}) for app={app} "
                           f"size={size} cfg='{extra_configs}'\n{tail}")
    res = {}
    for m in MOTI_RE.finditer(out):
        mode = int(m.group("mode"))
        if m.group("skipped"):
            continue  # engine absent: not a 0-cycle datapoint
        name = m.group("name")
        if MODES.get(mode) != name:
            sys.exit(f"FATAL: harness reports mode {mode} as '{name}', MODES says "
                     f"'{MODES.get(mode)}'. Update MODES before trusting the CSV.")
        res[mode] = (int(m.group("cycles")), int(m.group("errors")))
    if not res or "PASSED!" not in out or any(errors != 0 for _, errors in res.values()):
        raise RuntimeError(f"invalid result for app={app} size={size} "
                           f"cfg='{extra_configs}'")
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8, help="large ladder rung for exp2")
    ap.add_argument("--apps", default="1,3")
    ap.add_argument("--out",  default="exp2_results.csv")
    args = ap.parse_args()
    apps = [int(x) for x in args.apps.split(",")]
    bad_apps = [app for app in apps if app not in APPS]
    if bad_apps:
        raise SystemExit(f"unsupported apps: {bad_apps}; choose from {tuple(APPS)}")

    rows = []
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_DIR / "Makefile", TEST_DIR / "Makefile")
    lock_path = TEST_DIR / ".moti-build.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another cgo27 build holds .moti-build.lock") from exc

        # Every point changes the simulator binary, so Exp 2 is intentionally serial.
        for app in apps:
            print(f"[baseline] app={app} size={args.size}")
            res = run_case(app, args.size)
            for m in sorted(res):
                rows.append(["<default>", "", app, args.size, m, MODES[m], *res[m]])

        for knob, vals in KNOBS.items():
            for v in vals:
                cfg = f"-D{knob}={v}"
                for app in apps:
                    print(f"[knob] {cfg} app={app} size={args.size}")
                    res = run_case(app, args.size, extra_configs=cfg)
                    for m in sorted(res):
                        rows.append([knob, v, app, args.size, m, MODES[m], *res[m]])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["knob", "value", "app", "size", "mode", "mode_name", "cycles", "errors"])
        w.writerows(rows)
    print(f"\nCSV -> {args.out}")

    # sensitivity summary: for each knob, DTCU-mode cycle spread across values
    print("\n===== KNOB SENSITIVITY (DTCU modes) =====")
    for knob in KNOBS:
        for app in apps:
            for m in DTCU_MODES:
                pts = [(r[1], r[6]) for r in rows
                       if r[0] == knob and r[2] == app and r[4] == m and r[7] == 0]
                if len(pts) >= 2:
                    cyc = [c for _, c in pts]
                    spread = max(cyc) / min(cyc) if min(cyc) else 0
                    flag = "  <-- high, go fine" if spread >= 1.15 else ""
                    print(f"{knob:<22} app{app} {MODES[m]:<10} "
                          f"spread={spread:.2f}x over {[v for v,_ in pts]}{flag}")

if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1)
