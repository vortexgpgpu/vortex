#!/usr/bin/env python3
# Experiment 1 (cgo27_motivation): size x app sweep.
# Goal: for each app, find the sizes where the best (min-cycle) HW mode CHANGES.
#
# GENERATED FOR REVIEW — does not auto-run in the build. Run explicitly:
#   python3 sweep_exp1.py                 # default grid
#   python3 sweep_exp1.py --sizes 1,2,4,8,16 --apps 1,6,7
#
# Contract with the harness (implemented in main.cpp, Phase B):
#   ./cgo27_motivation -a <app_id> -M <m> -N <n> -K <k>
#     runs all HW modes for that (app,shape) and prints, per mode, a line:
#       [MOTI] app=<id> M=<M> N=<N> K=<K> mode=<m> name=<n> cycles=<c> errors=<e>
#     plus a trailing " skipped=1" when that mode's engine is absent from the build.
#
# `name=` is REQUIRED, not decorative. MODES below supplies the CSV column headers, so a
# stale table silently mislabels every column; cross-checking the harness's own name
# against MODES turns that into a hard error instead. An older binary emits no `name=`,
# so the regex matches nothing and the "no [MOTI] lines" WARN fires -- a loud failure
# rather than a wrong CSV.
import argparse, csv, os, re, subprocess, sys

BUILD_DIR = os.environ.get(
    "VX_BUILD", "/export/nethomes/sjeong306/vortex_scheduler/vortex/build")

APPS  = {1:"baseline",2:"relu",3:"gelu",4:"residual",5:"scale",
         6:"softmax",7:"dq+bias+gelu",8:"dq+softmax"}
# Must match main.cpp's kShortNames[] EXACTLY -- run_case() cross-checks and exits on a
# mismatch. Modes 3/4 used to be DTCU-without-TMA vs DTCU-with-TMA; that pair is retired
# (DTENSOR_FLAG_NO_TMA stays in the ISA, only the harness mode is gone) and the two
# indices now hold the two placement variants.
MODES = {0:"SIMT",1:"TCU",2:"TCU+DXA",3:"DTCU_cluster",4:"DTCU_socket",
         5:"TCU-pipe",6:"TCU+DXA-pipe"}

# The size ladder lives here now, not in the harness. `-s` used to mean "N x the DTCU
# native tile", but with two DTCU engines whose tiles differ (cluster 64x128, socket
# 32x16) there is no single native tile left to multiply, so the harness takes only
# absolute -M/-N/-K. Every mode must still run the SAME GEMM for the cycle comparison to
# mean anything, so one reference tile defines the ladder for all of them; the cluster
# tile is used because that is what the historical `-s` numbers were built on, which
# keeps old sweep data comparable.
REF_TILE_M, REF_TILE_N, REF_TILE_K = 64, 32, 16   # fp16 in / fp32 out

def shape_for(rung):
    """Ladder rung -> (M, N, K). Rung r reproduces what `-s r` used to produce."""
    return REF_TILE_M * rung, REF_TILE_N * rung, REF_TILE_K * rung


# No size= group: the harness no longer has -s, and the script already knows the rung it
# asked for. Named groups, because dropping a field is what made the old positional
# numbering a renumbering hazard in the first place.
MOTI_RE = re.compile(
    r'\[MOTI\]\s+app=(?P<app>\d+)\s+M=(?P<M>\d+)\s+N=(?P<N>\d+)\s+K=(?P<K>\d+)\s+'
    r'mode=(?P<mode>\d+)\s+name=(?P<name>\S+)\s+cycles=(?P<cycles>\d+)\s+'
    r'errors=(?P<errors>-?\d+)(?:\s+skipped=(?P<skipped>\d+))?')

def run_case(app, size, timeout=7200):
    """Run the harness once for (app, ladder rung); return {mode: (cycles, errors)}.

    Skipped modes are DROPPED, not recorded as cycles=0. The harness emits
    `cycles=0 errors=0 skipped=1` for a mode whose engine is absent from the build, and
    admitting that row would let it WIN the min-cycles best-mode pick below.
    """
    M, N, K = shape_for(size)
    cmd = ["./ci/blackbox.sh", "--driver=simx", "--app=cgo27_motivation",
           "--perf=1", f"--args=-a {app} -M {M} -N {N} -K {K}"]
    p = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True,
                       timeout=timeout)
    out = p.stdout + p.stderr
    res = {}
    for m in MOTI_RE.finditer(out):
        mode = int(m.group("mode"))
        if m.group("skipped"):
            print(f"  note: mode {mode} skipped (engine absent) app={app} size={size}",
                  file=sys.stderr)
            continue
        name = m.group("name")
        if MODES.get(mode) != name:
            sys.exit(f"FATAL: harness reports mode {mode} as '{name}', MODES says "
                     f"'{MODES.get(mode)}'. Update the MODES table in this script "
                     f"before trusting any CSV it writes.")
        res[mode] = (int(m.group("cycles")), int(m.group("errors")))
    if not res:
        print(f"  WARN: no [MOTI] lines for app={app} size={size} "
              f"(M={M} N={N} K={K}) -- stale binary missing name= ?", file=sys.stderr)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1,2,4,8,16")
    ap.add_argument("--apps",  default=",".join(map(str, APPS)))
    ap.add_argument("--modes", default=",".join(map(str, MODES)))
    ap.add_argument("--out",   default="exp1_results.csv")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    apps  = [int(x) for x in args.apps.split(",")]
    modes = [int(x) for x in args.modes.split(",")]

    # table[(app,size)][mode] = cycles ; also track correctness.
    table = {}
    for app in apps:
        for size in sizes:
            M, N, K = shape_for(size)
            print(f"[run] app={app} ({APPS[app]}) rung={size} -> {M}x{N}x{K}")
            res = run_case(app, size)
            table[(app, size)] = res
            for m in modes:
                if m in res:
                    c, e = res[m]
                    tag = "" if e == 0 else f"  !! errors={e}"
                    print(f"    mode {m:<2} {MODES[m]:<14} cycles={c}{tag}")

    # write CSV
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["app", "app_name", "size"] + [MODES[m] for m in modes] + ["best_mode"])
        for app in apps:
            for size in sizes:
                res = table[(app, size)]
                row = [app, APPS[app], size]
                best_m, best_c = None, None
                for m in modes:
                    c = res.get(m, (None, None))[0]
                    row.append(c if c is not None else "")
                    # best = min cycles among modes that ran correctly
                    if c is not None and res[m][1] == 0 and (best_c is None or c < best_c):
                        best_c, best_m = c, m
                row.append(f"{best_m}:{MODES.get(best_m,'?')}" if best_m is not None else "")
                w.writerow(row)
    print(f"\nCSV -> {args.out}")

    # crossover report: per app, where best-mode changes across sizes
    print("\n===== CROSSOVERS (best HW mode changes with size) =====")
    for app in apps:
        seq = []
        for size in sizes:
            res = table[(app, size)]
            cand = [(res[m][0], m) for m in modes if m in res and res[m][1] == 0]
            seq.append((size, min(cand)[1] if cand else None))
        flips = [f"size {seq[i-1][0]}->{seq[i][0]}: mode {seq[i-1][1]}->{seq[i][1]}"
                 for i in range(1, len(seq)) if seq[i][1] != seq[i-1][1]]
        line = " ".join(f"{s}:{MODES.get(m,'?')}" for s, m in seq)
        print(f"app {app} ({APPS[app]:<14}) best-by-size: {line}")
        for fl in flips:
            print(f"    * {fl}")

if __name__ == "__main__":
    main()
