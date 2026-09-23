#!/usr/bin/env python3
"""Turn a llama_dtcu_results.csv (written by sweep_llama.py) into the site x mode table of the
llama-c prefill experiment.

  usage: llama_table.py [--csv file] [--T 128[,64,256]] [--md]

For each T: one column per run (uniform:<mode> -> "m<mode>", a --modemap run -> "map<k>", legend
below the table), one row per GEMM site and per pass, then GEMM sum, pass sum, total, cycles per
token and tok/s at 400 MHz. The last row block is the oracle: for every GEMM site the cheapest
uniform run whose own mode ran that site (fallback sites are excluded), summed with the passes of
the best uniform run. Rows with the same (T, run, site) keep the latest one in the file.
"""
import argparse
import csv
from collections import OrderedDict
from pathlib import Path

DEFAULT_CSV = str(Path(__file__).resolve().parents[3] / "build/tests/regression/llama_dtcu/llama_dtcu_results.csv")
MODE_NAME = {0: "SIMT", 1: "TCU", 2: "TCU+DXA", 3: "TCU_wg+DXA", 4: "TCU_wg", 5: "TCU_wg+Acol", 6: "TCU_wg+Acol_SB",
             7: "DTCU_socket", 8: "DTCU_cluster", 14: "DTCU_socket_pipe", 15: "DTCU_cluster_pipe"}
FREQ = 400e6


SITE_ORDER = []   # site names in order of first appearance in the file


def col_key(r):
    """Column of a row: 'uniform:<m>' when the sweep ran mode m (its fallback sites included,
    marked in the table), otherwise the run tag of a --modemap run."""
    m = r.get("mode", "")
    return f"uniform:{int(m)}" if m.isdigit() else r["run"]


def load(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        if not r.get("run") or not (r.get("T") or "").isdigit():   # also skips stray header lines
            continue
        rows[(int(r["T"]), col_key(r), r["site"])] = r      # latest wins
        if r["site"] not in SITE_ORDER:
            SITE_ORDER.append(r["site"])
    return rows


def site_lists(rows, T):
    """GEMM sites (a mode ran them) and passes (site_mode simt) present at this T, file order."""
    present = {site: r for (t, _, site), r in rows.items() if t == T and site != "TOTAL"}
    gemm = [s for s in SITE_ORDER if s in present and present[s].get("site_mode") != "simt"]
    passes = [s for s in SITE_ORDER if s in present and present[s].get("site_mode") == "simt"]
    return gemm, passes


def run_key(run):
    if run.startswith("uniform:"):
        return (0, int(run.split(":")[1]))
    return (1, run)


def fmt_cyc(v):
    return f"{v/1e6:.2f}" if v is not None else "-"


def table(rows, T, md):
    runs = sorted({run for (t, run, _) in rows if t == T}, key=run_key)
    if not runs:
        return
    GEMM_SITES, PASS_SITES = site_lists(rows, T)
    label, legend = {}, []
    k = 0
    for run in runs:
        if run.startswith("uniform:"):
            label[run] = "m" + run.split(":")[1]
        else:
            k += 1
            label[run] = f"map{k}"
            legend.append(f"{label[run]} = {run}")

    def cell(run, site):
        r = rows.get((T, run, site))
        return r

    def site_cycles(run, site):
        r = cell(run, site)
        return int(r["cycles"]) if r and r.get("cycles") else None

    lines = []
    head = ["site"] + [label[r] for r in runs]
    lines.append(head)
    # mode actually used per GEMM site (marks fallback with '*')
    for site in GEMM_SITES:
        row = [site]
        for run in runs:
            r = cell(run, site)
            if r is None:
                row.append("-")
                continue
            mark = ""
            if run.startswith("uniform:") and r.get("site_mode") and r["site_mode"] != run.split(":")[1]:
                mark = f"*{r['site_mode']}"     # this site fell back to that mode
            row.append(fmt_cyc(int(r["cycles"])) + mark)
        lines.append(row)
    gemm = {run: sum(site_cycles(run, s) or 0 for s in GEMM_SITES) for run in runs}
    lines.append(["GEMM sum"] + [fmt_cyc(gemm[r]) for r in runs])
    for site in PASS_SITES:
        lines.append([site] + [fmt_cyc(site_cycles(r, site)) for r in runs])
    passes = {run: sum(site_cycles(run, s) or 0 for s in PASS_SITES) for run in runs}
    lines.append(["pass sum"] + [fmt_cyc(passes[r]) for r in runs])
    tot = {}
    for run in runs:
        r = cell(run, "TOTAL")
        tot[run] = int(r["cycles"]) if r and r.get("cycles") else None
    lines.append(["total"] + [fmt_cyc(tot[r]) for r in runs])
    lines.append(["cyc/token"] + [f"{tot[r]/T:,.0f}" if tot[r] else "-" for r in runs])
    lines.append(["tok/s@400MHz"] + [f"{FREQ/(tot[r]/T):,.0f}" if tot[r] else "-" for r in runs])
    lines.append(["status"] + [(cell(r, "TOTAL") or {}).get("status", "-") for r in runs])
    lines.append(["verify_fail"] + [(cell(r, "TOTAL") or {}).get("verify_fail_sites", "-") or "-" for r in runs])

    # oracle over uniform runs (own-mode sites only)
    def has_fallback(run):
        return any((cell(run, s) or {}).get("site_mode") not in (None, "", run.split(":")[1]) for s in GEMM_SITES)
    uni = [r for r in runs if r.startswith("uniform:") and tot.get(r)]            # own-mode sites feed the oracle
    strict = [r for r in uni if not has_fallback(r)]                              # truly uniform: every site on its mode
    oracle = []
    for site in GEMM_SITES:
        best = None
        for run in uni:
            r = cell(run, site)
            if not r or not r.get("cycles") or r.get("site_mode") != run.split(":")[1]:
                continue
            c = int(r["cycles"])
            if best is None or c < best[0]:
                best = (c, run.split(":")[1])
        oracle.append((site, best))
    best_uni = min(strict, key=lambda r: tot[r]) if strict else None

    kind = "B sequences, decode step" if "cls" in GEMM_SITES else "prefill rows"
    title = f"T={T} ({kind}), cycles in millions; '*m' = that site fell back to mode m"
    if md:
        print(f"\n**{title}**\n")
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for row in lines[1:]:
            print("| " + " | ".join(row) + " |")
    else:
        print(f"\n{title}")
        w = [max(len(str(row[i])) for row in lines) for i in range(len(head))]
        for row in lines:
            print("  ".join(str(c).rjust(w[i]) if i else str(c).ljust(w[i]) for i, c in enumerate(row)))
    for run in runs:
        if run.startswith("uniform:"):
            fb = [s for s in GEMM_SITES if (cell(run, s) or {}).get("site_mode") not in (None, "", run.split(":")[1])]
            if fb:
                legend.append(f"{label[run]}: {', '.join(fb)} fell back to mode {(cell(run, fb[0]) or {}).get('site_mode')} (illegal shape for mode {run.split(':')[1]})")
    for l in legend:
        print(("  " if not md else "") + l)
    if best_uni and all(b for _, b in oracle):
        og = sum(b[0] for _, b in oracle)
        ot = og + passes[best_uni]
        print(f"\nbest uniform: {label[best_uni]} ({MODE_NAME.get(int(best_uni.split(':')[1]), '?')}) total {fmt_cyc(tot[best_uni])} M, "
              f"GEMM {fmt_cyc(gemm[best_uni])} M, {FREQ/(tot[best_uni]/T):,.0f} tok/s")
        print("oracle (per-site cheapest own-mode run): " + ", ".join(f"{s}:m{b[1]}" for s, b in oracle))
        print(f"oracle GEMM {fmt_cyc(og)} M ({100*(og/gemm[best_uni]-1):+.1f}% vs best uniform), "
              f"total {fmt_cyc(ot)} M ({100*(ot/tot[best_uni]-1):+.1f}%), {FREQ/(ot/T):,.0f} tok/s")
        for run in runs:
            if tot.get(run) and (not run.startswith("uniform:") or has_fallback(run)):
                what = label[run] + (" (with fallbacks)" if run.startswith("uniform:") else "")
                print(f"{what}: total {fmt_cyc(tot[run])} M ({100*(tot[run]/tot[best_uni]-1):+.1f}% vs best uniform), "
                      f"GEMM {fmt_cyc(gemm[run])} M, {FREQ/(tot[run]/T):,.0f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--T", default=None, help="comma-separated; default: every T in the file")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    rows = load(a.csv)
    Ts = [int(t) for t in a.T.split(",")] if a.T else sorted({t for (t, _, _) in rows})
    for T in Ts:
        table(rows, T, a.md)


if __name__ == "__main__":
    main()
