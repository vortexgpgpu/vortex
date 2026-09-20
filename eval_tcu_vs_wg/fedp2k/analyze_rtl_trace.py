#!/usr/bin/env python3
"""Attribute rtlsim cycles from DBG_TRACE_DXA + DBG_TRACE_PIPELINE traces.

Input: blackbox rtlsim log with lines of the form
    <cycle>: cluster0-socket0-core0-... commit: wid=W, cta_id=C, sid=S, PC=0xP, ex=EX, ... eop=E, ... (#uuid)
    <cycle>: cluster0-socket0-dxa-worker0 setup-done: core=.., wid=.., bar=..
    <cycle>: cluster0-socket0-dxa-worker0 gmem-req-fire
    <cycle>: cluster0-socket0-dxa-worker0 <complete-ish event>

Outputs a JSON summary: DXA busy spans/occupancy, per-warp commit-gap
attribution by ex-type and by PC, and overlap classes.
"""
import sys, re, json, gzip
from collections import defaultdict

def openf(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")

RE_COMMIT = re.compile(
    r"^\s*(\d+):\s+\S*core0\S*\s+commit: wid=(\d+), cta_id=(\d+), sid=\d+, "
    r"PC=0x([0-9a-fA-F]+), ex=(\w+)")
RE_DXA = re.compile(r"^\s*(\d+):\s+(\S*dxa-worker\d+)\s+(\S+)")
RE_BAR = re.compile(
    r"^\s*(\d+):\s+\S*bar\S*\s+req: wid=(\d+), bar_id=(\d+), is_global=\d, "
    r"is_event=(\d), is_arrive=(\d), is_sync=(\d)")

def main(path):
    commits = defaultdict(list)          # wid -> [(cycle, pc, ex)]
    dxa_events = defaultdict(list)       # worker -> [(cycle, event)]
    bar_reqs = []                        # (cycle, wid, bar_id, ev, arr, sync)
    tmin, tmax = None, 0

    with openf(path) as f:
        for line in f:
            m = RE_COMMIT.match(line)
            if m:
                cyc = int(m.group(1)); wid = int(m.group(2))
                pc = m.group(4); ex = m.group(5)
                commits[wid].append((cyc, pc, ex))
                if tmin is None or cyc < tmin: tmin = cyc
                if cyc > tmax: tmax = cyc
                continue
            m = RE_DXA.match(line)
            if m:
                dxa_events[m.group(2)].append((int(m.group(1)), m.group(3)))
                continue
            m = RE_BAR.match(line)
            if m:
                bar_reqs.append(tuple(int(m.group(i)) for i in range(1, 7)))

    span = (tmax - tmin) if tmin is not None else 0

    # ── DXA busy: spans from setup-done to the last event before next setup ──
    dxa_summary = {}
    for w, evs in dxa_events.items():
        evs.sort()
        busy = 0
        spans = []
        start = None
        last = None
        n_setup = sum(1 for _, e in evs if e.startswith("setup-done"))
        n_fire  = sum(1 for _, e in evs if e.startswith("gmem-req-fire"))
        for cyc, e in evs:
            if e.startswith("setup-done"):
                if start is not None and last is not None:
                    spans.append((start, last)); busy += last - start
                start = cyc
            last = cyc
        if start is not None and last is not None:
            spans.append((start, last)); busy += last - start
        dxa_summary[w] = {
            "transfers": n_setup, "gmem_fires": n_fire,
            "busy_cycles": busy, "busy_frac_of_span": busy / span if span else 0,
            "avg_transfer_cycles": busy / n_setup if n_setup else 0,
            "spans": spans[:8],
        }

    # ── per-warp gap attribution: gap charged to the NEXT commit's ex/PC ──
    gap_by_ex = defaultdict(int)
    gap_by_pc = defaultdict(int)
    commit_count_by_ex = defaultdict(int)
    total_gap = 0
    for wid, lst in commits.items():
        lst.sort()
        for (c1, _, _), (c2, pc2, ex2) in zip(lst, lst[1:]):
            gap = c2 - c1
            gap_by_ex[ex2] += gap
            gap_by_pc[(pc2, ex2)] += gap
            total_gap += gap
        for _, _, ex in lst:
            commit_count_by_ex[ex] += 1

    top_pcs = sorted(gap_by_pc.items(), key=lambda kv: -kv[1])[:25]

    # ── overlap: fraction of DXA busy time during which TCU commits occur ──
    tcu_cycles = sorted(c for lst in commits.values() for (c, _, ex) in lst if ex == "TCU")
    def count_in(spans, arr):
        import bisect
        n = 0
        for a, b in spans:
            n += bisect.bisect_right(arr, b) - bisect.bisect_left(arr, a)
        return n
    all_spans = [s for w in dxa_summary.values() for s in w["spans"]]

    out = {
        "trace_span": [tmin, tmax, span],
        "warps": len(commits),
        "dxa": dxa_summary,
        "commit_counts_by_ex": dict(commit_count_by_ex),
        "gap_cycles_by_next_ex": dict(sorted(gap_by_ex.items(), key=lambda kv: -kv[1])),
        "total_gap_cycles_all_warps": total_gap,
        "top_gap_pcs": [{"pc": pc, "ex": ex, "gap_cycles": g} for (pc, ex), g in top_pcs],
        "bar_reqs": len(bar_reqs),
        "tcu_commits_in_first_dxa_spans": count_in(all_spans, tcu_cycles),
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main(sys.argv[1])
