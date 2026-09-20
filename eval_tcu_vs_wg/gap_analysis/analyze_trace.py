#!/usr/bin/env python3
"""Timeline analysis of a simx --debug=3 trace for the WGMMA+DXA gap study.

Extracts:
  - DXA worker busy spans ([start, complete] per transfer)
  - TCU activity (cycles with >=1 uop execute)
  - per-warp barrier suspend/resume spans (cycle taken from the nearest
    preceding TRACE line; simx logs are emitted in tick order)
  - overlap statistics: DXA-only / TCU-only / both / neither
  - binned timeline for plotting
Outputs a JSON summary next to the trace.
"""
import re, sys, json, gzip
from collections import defaultdict

def openf(p):
    return gzip.open(p, 'rt', errors='replace') if p.endswith('.gz') else open(p, errors='replace')

def main(path, out_json, bin_size=256):
    re_trace = re.compile(r'^TRACE\s+(\d+): (.*)$')
    re_dxa   = re.compile(r'^(\S*dxa\S*)\[(\d+)\] (start|complete): (.*)$')
    re_tcu   = re.compile(r'^(\S*tcu\S*) execute: op=(\S+?),')
    re_susp  = re.compile(r'^DEBUG \*\*\* Suspend core #(\d+), warp #(\d+) at barrier #(\d+)')
    re_resu  = re.compile(r'^DEBUG \*\*\* Resume core #(\d+), warp #(\d+) at barrier #(\d+)')
    re_barr  = re.compile(r'^DEBUG \*\*\* Barrier arrive: core #(\d+), warp #(\d+) at barrier #(\d+)')
    re_sched = re.compile(r'schedule: wid=(\d+)')

    cur_cycle = 0
    max_cycle = 0
    dxa_spans = []            # (start, end, worker, desc)
    dxa_open = {}             # worker -> (start, desc)
    tcu_exec_cycles = defaultdict(int)   # cycle -> uop count
    tcu_ops = defaultdict(int)
    warp_susp = {}            # (core,wid) -> (cycle, bar)
    susp_spans = []           # (start, end, core, wid, bar)
    barrier_arrivals = 0
    sched_events = defaultdict(int)  # cycle bin -> count

    with openf(path) as f:
        for line in f:
            m = re_trace.match(line)
            if m:
                cur_cycle = int(m.group(1))
                max_cycle = max(max_cycle, cur_cycle)
                rest = m.group(2)
                dm = re_dxa.match(rest)
                if dm:
                    worker = (dm.group(1), int(dm.group(2)))
                    if dm.group(3) == 'start':
                        dxa_open[worker] = (cur_cycle, dm.group(4)[:120])
                    else:
                        if worker in dxa_open:
                            s, d = dxa_open.pop(worker)
                            dxa_spans.append((s, cur_cycle, dm.group(1), d))
                    continue
                tm = re_tcu.match(rest)
                if tm:
                    tcu_exec_cycles[cur_cycle] += 1
                    tcu_ops[tm.group(2)] += 1
                    continue
                sm = re_sched.search(rest)
                if sm:
                    sched_events[cur_cycle // bin_size] += 1
                continue
            m = re_susp.match(line)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                warp_susp[key] = (cur_cycle, int(m.group(3)))
                continue
            m = re_resu.match(line)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                if key in warp_susp:
                    s, bar = warp_susp.pop(key)
                    susp_spans.append((s, cur_cycle, key[0], key[1], bar))
                continue
            if re_barr.match(line):
                barrier_arrivals += 1

    # busy bitmaps
    n = max_cycle + 1
    dxa_busy = bytearray(n)
    for s, e, w, d in dxa_spans:
        for c in range(s, min(e, n - 1) + 1):
            dxa_busy[c] = 1
    tcu_busy = bytearray(n)
    for c in tcu_exec_cycles:
        if c < n: tcu_busy[c] = 1

    both = dxa_only = tcu_only = neither = 0
    for c in range(n):
        d, t = dxa_busy[c], tcu_busy[c]
        if d and t: both += 1
        elif d: dxa_only += 1
        elif t: tcu_only += 1
        else: neither += 1

    # binned timeline
    bins = (n + bin_size - 1) // bin_size
    tl = []
    for b in range(bins):
        lo, hi = b * bin_size, min((b + 1) * bin_size, n)
        db = sum(dxa_busy[lo:hi]); tb = sum(tcu_busy[lo:hi])
        ub = sum(tcu_exec_cycles.get(c, 0) for c in range(lo, hi))
        tl.append({"bin": b, "cycle": lo, "dxa": db, "tcu": tb, "uops": ub})

    # suspended warp-cycles
    susp_total = sum(e - s for s, e, _, _, _ in susp_spans)
    susp_by_bar = defaultdict(int)
    for s, e, _, _, bar in susp_spans:
        susp_by_bar[bar] += e - s

    dxa_busy_total = sum(dxa_busy)
    dxa_span_durs = [e - s for s, e, _, _ in dxa_spans]
    summary = {
        "trace": path,
        "total_cycles": n,
        "dxa_transfers": len(dxa_spans),
        "dxa_busy_cycles": dxa_busy_total,
        "dxa_busy_pct": round(100 * dxa_busy_total / n, 1),
        "dxa_span_avg": round(sum(dxa_span_durs) / max(1, len(dxa_span_durs)), 1),
        "dxa_span_min": min(dxa_span_durs, default=0),
        "dxa_span_max": max(dxa_span_durs, default=0),
        "tcu_busy_cycles": sum(tcu_busy),
        "tcu_busy_pct": round(100 * sum(tcu_busy) / n, 1),
        "tcu_uops": sum(tcu_exec_cycles.values()),
        "tcu_ops": dict(tcu_ops),
        "overlap_both": both, "dxa_only": dxa_only,
        "tcu_only": tcu_only, "neither": neither,
        "overlap_pct_of_dxa": round(100 * both / max(1, dxa_busy_total), 1),
        "barrier_arrivals": barrier_arrivals,
        "warp_suspend_spans": len(susp_spans),
        "warp_suspended_cycles_total": susp_total,
        "avg_suspended_warps": round(susp_total / n, 2),
        "susp_by_bar": {str(k): v for k, v in sorted(susp_by_bar.items())},
        "timeline_bin_size": bin_size,
        "timeline": tl,
        "dxa_first_spans": [(s, e, d) for s, e, _, d in dxa_spans[:40]],
    }
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=1)
    for k, v in summary.items():
        if k in ("timeline", "dxa_first_spans", "trace"): continue
        print(f"{k}: {v}")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 256)
