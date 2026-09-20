#!/usr/bin/env python3
"""Companion to ci/perfetto.py for the TCU_OP (SparseWeaver) engine.

ci/perfetto.py turns pipeline/warp/cache trace lines into Perfetto tracks, but it
keys everything on instruction UUIDs and module paths. The TCU_OP engine's own
TRACE lines (VX_tcu_op_core.sv) carry neither, so the stock exporter drops them
all -- and the engine is where a TCU_OP kernel spends its time. This script
parses those lines and emits a second Perfetto process ("TCU_OP engine"),
then merges it with the stock export so both share one time axis.

Time axis: rtlsim prints a tick that advances twice per clock (processor.cpp
eval() runs once per clock edge). With --tick-ns 0.5 (the default) one core
cycle renders as 1 ns, so every duration in the UI reads directly as cycles.
The stock export must be produced with the same scale: --cycle-ns 0.5.

Engine lifecycle per MMA_OP (one 32x32 output tile), from the RTL TRACE sites:
  "TCU execution fired"      op accepted by the engine            (L1, :1641)
  "LMEM: Issuing read ..."   operand fetch request, matrix=A/B/C  (L2, :1689)
  "LMEM: read rsp ..."       operand fetch response               (L2, :1697)
  "Issue Busy processing"    one per cycle while issuing steps    (L1, :1663)
  "All iterations issued"    last FEOP step issued                (L2, :1672)
  "Flushing ..." / "LMEM: Issuing write request"  D writeback     (L2, :1746/1749)
  "Result fired downstream"  op retired back to the core          (L1, :1675)
  "execute stall ..."        an MMA_OP is presented but not taken (L1, :1371)
Phases drawn:  fetch (fire -> first issue), issue (first -> last issue),
               drain (last issue -> result), idle (result -> next fire).
"""
import argparse, gzip, json, re, sys
from collections import defaultdict

TICK_RE = re.compile(r"^\s*(\d+):\s*(.*)$")

def us(tick, tick_ns):
    return tick * tick_ns / 1000.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--stock", help="ci/perfetto.py output (.json/.json.gz) to merge with")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--tick-ns", type=float, default=0.5)
    ap.add_argument("--summary", help="write per-tile phase table (TSV) here")
    a = ap.parse_args()
    T = a.tick_ns

    tiles = []            # dicts: fire, first_issue, last_issue, result, stall_first
    cur = None
    issue_ticks = []      # every Issue Busy tick
    set_samples = []      # (tick, set) progress within the tile
    rd_req = defaultdict(list); rd_rsp = defaultdict(list)   # matrix -> [ticks]
    wr_req = []
    flush_ticks = []
    exec_stall = []
    txbar = []
    fullq = []
    fire_addrs = []       # per tile: dict of A/B/C/D addr, from the line after "TCU execution fired"
    pending_addr_line = False
    # DXA (needs -DDBG_TRACE_DXA): per-worker transfer lifetimes
    dxa_req = []          # (tick, smem, meta)
    dxa_issue = []        # (tick, worker, meta)
    dxa_done = []         # (tick, worker_instance, bar, wr_count)
    dxa_setup = []        # (tick, worker_instance): worker actually starts the transfer
    # binned bandwidth counters: name -> list of ticks
    BIN_PATTERNS = {
        "LMEM read lanes (engine operand fetch)": re.compile(r"-lmem core-rd-req\["),
        "DRAM reads (DXA staging + icache refills)": re.compile(r"^MEM Rd Req\["),
        "DRAM writes (dcache evicting dirty D lines)": re.compile(r"^MEM Wr Req\["),
        "dcache writes (D writeback to global)": re.compile(r"-dcache\d+ core-wr-req\["),
    }
    binned = defaultdict(list)
    # icache misses: tags-miss (uuid) -> first core-rd-rsp for that uuid
    ic_miss = {}          # uuid -> (tick, addr)
    ic_done = {}          # uuid -> tick

    opener = gzip.open if a.log.endswith(".gz") else open
    with opener(a.log, "rt", errors="replace") as f:
        for line in f:
            # the operand addresses follow "TCU execution fired" on an untimed line
            if pending_addr_line and line.startswith("A_addr="):
                fire_addrs[-1] = {k: int(v, 16) for k, v in re.findall(r"(\w)_addr=0x([0-9a-fA-F]+)", line)}
                pending_addr_line = False
                continue
            m = TICK_RE.match(line)
            if not m:
                continue
            t = int(m.group(1)); body = m.group(2)
            for name, rx in BIN_PATTERNS.items():
                if rx.search(body):
                    binned[name].append(t)
            if "-icache" in body:
                mu = re.search(r"\(#(\d+)\)", body)
                if mu:
                    if " tags-miss:" in body:
                        ma = re.search(r"addr=(0x[0-9a-fA-F]+)", body)
                        ic_miss.setdefault(mu.group(1), (t, ma.group(1) if ma else "?"))
                    elif " core-rd-rsp[" in body and mu.group(1) in ic_miss:
                        ic_done.setdefault(mu.group(1), t)
                continue
            if " dxa-req: " in body:
                mm = re.search(r"smem=0x([0-9a-fA-F]+), meta=0x([0-9a-fA-F]+)", body)
                if mm: dxa_req.append((t, int(mm.group(1), 16), mm.group(2)))
                continue
            if " dispatch-issue: " in body:
                mm = re.search(r"worker=(\d+).*meta=0x([0-9a-fA-F]+)", body)
                if mm: dxa_issue.append((t, int(mm.group(1)), mm.group(2)))
                continue
            if " setup-done: " in body and "dxa" in body.split(" ", 1)[0].lower():
                dxa_setup.append((t, body.split(" ", 1)[0]))
                continue
            if " done: core=" in body and "dxa" in body.split(" ", 1)[0].lower():
                inst = body.split(" ", 1)[0]
                mm = re.search(r"bar=(\d+), wr_count=(\d+)", body)
                dxa_done.append((t, inst, mm.group(1) if mm else "?", mm.group(2) if mm else "?"))
                continue
            if "[tcu_op_core] TCU execution fired" in body:
                cur = {"fire": t, "first_issue": None, "last_issue": None,
                       "result": None, "issue_cycles": 0}
                tiles.append(cur)
                fire_addrs.append({})
                pending_addr_line = True
            elif "[tcu_op_core] Issue Busy processing" in body:
                issue_ticks.append(t)
                if cur is not None:
                    if cur["first_issue"] is None:
                        cur["first_issue"] = t
                    cur["issue_cycles"] += 1
            elif body.startswith("c_blk_idx="):
                ms = re.search(r"set=(\d+)", body)
                if ms: set_samples.append((t, int(ms.group(1))))
            elif "[tcu_op_core] All iterations issued" in body:
                if cur is not None: cur["last_issue"] = t
            elif "[tcu_op_core] Result fired downstream" in body:
                if cur is not None: cur["result"] = t
            elif "[tcu_op_core] Full-queue stall cycles=" in body:
                fullq.append((t, int(body.rsplit("=", 1)[1])))
            elif "LMEM: Issuing read request" in body:
                mm = re.search(r"matrix=(\w+)", body)
                rd_req[mm.group(1) if mm else "?"].append(t)
            elif "LMEM: read rsp" in body:
                mm = re.search(r"matrix=(\w+)", body)
                rd_rsp[mm.group(1) if mm else "?"].append(t)
            elif "LMEM: Issuing write request" in body:
                wr_req.append(t)
            elif "[tcu_op_core]: Flushing" in body:
                flush_ticks.append(t)
            elif "[tcu_op_core]: execute stall" in body:
                exec_stall.append(t)
            elif "[tcu_op_core-txbar]" in body and ("START fire" in body or "DONE fire" in body):
                txbar.append((t, "START" if "START" in body else "DONE", body.split("] ", 1)[1].strip()))

    PID = 2
    ev = [
        {"ph": "M", "pid": PID, "name": "process_name", "args": {"name": "TCU_OP analysis layer (engine block 0, DXA, icache)"}},
        {"ph": "M", "pid": PID, "name": "process_sort_index", "args": {"sort_index": 0}},
    ]
    names = {1: "1 MMA_OP (tile) lifetime", 2: "2 engine phase", 3: "3 issue busy (1 = FEOP step issued)",
             4: "4 k-progress (set)", 6: "6 D writeback",
             7: "7 MMA_OP waiting at engine (execute stall)", 8: "8 txbar"}
    for tid, n in names.items():
        ev.append({"ph": "M", "pid": PID, "tid": tid, "name": "thread_name", "args": {"name": n}})
        ev.append({"ph": "M", "pid": PID, "tid": tid, "name": "thread_sort_index", "args": {"sort_index": tid}})

    def X(tid, name, t0, t1, args=None, cname=None):
        if t0 is None or t1 is None or t1 < t0: return
        e = {"ph": "X", "pid": PID, "tid": tid, "name": name, "ts": us(t0, T),
             "dur": max(us(t1, T) - us(t0, T), 0.0005), "args": args or {}}
        if cname: e["cname"] = cname
        ev.append(e)

    rows = []
    for i, tl in enumerate(tiles):
        fire, fi, li, res = tl["fire"], tl["first_issue"], tl["last_issue"], tl["result"]
        nxt = tiles[i + 1]["fire"] if i + 1 < len(tiles) else None
        cyc = lambda x, y: (y - x) / 2 if (x is not None and y is not None) else None
        r = {"tile": i, "fire_cyc": fire / 2,
             "fetch": cyc(fire, fi), "issue": cyc(fi, li), "drain": cyc(li, res),
             "idle_to_next": cyc(res, nxt), "issue_cycles": tl["issue_cycles"],
             "op_total": cyc(fire, res)}
        rows.append(r)
        X(1, f"tile {i}", fire, res, {k: v for k, v in r.items()})
        X(2, "fetch operands", fire, fi, {"cycles": r["fetch"]}, "thread_state_iowait")
        X(2, "issue (compute)", fi, li, {"cycles": r["issue"], "issue_cycles": tl["issue_cycles"]}, "thread_state_running")
        X(2, "drain + D writeback", li, res, {"cycles": r["drain"]}, "thread_state_runnable")
        if nxt is not None:
            X(2, "IDLE (no MMA_OP)", res, nxt, {"cycles": r["idle_to_next"]}, "thread_state_sleeping")

    # issue busy as a 0/1 counter, transitions only
    prev = None
    for t in issue_ticks:
        if prev is None or t - prev > 2:
            if prev is not None:
                ev.append({"ph": "C", "pid": PID, "tid": 3, "name": "issue busy", "ts": us(prev + 2, T), "args": {"busy": 0}})
            ev.append({"ph": "C", "pid": PID, "tid": 3, "name": "issue busy", "ts": us(t, T), "args": {"busy": 1}})
        prev = t
    if prev is not None:
        ev.append({"ph": "C", "pid": PID, "tid": 3, "name": "issue busy", "ts": us(prev + 2, T), "args": {"busy": 0}})

    # k-progress, decimated to changes of set
    last = None
    for t, s in set_samples:
        if s != last:
            ev.append({"ph": "C", "pid": PID, "tid": 4, "name": "k set", "ts": us(t, T), "args": {"set": s}})
            last = s

    # Operand fetch requests per matrix, binned. NOT a req-minus-rsp in-flight
    # count: the response TRACE (:1697) prints on every cycle a response is
    # presented, so a held response prints repeatedly (tile 0: 64 B requests,
    # 80 B response lines) and a difference would drift. Requests are 1:1.
    for mtx, ts in rd_req.items():
        binned[f"LMEM operand requests: {mtx}"] = ts

    # D writeback: contiguous write bursts
    def bursts(ts, gap):
        out = []
        for t in ts:
            if out and t - out[-1][1] <= gap: out[-1][1] = t
            else: out.append([t, t])
        return out
    for b0, b1 in bursts(wr_req, 8):
        X(6, "D write", b0, b1 + 2, {"cycles": (b1 + 2 - b0) / 2})
    for b0, b1 in bursts(exec_stall, 4):
        X(7, "MMA_OP waiting", b0, b1 + 2, {"cycles": (b1 + 2 - b0) / 2}, "bad")
    for t, kind, txt in txbar:
        ev.append({"ph": "i", "s": "t", "pid": PID, "tid": 8, "name": f"txbar {kind}", "ts": us(t, T), "args": {"detail": txt}})

    # ---- icache misses: the fetch of warp 0 is blocked for the whole refill ----
    ev.append({"ph": "M", "pid": PID, "tid": 5, "name": "thread_name", "args": {"name": "5 icache miss -> refill (warp 0 fetch blocked)"}})
    ev.append({"ph": "M", "pid": PID, "tid": 5, "name": "thread_sort_index", "args": {"sort_index": 5}})
    for u, (t0, addr) in sorted(ic_miss.items(), key=lambda kv: kv[1][0]):
        t1 = ic_done.get(u)
        if t1 is None or t1 < t0: continue
        # on the critical path iff it overlaps an engine IDLE window
        crit = any(tl["result"] is not None and i + 1 < len(tiles) and
                   t0 < tiles[i + 1]["fire"] and t1 > tl["result"] for i, tl in enumerate(tiles))
        X(5, f"icache miss {addr}" + (" [ENGINE IDLE]" if crit else ""), t0, t1,
          {"addr": addr, "cycles": (t1 - t0) / 2, "engine_idle_overlap": crit},
          "terrible" if crit else "generic_work")

    # ---- binned bandwidth counters (per-cycle rate over BIN-cycle windows) ----
    BIN = 64  # cycles
    for k, (name, ticks) in enumerate(sorted(binned.items())):
        tid = 9 + k
        ev.append({"ph": "M", "pid": PID, "tid": tid, "name": "thread_name", "args": {"name": f"{tid} {name}"}})
        ev.append({"ph": "M", "pid": PID, "tid": tid, "name": "thread_sort_index", "args": {"sort_index": tid}})
        hist = defaultdict(int)
        for t in ticks:
            hist[(t // 2) // BIN] += 1
        if not hist: continue
        lo, hi = min(hist), max(hist)
        for b in range(lo, hi + 2):
            ev.append({"ph": "C", "pid": PID, "tid": tid, "name": name, "ts": us(b * BIN * 2, T),
                       "args": {"per_cycle": round(hist.get(b, 0) / BIN, 3)}})

    # ---- DXA transfers ----
    # A worker has a one-deep hand-off slot: dispatch-issue queues a transfer,
    # the worker STARTS it at setup-done (e.g. A dispatched at tick 2321 starts
    # at 3941, right after C's done at 3939) and finishes at done. Service
    # intervals (setup-done -> done) are therefore back-to-back and never
    # overlap; dispatch -> setup-done is queueing and is drawn separately.
    # Transfers on one worker complete in dispatch order, so all three event
    # streams are paired FIFO per worker.
    # Each transfer is labelled by the LMEM buffer it fills, using the operand
    # addresses the engine later reports for the tile that consumes it.
    lmem_off = lambda a: a & 0xFFFF
    buf_name = {}
    for i, ad in enumerate(fire_addrs):
        for mtx in ("A", "B", "C"):
            if mtx in ad:
                buf_name.setdefault(lmem_off(ad[mtx]), f"{mtx}{i % 2}")
    req_by_meta = defaultdict(list)
    for t, smem, meta in dxa_req:
        req_by_meta[meta].append((t, smem))
    wnum = lambda inst: int(re.findall(r"(\d+)", inst)[-1]) if re.findall(r"(\d+)", inst) else 0
    done_by_w = defaultdict(list)
    for t, inst, bar, wr in dxa_done:
        done_by_w[wnum(inst)].append((t, bar, wr))
    setup_by_w = defaultdict(list)
    for t, inst in dxa_setup:
        setup_by_w[wnum(inst)].append(t)
    issue_by_w = defaultdict(list)
    for t, w, meta in dxa_issue:
        smem = req_by_meta[meta].pop(0)[1] if req_by_meta[meta] else None
        issue_by_w[w].append((t, smem))
    fires = [tl["fire"] for tl in tiles]
    dxa_rows = []
    for w in sorted(set(issue_by_w) | set(done_by_w)):
        tid = 20 + w
        qtid = 30 + w
        ev.append({"ph": "M", "pid": PID, "tid": tid, "name": "thread_name", "args": {"name": f"DXA worker {w}: transfer in service"}})
        ev.append({"ph": "M", "pid": PID, "tid": tid, "name": "thread_sort_index", "args": {"sort_index": tid}})
        ev.append({"ph": "M", "pid": PID, "tid": qtid, "name": "thread_name", "args": {"name": f"DXA worker {w}: transfer queued (waiting for worker)"}})
        ev.append({"ph": "M", "pid": PID, "tid": qtid, "name": "thread_sort_index", "args": {"sort_index": tid + 1}})
        for (td, smem), ts, (t1, bar, wr) in zip(issue_by_w[w], setup_by_w[w], done_by_w[w]):
            bn = buf_name.get(lmem_off(smem), f"0x{smem:x}") if smem is not None else "?"
            # consumer = first tile fired after this transfer completes that reads this buffer
            cons = next((j for j, (ft, ad) in enumerate(zip(fires, fire_addrs))
                         if ft >= t1 and smem is not None and lmem_off(smem) in {lmem_off(v) for v in ad.values()}), None)
            label = f"{bn[0]} for tile {cons}" if cons is not None else f"{bn} (unconsumed)"
            slack = (fires[cons] - t1) / 2 if cons is not None else None
            X(tid, label, ts, t1, {"buffer": bn, "bar": bar, "wr_count": wr, "service_cycles": (t1 - ts) / 2,
                                   "queued_cycles": (ts - td) / 2, "consumer_tile": cons,
                                   "slack_before_consumer_fires": slack})
            if ts > td:
                X(qtid, f"{bn[0]} queued", td, ts, {"cycles": (ts - td) / 2})
            dxa_rows.append((w, bn, cons, td / 2, ts / 2, t1 / 2, (t1 - ts) / 2, slack))
    if dxa_rows:
        print("\nDXA transfers (worker, buffer, consumer_tile, dispatch_cyc, start_cyc, end_cyc, service, slack_to_consumer_fire):")
        for r in sorted(dxa_rows, key=lambda r: r[3]):
            print("  ", r)

    if a.stock:
        so = gzip.open if a.stock.endswith(".gz") else open
        with so(a.stock, "rt") as f:
            sd = json.load(f)
        sev = sd["traceEvents"] if isinstance(sd, dict) else sd
        ev = sev + ev

    oo = gzip.open if a.output.endswith(".gz") else open
    with oo(a.output, "wt") as f:
        json.dump({"traceEvents": ev, "displayTimeUnit": "ns"}, f)

    # ---- summary ----
    keys = ["tile", "fire_cyc", "fetch", "issue", "drain", "idle_to_next", "op_total", "issue_cycles"]
    lines = ["\t".join(keys)] + ["\t".join("" if r[k] is None else f"{r[k]:g}" for k in keys) for r in rows]
    if a.summary:
        open(a.summary, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    tot = lambda k: sum(r[k] for r in rows if r[k] is not None)
    print(f"\ntiles={len(tiles)}  sum: fetch={tot('fetch'):g} issue={tot('issue'):g} "
          f"drain={tot('drain'):g} idle_between={tot('idle_to_next'):g} issue_cycles={tot('issue_cycles'):g}")
    if tiles:
        print(f"first fire cyc={tiles[0]['fire']/2:g}  last result cyc={(tiles[-1]['result'] or 0)/2:g}")
    print("reads per matrix:", {k: len(v) for k, v in rd_req.items()},
          " writes:", len(wr_req), " exec_stall lines:", len(exec_stall),
          " full-queue stalls:", sum(c for _, c in fullq))

if __name__ == "__main__":
    main()
