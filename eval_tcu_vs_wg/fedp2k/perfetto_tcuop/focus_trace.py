#!/usr/bin/env python3
"""Make a rendering-focused copy of a merged trace. Screenshots only; the full
merged trace keeps every event.

1. Drops stock tracks that are illegible at kernel scale and redundant with the
   engine layer: per-bank cache/MSHR raw tracks, the ~327k per-bank LMEM
   instants (the engine layer's binned 'LMEM read lanes' counter carries that
   information), the coalescer raw track, and the l2/l3 pass-through
   duplicates of DRAM traffic. The icache track is KEPT: the kernel is fully
   unrolled, so every tile runs on never-fetched lines and compulsory icache
   misses are on the critical path.
2. Drops every warp other than warp 0. TCU_OP runs one self-managed warp; the
   other warps' events all sit at cycle 1 with random PCs and thread masks --
   X-initialisation noise during reset, not execution.
3. Re-keys instruction lifetimes. ci/perfetto.py emits them as JSON async
   events named "inst" with a global id, which Perfetto files under "Global
   Legacy Events" as anonymous stacked bars. Here each gets its opcode class as
   its name and a process-local id, so Perfetto groups them into one track per
   opcode class (MMA_OP, BAR wait, DXA.ISSUE, ...) under Vortex GPU.
"""
import json, re, sys
src, dst = sys.argv[1], sys.argv[2]
ev = json.load(open(src))["traceEvents"]
names = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev
         if e.get("ph") == "M" and e.get("name") == "thread_name"}
DROP = re.compile(r"(-mshr|-bank\d|coalescer|: unknown|: l2|: l3|: warp(?!0\b)\d+)")
drop = {k for k, n in names.items() if DROP.search(n)}

op_of = {}
for e in ev:
    if e.get("cat") == "vortex.stage":
        a = e.get("args", {})
        if a.get("op") and a.get("uuid") is not None:
            op_of.setdefault(a["uuid"], a["op"])

def klass(op):
    if op is None: return "inst (unknown op)"
    if op.startswith("MMA_OP"): return "MMA_OP (warp-side lifetime)"
    if op == "BAR": return "BAR (barrier wait)"
    if op.startswith("BAR."): return op
    if op.startswith("DXA"): return "DXA.ISSUE"
    if op == "WGATHER": return "WGATHER (descriptor pack)"
    if op in ("LW", "SW", "LSU"): return "LW/SW"
    return "scalar ALU/branch/CSR"

# 4. Drops instruction slices ci/perfetto.py could not close (no commit seen);
#    its finalize() ends them at the last timestamp with args.incomplete=1, so
#    each renders as a bar spanning the whole kernel. 5 of 674 in this trace.
incomplete = {e.get("id") for e in ev if e.get("ph") == "e" and e.get("args", {}).get("incomplete")}

out = []
for e in ev:
    if (e.get("pid"), e.get("tid")) in drop:
        continue
    if e.get("ph") in ("b", "e") and e.get("id") in incomplete:
        continue
    if e.get("ph") in ("b", "e") and e.get("cat") == "vortex.inst":
        uid = e.get("id")
        e = dict(e)
        e["name"] = klass(op_of.get(uid))
        e.pop("id", None)
        e["id2"] = {"local": str(uid)}
    out.append(e)
kept = sorted({n for k, n in names.items() if k not in drop})
json.dump({"traceEvents": out, "displayTimeUnit": "ns"}, open(dst, "w"))
print(f"kept {len(out)}/{len(ev)} events; tracks kept: {kept}")
