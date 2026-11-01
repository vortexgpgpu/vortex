#!/usr/bin/env python3
"""Summarize a Beacon run (foldsim-estimate-timing, foldsim-estimate-area)
for logic_gate.

Beacon writes one row per end point bit: the logic levels of its longest
path, its start and its end; and one row per instance: its resources and its
subtree's. ci/logic_gate.py gates depth per RTL unit and the design's area,
so this collapses the tables into what it compares:

  beacon_summary.csv  one row: the deepest end point, the end point bits at or
                      above the fix and watch thresholds, all end points, and
                      the design's LUTs, flip-flops, distributed RAM LUTs,
                      block RAMs, UltraRAMs and DSPs (area.csv)
  beacon_units.json   the deepest end point of every unit at or above the
                      watch threshold, the deepest paths (diagnostics), and
                      `area`: each unit's resources, one copy's subtree
  beacon_paths.txt    if Beacon wrote paths.jsonl: every path at or above
                      the watch threshold, op by op, by unit (write_paths)
  beacon_area.txt     if Beacon wrote area.csv: the area summary, the units
                      by LUTs, and where each memory goes (write_area)

If Beacon wrote ports.csv (BEACON_INTERFACES), beacon_units.json also holds
`interfaces`: per output port of those modules' instances, folded like
units, the bits a combinational path from outside the instance reaches.

A unit is an end point's instance path with generate indices folded
(g_banks_3_bank -> g_banks_*_bank), so one deep bank reads as one unit.

  usage: beacon_summary.py <run dir> --top <module> [--output-dir <dir>]
"""

import argparse
import collections
import csv
import json
import os
import re

# 300 MHz on the V80: failing end points have a median of 7-8 levels, and
# about 55-60% of the registers at 12 or more fail
# (docs/designs/beacon.md §7).
FIX_LEVELS = 12
WATCH_LEVELS = 8
NUM_PATHS = 10

COLUMNS = ("top", "max_levels", "deep_endpoints", "watch_endpoints",
           "endpoints")
AREA = ("lut", "ff", "lutram", "bram", "uram", "dsp")
# Units of at least this share of the design's LUTs are listed in
# beacon_area.txt.
AREA_UNIT_SHARE = 0.01

INDEX = re.compile(r"_\d+(?=_|/|$)")


def norm(name):
    return name.replace(".", "_").replace("[", "_").replace("]", "")


def instance_paths(path):
    """Instance id -> path from the top, from a run's instances.csv."""
    rows = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            rows[int(row["id"])] = (int(row["parent"]), row["instance"])
    paths = {0: ""}

    def get(ctx):
        if ctx not in paths:
            parent, name = rows[ctx]
            p = get(parent)
            paths[ctx] = "%s/%s" % (p, name) if p else name
        return paths[ctx]

    for ctx in rows:
        get(ctx)
    return paths


def unit(path):
    """The unit of an end point: its directory, generate indices folded."""
    return INDEX.sub("_*", norm(path).rpartition("/")[0]) or "<top>"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--top", required=True)
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    paths = instance_paths(os.path.join(args.run_dir, "instances.csv"))
    units = collections.defaultdict(int)
    counts = collections.Counter()
    deepest = []
    with open(os.path.join(args.run_dir, "endpoints.csv")) as fh:
        for row in csv.DictReader(fh):
            levels = int(row["levels"])
            counts["endpoints"] += 1
            if levels < WATCH_LEVELS:
                continue
            counts["watch_endpoints"] += 1
            counts["deep_endpoints"] += levels >= FIX_LEVELS
            ip = paths[int(row["end_inst"])]
            end = "%s/%s" % (ip, row["end"]) if ip else row["end"]
            u = unit(end)
            units[u] = max(units[u], levels)
            sp = paths[int(row["start_inst"])]
            deepest.append((levels, "%s/%s" % (sp, row["start"]) if sp
                            else row["start"], end))
            # Keep the list short: one sort at the end of a few thousand rows.
            if len(deepest) > 50 * NUM_PATHS:
                deepest = sorted(deepest, key=lambda d: -d[0])[:NUM_PATHS]
    deepest = sorted(deepest, key=lambda d: -d[0])[:NUM_PATHS]

    summary = {"top": args.top,
               "max_levels": max(units.values(), default=0),
               "deep_endpoints": counts["deep_endpoints"],
               "watch_endpoints": counts["watch_endpoints"],
               "endpoints": counts["endpoints"]}
    if not units:
        # Nothing reaches the watch threshold; the deepest is still recorded.
        with open(os.path.join(args.run_dir, "endpoints.csv")) as fh:
            summary["max_levels"] = max(
                (int(r["levels"]) for r in csv.DictReader(fh)), default=0)

    area = read_area(args.run_dir)
    columns = COLUMNS
    if area:
        summary.update(area["total"])
        columns = COLUMNS + AREA
    with open(os.path.join(args.output_dir, "beacon_summary.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerow(summary)
    doc = {"fix_levels": FIX_LEVELS, "watch_levels": WATCH_LEVELS,
           "units": dict(sorted(units.items())),
           "deepest_paths": ["%d  %s -> %s" % d for d in deepest]}
    if area:
        doc["area"] = area["units"]
    ports = os.path.join(args.run_dir, "ports.csv")
    if os.path.exists(ports):
        doc["interfaces"] = interfaces(ports)
        print("beacon: %d interface outputs reached combinationally from "
              "outside" % len(doc["interfaces"]))
    with open(os.path.join(args.output_dir, "beacon_units.json"), "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    print("beacon: %s max %d levels, %d end point bits >= %d, %d units >= %d"
          % (args.top, summary["max_levels"], summary["deep_endpoints"],
             FIX_LEVELS, len(units), WATCH_LEVELS))
    if os.path.exists(os.path.join(args.run_dir, "paths.jsonl")):
        with open(os.path.join(args.output_dir, "beacon_paths.txt"), "w") as fh:
            n, u = write_paths(fh, args.run_dir, WATCH_LEVELS, paths=paths)
        print("beacon: %d paths >= %d levels in %d units -> beacon_paths.txt"
              % (n, WATCH_LEVELS, u))
    if area:
        with open(os.path.join(args.output_dir, "beacon_area.txt"), "w") as fh:
            write_area(fh, args.run_dir, area)
        t = area["total"]
        print("beacon: %s %d LUTs (%d distributed RAM), %d FFs, %g block "
              "RAMs, %d UltraRAMs, %d DSPs -> beacon_area.txt"
              % (args.top, t["lut"], t["lutram"], t["ff"], t["bram"],
                 t["uram"], t["dsp"]))


def read_area(run_dir):
    """The design's resources and each unit's from a run's area.csv, or None.
    `lut` counts every LUT, distributed RAM included, as Vivado's utilization
    does. A unit's resources are one copy's subtree; `copies` counts its
    instances."""
    path = os.path.join(run_dir, "area.csv")
    if not os.path.exists(path):
        return None
    total, units = None, {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            res = {k: float(row["total_" + k]) for k in AREA}
            res["lut"] += res["lutram"]
            res = {k: (round(v, 1) if k == "bram" else int(round(v)))
                   for k, v in res.items()}
            if total is None:
                total = res
                continue
            u = fold(row["path"])
            if u in units:
                units[u]["copies"] += 1
            else:
                units[u] = dict(res, module=row["module"], copies=1)
    return {"total": total, "units": dict(sorted(units.items()))}


def write_area(out, run_dir, area):
    """The area report, the units holding at least AREA_UNIT_SHARE of the
    LUTs, and every memory with its placement."""
    with open(os.path.join(run_dir, "area.txt")) as fh:
        out.write(fh.read())
    t = area["total"]
    floor = AREA_UNIT_SHARE * t["lut"]
    out.write("\nUnits with at least %g%% of the LUTs, one copy's subtree "
              "(x copies):\n" % (100 * AREA_UNIT_SHARE))
    out.write("  %9s %8s %7s %7s %5s %5s  %s\n"
              % ("lut", "ff", "lutram", "bram", "uram", "dsp", "unit"))
    rows = sorted(area["units"].items(), key=lambda kv: -kv[1]["lut"])
    for u, r in rows:
        if r["lut"] * r["copies"] < floor:
            continue
        out.write("  %9d %8d %7d %7g %5d %5d  %s (%s) x%d\n"
                  % (r["lut"], r["ff"], r["lutram"], r["bram"], r["uram"],
                     r["dsp"], u, r["module"], r["copies"]))
    mems = os.path.join(run_dir, "area_memories.csv")
    if not os.path.exists(mems):
        return
    out.write("\nMemories (depth x width, read latency, ram_style -> "
              "placement and cost, x copies):\n")
    groups = collections.OrderedDict()
    with open(mems) as fh:
        for r in csv.DictReader(fh):
            name = fold("%s/%s" % (r["path"], r["name"]) if r["path"]
                        else r["name"])
            key = (name, r["depth"], r["width"], r["latency"], r["style"],
                   r["placement"])
            if key in groups:
                groups[key][1] += 1
            else:
                groups[key] = [r, 1]
    for (name, depth, width, latency, style, placement), (r, n) in \
            groups.items():
        cost = ("%s LUTs" % r["lutram"] if placement == "lutram" else
                "%s blocks" % r["bram"] if placement == "bram" else
                "%s UltraRAMs" % r["uram"] if placement == "uram" else
                "%s FFs" % r["ff"])
        out.write("  %6sx%-5s lat %s %-11s -> %-6s %-12s %s x%d\n"
                  % (depth, width, latency, style, placement, cost, name, n))

def fold(name):
    return INDEX.sub("_*", norm(name))


def interfaces(path):
    """'<instance>.<port>' -> bits, for every output port of a kept module's
    instance that a combinational path from outside reaches; generate copies
    folded, keeping the most bits."""
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            bits = int(row["through"])
            if row["direction"] != "out" or not bits:
                continue
            key = "%s.%s" % (fold(row["instance"]), row["port"])
            out[key] = max(out.get(key, 0), bits)
    return dict(sorted(out.items()))


def write_paths(out, run_dir, min_levels, unit_filter=None, paths=None):
    """Write a run's deep paths (paths.jsonl) by unit, op by op.

    Beacon lists one path per pair of start and end registers; this folds
    the pairs whose names differ only in generate indices and bits, as units
    fold them, keeping the deepest. Returns (paths, units) written.
    """
    if paths is None:
        paths = instance_paths(os.path.join(run_dir, "instances.csv"))

    def full(ctx, name):
        return "%s/%s" % (paths[ctx], name) if paths[ctx] else name

    units = collections.defaultdict(dict)
    with open(os.path.join(run_dir, "paths.jsonl")) as fh:
        for line in fh:
            p = json.loads(line)
            if p["levels"] < min_levels:
                continue
            start = full(p["start_inst"], p["start"])
            end = full(p["end_inst"], p["end"])
            u = unit(end)
            if unit_filter and unit_filter not in u:
                continue
            group = units[u]
            key = (fold(start), fold(end))
            if key in group:
                group[key]["copies"] += 1
                group[key]["bits"] += p["bits"]
            else:
                group[key] = {"path": p, "copies": 1, "bits": p["bits"],
                              "start": start, "end": end}

    def deepest(u):
        return max(g["path"]["levels"] for g in units[u].values())

    order = sorted(units, key=lambda u: (-deepest(u), u))
    count = sum(len(g) for g in units.values())
    out.write("%d paths at or above %d levels in %d units, deepest unit "
              "first.\n" % (count, min_levels, len(units)))
    out.write("A path is its deepest end point bit, op by op from the end "
              "point back to its\nstart: levels, kind, fanout, source line, "
              "instance/name. Bits of the same pair\nof registers, and "
              "generate copies (_0, _1, ...), are folded into it.\n")
    for u in order:
        group = sorted(units[u].values(), key=lambda g: -g["path"]["levels"])
        out.write("\n== %s: %d levels, %d paths\n" % (u, deepest(u),
                                                       len(group)))
        for g in group:
            p = g["path"]
            retimed = (", %d after retiming" % p["retimed"]
                       if p["retimed"] != p["levels"] else "")
            copies = " in %d copies" % g["copies"] if g["copies"] > 1 else ""
            out.write("\n  %d levels%s, %d %s bits%s\n  from %s\n  to   %s\n"
                      % (p["levels"], retimed, g["bits"], p["kind"], copies,
                         g["start"], g["end"]))
            for levels, kind, fanout, ctx, name, source in p["ops"]:
                out.write("    %4d %-6s fo %-5d %-24s %s\n"
                          % (levels, kind, fanout, source, full(ctx, name)))
    return count, len(units)


if __name__ == "__main__":
    main()
