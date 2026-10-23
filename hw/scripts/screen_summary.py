#!/usr/bin/env python3
"""Summarize a timing screen (foldsim-estimate-timing) for the timing gate.

The screen writes one row per end point bit: the logic levels of its longest
path, its start and its end. ci/timing_gate.py gates depth per RTL unit, so
this collapses the table into what it compares:

  screen_summary.csv  one row: the deepest end point, the end point bits at or
                      above the fix and watch thresholds, and all end points
  screen_units.json   the deepest end point of every unit at or above the
                      watch threshold, and the deepest paths (diagnostics)

A unit is an end point's instance path with generate indices folded
(g_banks_3_bank -> g_banks_*_bank), so one deep bank reads as one unit.

  usage: screen_summary.py <screen dir> --top <module> [--output-dir <dir>]
"""

import argparse
import collections
import csv
import json
import os
import re

# 300 MHz on the V80: failing end points have a median of 7-8 levels, and
# about 55-60% of the registers at 12 or more fail (timing_screen_proposal.md).
FIX_LEVELS = 12
WATCH_LEVELS = 8
NUM_PATHS = 10

COLUMNS = ("top", "max_levels", "deep_endpoints", "watch_endpoints",
           "endpoints")

INDEX = re.compile(r"_\d+(?=_|/|$)")


def norm(name):
    return name.replace(".", "_").replace("[", "_").replace("]", "")


def instance_paths(path):
    """Instance id -> path from the top, from the screen's instances.csv."""
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
    ap.add_argument("screen")
    ap.add_argument("--top", required=True)
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    paths = instance_paths(os.path.join(args.screen, "instances.csv"))
    units = collections.defaultdict(int)
    counts = collections.Counter()
    deepest = []
    with open(os.path.join(args.screen, "endpoints.csv")) as fh:
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
        with open(os.path.join(args.screen, "endpoints.csv")) as fh:
            summary["max_levels"] = max(
                (int(r["levels"]) for r in csv.DictReader(fh)), default=0)

    with open(os.path.join(args.output_dir, "screen_summary.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerow(summary)
    with open(os.path.join(args.output_dir, "screen_units.json"), "w") as fh:
        json.dump({"fix_levels": FIX_LEVELS, "watch_levels": WATCH_LEVELS,
                   "units": dict(sorted(units.items())),
                   "deepest_paths": ["%d  %s -> %s" % d for d in deepest]},
                  fh, indent=2)
        fh.write("\n")
    print("screen: %s max %d levels, %d end point bits >= %d, %d units >= %d"
          % (args.top, summary["max_levels"], summary["deep_endpoints"],
             FIX_LEVELS, len(units), WATCH_LEVELS))


if __name__ == "__main__":
    main()
